"""Find a named object in the room: one smooth sweep, then look at it (F5, on request).

The sweep (idea from reachy_mini_conversation_app `tools/sweep_look.py`, Apache-2.0, Pollen
Robotics) is one long move: world yaw = SWEEP_YAW * sin(2*pi*u), with u eased from 0 to 1
over SWEEP_S. So the head goes front -> left -> right -> front, slows down by itself at both
ends, and starts and stops at zero speed. The body takes BODY_SHARE of every turn, so the head
stays within 40° of the body, and pitch stays at SWEEP_PITCH: the camera (about ±28° high) then
covers eye level and a table in one pass.

While it sweeps, YOLOX runs on frame after frame (whole frame + 3x3 tiles, ~130 ms, so at most
~10° of travel between checks). On a candidate the sweep brakes (its clock slows
exponentially, no jolt), the head turns to the candidate and checks it on a sharp still frame.
A lone confident hit is accepted, otherwise the vision model checks it (several candidates,
"my phone", a weak score). Accepted: centre it, describe it (vision model) and report, and
the agent asks the user "is that it?". Rejected, or the user says no (`resume`): that
direction is skipped and the sweep goes on where it braked. At most MAX_PASSES sweeps.

A spoken stop cancels the tool call (barge-in), which ends the sweep and frees the detector.
Angles are degrees; yaw + = left, pitch + = down (as in robot/motion.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from reachy_demo.audio.sound_events import describe, plan_turn, wrap
from reachy_demo.perception.head_pose import TAN_HALF_H
from reachy_demo.perception.objects import (
    COCO,
    MIN_SCORE,
    Ask,
    Detection,
    Frame,
    GrabFrame,
    ObjectDetector,
    choose,
    describe_box,
    detect_tiled,
)
from reachy_demo.robot.motion import (
    AXES,
    BUSY_LONG,
    LIMITS,
    NEUTRAL,
    Motion,
    Move,
    Pose,
    goto,
)

log = logging.getLogger(__name__)

SWEEP_S = 20.0  # default length (SEARCH_SWEEP_S); peak speed ~75°/s * 20 / length
MIN_SWEEP_S, MAX_SWEEP_S = 8.0, 120.0  # under 8 s blurs (peak ~190°/s)
SWEEP_YAW = 160.0  # body 120° + head 40°: only ~30° straight behind is never seen
BODY_SHARE = LIMITS["body_yaw"] / SWEEP_YAW
SWEEP_PITCH = 12.0  # camera sees ~-16°..+40°: eye level and table level at once
EASE_S = 0.6  # blend into the sweep (and back into it after a check)
BRAKE_TAU = 0.12  # the sweep clock slows as exp(-t/tau) on a candidate
BRAKE_S = 5 * BRAKE_TAU
SETTLE_S = 0.3  # a still, sharp frame for the check
SKIP_DEG = 12.0  # a rejected candidate's direction is ignored for the rest of the sweep
SPEED_DPS = 60.0  # turn to a candidate; smoothstep peaks at 1.5x
MIN_MOVE_S = 0.4
MAX_PASSES = 2  # full sweeps per search, resumes included
RESUME_S = 60.0  # "no, not that one" within this long resumes the search
SURE = 0.55  # one detection at least this confident needs no vision check


def _smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


def path(t: float, length_s: float = SWEEP_S) -> Pose:
    """The sweep's pose at path time t (s) of a sweep lasting `length_s`."""
    yaw = SWEEP_YAW * math.sin(2 * math.pi * _smooth(t / length_s))
    return {**NEUTRAL, "yaw": yaw, "body_yaw": yaw * BODY_SHARE, "pitch": SWEEP_PITCH}


def _ramp(x: float) -> float:
    """Path time gained x s after (re)starting: the speed eases in over EASE_S."""
    if x <= 0:
        return 0.0
    if x >= EASE_S:
        return x - EASE_S / 2
    r = x / EASE_S
    return EASE_S * (r**3 - r**4 / 2)


class Sweep:
    """The sweep's path clock, shared by the long moves that play it (start, resume)."""

    def __init__(self, length_s: float = SWEEP_S) -> None:
        self.length_s = length_s
        self.time = 0.0  # path time reached
        self._base = 0.0
        self._halt: tuple[float, float] | None = None  # (move time, path time) at the brake
        self._start: Pose | None = None
        self._last_t = 0.0

    @property
    def done(self) -> bool:
        return self.time >= self.length_s

    def move(self) -> Move:
        """A long move playing the rest of the sweep, blending in from the current pose."""
        self._base, self._halt, self._start, self._last_t = self.time, None, None, 0.0
        remaining = self.length_s - self.time + EASE_S
        return Move("search sweep", True, remaining, self._pose_at, NEUTRAL)

    def brake(self) -> None:
        if self._halt is None:
            self._halt = (-1.0, self.time)  # the move time is filled on the next tick

    def _pose_at(self, t: float, start: Pose) -> Pose:
        if self._start is None:
            self._start = dict(start)
        if self._halt is not None:
            if self._halt[0] < 0:  # brake from the previous tick, so no tick stands still
                self._halt = (self._last_t, self.time)
            t0, p0 = self._halt
            self.time = p0 + BRAKE_TAU * (1 - math.exp(-(t - t0) / BRAKE_TAU))
        else:
            self.time = min(self._base + _ramp(t), self.length_s)
        self._last_t = t
        on_path, blend = path(self.time, self.length_s), _smooth(t / EASE_S)
        return {k: self._start[k] + (on_path[k] - self._start[k]) * blend for k in AXES}


@dataclass
class _Run:
    """One search: kept after a find, so "no, not that one" can resume it."""

    label: str
    sweep: Sweep
    skipped: list[float] = field(default_factory=list)  # world yaws of rejected objects
    passes: int = 0  # full sweeps finished
    found_yaw: float | None = None
    at: float = 0.0  # when it was found


def height(pitch: float) -> str:
    if pitch > 12:
        return "low, below my eye level"
    if pitch < -12:
        return "high, above my eye level"
    return "at my eye level"


class Search:
    def __init__(
        self,
        motion: Motion,
        grab: GrabFrame,
        make_detector: Callable[[], ObjectDetector],
        ask: Ask | None = None,
        clock: Callable[[], float] = time.monotonic,
        settle_s: float = SETTLE_S,
        brake_s: float = BRAKE_S,
        sweep_s: float = SWEEP_S,
    ) -> None:
        if not MIN_SWEEP_S <= sweep_s <= MAX_SWEEP_S:
            raise ValueError(f"sweep must last {MIN_SWEEP_S:g}-{MAX_SWEEP_S:g} s, got {sweep_s}")
        self._sweep_s = sweep_s
        self._motion = motion
        self._grab = grab
        self._make_detector = make_detector
        self._ask = ask
        self._clock = clock
        self._settle_s = settle_s
        self._brake_s = brake_s
        self._last: _Run | None = None  # the last found search, for a resume

    async def find(self, target: str, label: str, resume: bool = False) -> dict[str, Any]:
        """Sweep for `target` (COCO class `label`); returns the tool result.

        `resume`: the user said the last find was the wrong object. Within RESUME_S, the same
        search goes on from where it stopped and skips that object; otherwise it starts over.
        At most MAX_PASSES sweeps per search, resumes included.
        """
        if label not in COCO:
            return {"error": f"unsupported_object: {target} is not one I can recognise"}
        if self._motion.dancing:
            return {"error": BUSY_LONG}
        last, self._last = self._last, None
        now = self._clock()
        resumed = resume and last is not None and last.label == label and now - last.at <= RESUME_S
        if resumed:
            assert last is not None
            run = last
            if run.found_yaw is not None:
                run.skipped.append(run.found_yaw)  # the user said it isn't this one
        else:
            run = _Run(label, Sweep(self._sweep_s))
        try:
            detector = await asyncio.to_thread(self._make_detector)
        except (OSError, ImportError, RuntimeError) as e:
            log.warning("object detector unavailable: %s", e)
            return {"error": f"find_unavailable: {e}"}
        t0 = self._clock()
        done = False
        try:
            self._submit(run.sweep.move())
            while True:
                if run.sweep.done:
                    run.passes += 1
                    if run.passes >= MAX_PASSES:
                        break
                    log.info("sweep %d done, sweeping again", run.passes)
                    await self._motion.settled()  # the finished sweep's move ends at rest
                    run.sweep = Sweep(self._sweep_s)
                    self._submit(run.sweep.move())
                if not self._motion.dancing:  # the sweep was ended from elsewhere
                    break
                seen_from = dict(self._motion.pose)  # the head when the frame is taken
                frame = await asyncio.to_thread(self._grab)
                if frame is None:
                    return {"error": "no camera frame (camera off or not connected)"}
                hits = await asyncio.to_thread(detect_tiled, detector, frame, label)
                hits = [
                    d
                    for d in hits
                    if d.score >= MIN_SCORE
                    and all(
                        abs(wrap(_aim(frame, d, seen_from)[0] - s)) > SKIP_DEG for s in run.skipped
                    )
                ]
                if not hits:
                    continue
                # Candidate: brake, turn to it, check it on a sharp still frame.
                best = max(hits, key=lambda d: d.score)
                yaw, pitch = _aim(frame, best, seen_from)
                run.sweep.brake()
                await asyncio.sleep(self._brake_s)
                self._motion.stop(neutral=False)
                await self._look(yaw, pitch)
                await asyncio.sleep(self._settle_s)
                still = await asyncio.to_thread(self._grab)
                if still is not None:
                    found = await asyncio.to_thread(detect_tiled, detector, still, label)
                    hit = await self._accept(still, found, target)
                    if hit is not None:
                        where = _aim(still, hit, dict(self._motion.pose))
                        await self._look(*where)  # centre it
                        looks = await describe_box(self._ask, still, hit) if self._ask else None
                        done = True
                        run.found_yaw, run.at, self._last = where[0], self._clock(), run
                        log.info("found %s in %.1f s", target, self._clock() - t0)
                        result: dict[str, Any] = {
                            "status": "found",
                            "object": label,
                            "where": describe(where[0]),
                            "height": height(where[1]),
                            "seconds": round(self._clock() - t0),
                        }
                        if looks:
                            result["description"] = looks
                        return result
                log.info("candidate at %.0f° rejected, sweeping on", yaw)
                run.skipped.append(yaw)
                self._submit(run.sweep.move())
            log.info("%s not found after %d sweeps", target, run.passes)
            return {
                "status": "not_found",
                "sweeps": run.passes,
                "note": "I looked all around, except straight behind me",
                "seconds": round(self._clock() - t0),
            }
        finally:
            if not done:  # not found, failed or cancelled by a stop: face the front again
                self._motion.stop()
                with contextlib.suppress(Exception):
                    self._motion.submit(goto("search done", dict(NEUTRAL), 1.0))

    def _submit(self, move: Move) -> None:
        if error := self._motion.submit(move):
            raise RuntimeError(error)

    async def _look(self, yaw: float, pitch: float) -> None:
        """Turn to (world yaw, pitch) with a short move and wait until there."""
        now = self._motion.planned()
        reach = LIMITS["body_yaw"] + LIMITS["yaw"]
        plan = plan_turn(max(-reach, min(reach, wrap(yaw))), now["body_yaw"])
        assert plan is not None  # the clip above keeps it in reach
        body, head = plan
        pitch = max(-LIMITS["pitch"], min(LIMITS["pitch"], pitch))
        target: Pose = {**now, "yaw": head, "body_yaw": body, "pitch": pitch}
        swing = max(
            abs(body - now["body_yaw"]), abs(wrap(head - now["yaw"])), abs(pitch - now["pitch"])
        )
        self._submit(goto("look at it", target, max(MIN_MOVE_S, swing / SPEED_DPS)))
        await self._motion.settled()

    async def _accept(self, frame: Frame, found: list[Detection], target: str) -> Detection | None:
        found = [d for d in found if d.score >= MIN_SCORE]
        if not found:
            return None
        best = max(found, key=lambda d: d.score)
        if self._ask is None:
            return best
        if len(found) == 1 and best.score >= SURE and not _personal(target):
            return best
        return await choose(self._ask, frame, found, target)


def _aim(frame: Frame, d: Detection, seen_from: Pose) -> tuple[float, float]:
    """World (yaw, pitch) of a detection's centre, seen from the head pose `seen_from`."""
    h, w = frame.shape[:2]
    fx = w / 2 / TAN_HALF_H
    x, y, bw, bh = d.box
    yaw = seen_from["yaw"] - math.degrees(math.atan((x + bw / 2 - w / 2) / fx))
    pitch = seen_from["pitch"] + math.degrees(math.atan((y + bh / 2 - h / 2) / fx))
    return wrap(yaw), pitch


def _personal(target: str) -> bool:
    """'my phone' must be the user's: let the vision model check even a lone hit."""
    words = target.lower().split()
    return any(w in ("my", "mine", "của", "tôi", "mình") for w in words)
