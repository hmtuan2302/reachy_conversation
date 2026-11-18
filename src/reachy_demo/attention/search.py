"""Find a named object in the room: stop and look around, then look at it (F5, on request).

The head and body turn to STOPS one by one (every STEP_DEG, front and left first, then the
right side, out to SEARCH_YAW), and at each stop wait for a still frame before YOLOX looks
(whole frame + 3x3 tiles). Frames taken while turning were blurred and, over Wi-Fi, broken up
by the video codec, so a phone in plain view was missed; a still frame has neither problem.
The camera sees about ±44° across, so neighbouring stops overlap. Pitch stays at SEARCH_PITCH.

A hit is checked on that same still frame: a lone confident hit is accepted, otherwise the
vision model checks it (several candidates, "my phone", a weak score). Accepted: centre it,
describe it (vision model) and report, and the agent asks the user "is that it?". Rejected,
or the user says no (`resume`): that direction is skipped and the search goes on from that
stop. At most MAX_PASSES passes over all stops. Not found or stopped: the head goes back to
where it was before the search. Found: it stays on the object until the user says yes
(`confirm`), or RESUME_S passes, and then goes back.

A spoken stop cancels the tool call (barge-in), which ends the search and frees the detector.
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
    CAMERA_LAG_S,
    COCO,
    MARK_STALE_S,
    MIN_SCORE,
    Ask,
    Detection,
    Frame,
    GrabFrame,
    Mark,
    ObjectDetector,
    choose,
    describe_box,
    detect_tiled,
)
from reachy_demo.robot.motion import (
    BUSY_LONG,
    LIMITS,
    NEUTRAL,
    Motion,
    Move,
    Pose,
    goto,
)

log = logging.getLogger(__name__)

SEARCH_YAW = 160.0  # body 120° + head 40°: only ~30° straight behind is never seen
STEP_DEG = 40.0  # between stops; the camera sees ~88° across, so stops overlap
STOPS = (
    *(STEP_DEG * i for i in range(int(SEARCH_YAW / STEP_DEG) + 1)),  # front, then left
    *(-STEP_DEG * i for i in range(1, int(SEARCH_YAW / STEP_DEG) + 1)),  # then right
)
SEARCH_PITCH = 0.0  # level (asked on the robot): the camera sees ~-28°..+28°
SETTLE_S = 0.3  # after a turn, before the frame counts as still (plus the camera lag)
SKIP_DEG = 12.0  # a rejected object's direction is ignored for the rest of the search
SPEED_DPS = 60.0  # turning speed; smoothstep peaks at 1.5x
MIN_MOVE_S = 0.4
MAX_PASSES = 2  # passes over all stops per search, resumes included
RESUME_S = 60.0  # "no, not that one" within this long resumes the search
FOUND_MARK_S = RESUME_S  # the found box stays until the head turns back
SURE = 0.55  # one detection at least this confident needs no vision check


@dataclass
class _Run:
    """One search: kept after a find, so "no, not that one" can resume it."""

    label: str
    skipped: list[float] = field(default_factory=list)  # world yaws of rejected objects
    stop: int = 0  # index in STOPS of the stop to look from next
    passes: int = 0  # passes over all stops finished
    found_yaw: float | None = None
    at: float = 0.0  # when it was found
    home: Pose = field(default_factory=lambda: dict(NEUTRAL))  # the head before the search


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
        lag_s: float = CAMERA_LAG_S,
    ) -> None:
        """`lag_s`: how old a grabbed frame is, from head command to image (CAMERA_LAG_S)."""
        self._lag_s = lag_s
        self._mark: tuple[Mark, float] | None = None  # (box for the debug view, shown until)
        self._motion = motion
        self._grab = grab
        self._make_detector = make_detector
        self._ask = ask
        self._clock = clock
        self._settle_s = settle_s
        self._last: _Run | None = None  # the last found search, for a resume
        self._home_later: asyncio.Task[None] | None = None  # back home if nobody answers

    async def find(self, target: str, label: str, resume: bool = False) -> dict[str, Any]:
        """Look around for `target` (COCO class `label`); returns the tool result.

        `resume`: the user said the last find was the wrong object. Within RESUME_S, the same
        search goes on from the stop where it found it and skips that object; otherwise it
        starts over. At most MAX_PASSES passes per search, resumes included.
        """
        if label not in COCO:
            return {"error": f"unsupported_object: {target} is not one I can recognise"}
        if self._motion.dancing:
            return {"error": BUSY_LONG}
        last, self._last = self._last, None
        self._cancel_home_later()
        now = self._clock()
        resumed = resume and last is not None and last.label == label and now - last.at <= RESUME_S
        if resumed:
            assert last is not None
            run = last
            if run.found_yaw is not None:
                run.skipped.append(run.found_yaw)  # the user said it isn't this one
        else:
            # after an unanswered find the head is still on that object: keep the first home
            run = _Run(label, home=last.home if last else self._motion.planned())
        try:
            detector = await asyncio.to_thread(self._make_detector)
        except (OSError, ImportError, RuntimeError) as e:
            log.warning("object detector unavailable: %s", e)
            return {"error": f"find_unavailable: {e}"}
        t0 = self._clock()
        done = False
        try:
            while run.passes < MAX_PASSES:
                while run.stop < len(STOPS):
                    await self._look(STOPS[run.stop], SEARCH_PITCH, "search stop")
                    await asyncio.sleep(self._settle_s + self._lag_s)  # a frame from this stop
                    seen_from = dict(self._motion.pose)
                    frame = await asyncio.to_thread(self._grab)
                    if frame is None:
                        return {"error": "no camera frame (camera off or not connected)"}
                    found = await asyncio.to_thread(detect_tiled, detector, frame, label)
                    hits = [
                        d
                        for d in found
                        if d.score >= MIN_SCORE
                        and all(
                            abs(wrap(_aim(frame, d, seen_from)[0] - s)) > SKIP_DEG
                            for s in run.skipped
                        )
                    ]
                    if not hits:
                        run.stop += 1
                        continue
                    self._show(frame, max(hits, key=lambda d: d.score), "yolox")
                    hit = await self._accept(frame, hits, target)
                    if hit is None:  # the vision model said none of them is it
                        yaws = [_aim(frame, d, seen_from)[0] for d in hits]
                        log.info("vision said no to %d boxes at %s°, looking on", len(hits), yaws)
                        run.skipped += yaws
                        run.stop += 1
                        continue
                    where = _aim(frame, hit, seen_from)
                    await self._look(*where)  # centre it
                    await self._show_found(detector, label)  # boxed while it describes
                    looks = await describe_box(self._ask, frame, hit) if self._ask else None
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
                run.passes += 1
                run.stop = 0
                log.info("search pass %d done", run.passes)
            log.info("%s not found after %d passes", target, run.passes)
            return {
                "status": "not_found",
                "sweeps": run.passes,
                "note": "I looked all around, except straight behind me",
                "seconds": round(self._clock() - t0),
            }
        finally:
            if done:  # stay on it until the user says yes, or nobody answers
                self._home_later = asyncio.create_task(self._back_later(run))
            else:  # not found, failed or stopped: back to where the head was
                self._motion.stop(neutral=False)
                self._go_home(run)

    async def confirm(self) -> dict[str, Any]:
        """The user said the found object is the right one: turn back to where the head was."""
        last, self._last = self._last, None
        self._cancel_home_later()
        if last is None:
            return {"status": "nothing_found_to_confirm"}
        self._go_home(last)
        return {"status": "confirmed"}

    def _go_home(self, run: _Run) -> None:
        self._mark = None  # the found box is off once it turns
        with contextlib.suppress(Exception):
            self._motion.submit(self._turn("search done", run.home))

    async def _back_later(self, run: _Run) -> None:
        on_it = self._motion.planned()
        await asyncio.sleep(RESUME_S)
        if self._last is not run:
            return
        self._last = None
        moved = any(
            abs(wrap(self._motion.planned()[k] - on_it[k])) > 1 for k in ("yaw", "body_yaw")
        )
        if moved or self._motion.dancing:  # something else has the head now: leave it
            self._mark = None
            return
        log.info("no answer about the found object, turning back")
        self._go_home(run)

    def _cancel_home_later(self) -> None:
        if self._home_later is not None:
            self._home_later.cancel()
            self._home_later = None

    def mark(self) -> Mark | None:
        """The latest box for the debug view: a hit at a stop, or the found object; else None."""
        if self._mark is None or self._clock() > self._mark[1]:
            return None
        return self._mark[0]

    def _show(self, frame: Frame, d: Detection, source: str, hold_s: float = MARK_STALE_S) -> None:
        h, w = frame.shape[:2]
        x, y, bw, bh = d.box
        aim = ((x + bw / 2) / w * 2 - 1, (y + bh / 2) / h * 2 - 1)
        now = self._clock()
        self._mark = (Mark(d.box, d.score, d.label, source, aim, now), now + hold_s)

    async def _show_found(self, detector: ObjectDetector, label: str) -> None:
        """Box the centred object on a fresh frame for the debug view (still's box is off now)."""
        await asyncio.sleep(self._settle_s + self._lag_s)  # a frame from after the turn
        frame = await asyncio.to_thread(self._grab)
        if frame is None:
            return
        found = await asyncio.to_thread(detect_tiled, detector, frame, label)
        h, w = frame.shape[:2]
        near = [d for d in found if d.score >= MIN_SCORE]
        if near:  # the one nearest the centre, where the head now points
            d = min(
                near,
                key=lambda d: (
                    abs(d.box[0] + d.box[2] / 2 - w / 2) + abs(d.box[1] + d.box[3] / 2 - h / 2)
                ),
            )
            self._show(frame, d, "found", FOUND_MARK_S)

    def _turn(self, name: str, target: Pose) -> Move:
        """A short move to `target` at the search's turning speed (at least 1 s)."""
        now = self._motion.planned()
        swing = max(abs(wrap(target[k] - now[k])) for k in ("yaw", "body_yaw", "pitch"))
        return goto(name, dict(target), max(1.0, swing / SPEED_DPS))

    def _submit(self, move: Move) -> None:
        if error := self._motion.submit(move):
            raise RuntimeError(error)

    async def _look(self, yaw: float, pitch: float, name: str = "look at it") -> None:
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
        self._submit(goto(name, target, max(MIN_MOVE_S, swing / SPEED_DPS)))
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
