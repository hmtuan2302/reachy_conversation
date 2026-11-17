"""The motion loop: the only code that sends head poses to the robot (01-overview rule 1).

Moves (short head moves, long dances) wait in one queue; one loop plays them by sending a full
pose (head and antennas) about 50 times a second, so two moves can never fight over the robot.
No SDK import here: the robot is reached through an injected `set_pose` (robot/adapter.py),
which lets tests use fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

# A full pose: head position (mm), head orientation (degrees, yaw in the world frame), antenna
# joints (degrees, in set_target's order) and body yaw (degrees). Head tools change roll/pitch/
# yaw; dances use every head axis; only sound turns move the body.
Pose = dict[str, float]
SetPose = Callable[[Pose], None]  # set_target(pose); returns at once
AXES = ("x", "y", "z", "roll", "pitch", "yaw", "antenna_0", "antenna_1", "body_yaw")
# The SDK's neutral: head centered, antennas ~10° apart (INIT_ANTENNAS_JOINT_POSITIONS,
# "to reduce shaking at vertical").
NEUTRAL: Pose = {**dict.fromkeys(AXES, 0.0), "antenna_0": -10.0, "antenna_1": 10.0}
# Safety net: no pose outside these is ever sent, whatever a move computes. Set to the widest
# range the dances library uses (sampled 2026-11-16: yaw 45°, pitch 30°, roll 25°, y/z 40 mm,
# x 20 mm, antennas 60° + the 10° neutral), so its moves play unclipped. `yaw` is the head's
# yaw relative to the body. Body yaw stays well inside the SDK's hard limit (160°; head-to-body
# 65°).
LIMITS: Pose = {
    "x": 20.0,
    "y": 40.0,
    "z": 40.0,
    "roll": 25.0,
    "pitch": 30.0,
    "yaw": 45.0,
    "antenna_0": 70.0,
    "antenna_1": 70.0,
    "body_yaw": 120.0,
}
EASE_S = 0.5  # easing into and out of a dance, and back to neutral after a stop
MAX_LONG_S = 15.0  # longest dance, repeats included
BUSY_LONG = "busy: dancing or following the user, say stop first"
BUSY_FULL = "busy: too many moves queued"
Offset = Callable[[float], Pose]  # seconds since the dance started -> offset from neutral


@dataclass(frozen=True)
class Move:
    name: str
    # long moves (dances, face tracking) block every other move until they end or are stopped
    long: bool
    duration_s: float
    pose_at: Callable[[float, Pose], Pose]  # (seconds since start, start pose) -> pose
    end: Pose  # where the move leaves the head


def _smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


def _lerp(a: Pose, b: Pose, f: float) -> Pose:
    return {k: a[k] + (b[k] - a[k]) * f for k in AXES}


def _clip(v: float, limit: float, center: float = 0.0) -> float:
    return min(max(v, center - limit), center + limit)


def clamp(pose: Pose) -> Pose:
    out = {k: _clip(pose.get(k, 0.0), LIMITS[k]) for k in AXES}
    out["yaw"] = _clip(pose.get("yaw", 0.0), LIMITS["yaw"], out["body_yaw"])  # head vs body
    return out


def goto(name: str, target: Pose, duration_s: float = 1.0) -> Move:
    """A short move: ease from wherever the head is to `target`."""
    return Move(
        name,
        False,
        duration_s,
        lambda t, start: _lerp(start, target, _smooth(t / duration_s)),
        target,
    )


def dance(name: str, pass_s: float, offset: Offset, repeat: int = 1) -> Move:
    """A long move: ease in from the start pose, dance `repeat` passes, end at neutral.

    `repeat` is clamped so the dance lasts at most MAX_LONG_S.
    """
    repeat = max(1, min(repeat, int(MAX_LONG_S // pass_s)))
    duration = pass_s * repeat

    def pose_at(t: float, start: Pose) -> Pose:
        ease_in = _smooth(t / EASE_S)
        amount = min(ease_in, _smooth((duration - t) / EASE_S))
        base = _lerp(start, NEUTRAL, ease_in)
        d = offset(t)
        return {k: base[k] + d.get(k, 0.0) * amount for k in AXES}

    return Move(f"dance {name}", True, duration, pose_at, dict(NEUTRAL))


class Motion:
    """One queue, one loop. Rules: a dance blocks other moves; stop is always accepted."""

    def __init__(
        self,
        set_pose: SetPose,
        hz: float = 50.0,
        max_pending: int = 4,
        on_error: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._set_pose = set_pose
        self._tick_s = 1.0 / hz
        self._max_pending = max_pending
        self._on_error = on_error
        self._clock = clock
        self._queue: deque[Move] = deque()
        self._current: Move | None = None
        self._abort = False  # end the current move on the next tick
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task[None] | None = None
        self.pose: Pose = dict(NEUTRAL)  # last pose sent: this loop is the only sender

    @property
    def busy(self) -> bool:
        return self._current is not None or bool(self._queue)

    @property
    def dancing(self) -> bool:
        """A long move is running or queued."""
        return any(m.long for m in [*self._queue, *([self._current] if self._current else [])])

    def planned(self) -> Pose:
        """Where the head will be once every queued move has run."""
        if self._queue:
            return dict(self._queue[-1].end)
        return dict(self._current.end if self._current else self.pose)

    def submit(self, move: Move) -> str | None:
        """Queue `move`; returns an error string instead when the rules refuse it."""
        if self.dancing:
            return BUSY_LONG
        if len(self._queue) >= self._max_pending:
            return BUSY_FULL
        self._queue.append(move)
        self._idle.clear()
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        return None

    def stop(self) -> None:
        """Drop queued moves. A dance ends now and the head eases to neutral; a short move
        in progress finishes (at most 1 s), so short moves always end on their fixed angle."""
        self._queue.clear()
        if self._current and self._current.long:
            self._abort = True
            self._current = None  # free at once: a new move in the same reply is accepted
            self._queue.append(goto("neutral", NEUTRAL, EASE_S))

    async def close(self, timeout_s: float = 3.0) -> None:
        """Exit: end whatever runs now, ease to neutral, stop the loop."""
        if self._task is None or self._task.done():
            return
        self._queue.clear()
        self._abort = self._current is not None
        self._queue.append(goto("neutral", NEUTRAL, EASE_S))
        self._idle.clear()
        self._wake.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._idle.wait(), timeout_s)
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        try:
            while True:
                if not self._queue:
                    self._idle.set()
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                move = self._queue.popleft()
                self._current = move
                await self._play(move)
                self._current = None
        except asyncio.CancelledError:
            raise
        except Exception as e:  # fail small: report it, forget the queue, never stay busy
            log.warning("motion failed: %s", e)
            self._queue.clear()
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._current = None
            self._abort = False
            self._idle.set()

    async def _play(self, move: Move) -> None:
        start, t0 = dict(self.pose), self._clock()
        log.info("motion %s for %.1f s", move.name, move.duration_s)
        while True:
            if self._abort:
                self._abort = False
                return
            t = min(self._clock() - t0, move.duration_s)
            pose = clamp(move.pose_at(t, start))
            # shortcut: one thread hop per tick keeps the event loop free; fine at 50 Hz
            await asyncio.to_thread(self._set_pose, pose)
            self.pose = pose
            if t >= move.duration_s:
                return
            await asyncio.sleep(self._tick_s)
