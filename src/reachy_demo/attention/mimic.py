"""Mimic: Reachy copies the user's head moves like a mirror until told to stop (F6).

Each camera frame gives the user's head pose (perception/head_pose.py). Reachy faces the user
and adds the user's head turn on top, mirrored: when you turn toward your left, Reachy turns
toward that same side of the room (its right), as your reflection would. Nods (pitch), head
tilts (roll) and leaning toward or away from the robot (head forward/back, x) are copied too.

Smoothness, in two stages:
1. A one-euro filter on each measured value: steady when you hold still, quick when you move.
2. A critically damped spring per axis in the motion loop (50 Hz): position and speed stay
   continuous, so the head never jumps, and speed is capped at SPEED_DPS.

Where the user is: camera frames arrive 0.2-0.5 s late, so the image position of the face lags
the head. Re-aiming at it every frame made a fast feedback loop that swung head and body
around a still user (±55° at 0.4 s late). So that position only drifts the "facing the user"
direction slowly (FACE_TAU_S). The copied head angles need no such care: they are measured
relative to the user's line of sight, which the robot's own turn doesn't change.

Limits: robot/motion.py clamps every pose to the safety box (pitch ±30°, roll ±25°, head yaw
±45° from the body, body ±120°, x ±20 mm; every combination checked with the SDK's IK on
2026-11-17). On top of that, the camera sits in the head: turn it too far from the user and they
leave the picture. So the copied turn is capped at MAX_YAW / MAX_PITCH from facing the user.

Angles are degrees; robot yaw + = left, pitch + = down, roll + = toward its left shoulder.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import statistics
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt

from reachy_demo.audio.sound_events import plan_turn, wrap
from reachy_demo.perception.head_pose import HeadPose
from reachy_demo.robot.motion import AXES, BUSY_LONG, LIMITS, NEUTRAL, Motion, Move, Pose

log = logging.getLogger(__name__)

Frame = npt.NDArray[np.uint8]
GrabFrame = Callable[[], Frame | None]  # current BGR camera frame, None if unavailable


class Estimator(Protocol):
    def estimate(self, bgr: Frame, at_s: float) -> HeadPose | None: ...

    def close(self) -> None: ...


# The camera sees about ±44° wide and ±28° high (wireless camera at 1280x720). Keep the user's
# face this far inside, so turning to copy them never loses them.
MAX_YAW = 35.0
MAX_PITCH = 22.0
MAX_ROLL = LIMITS["roll"]
X_MM_PER_CM = 2.0  # lean 10 cm toward the robot -> its head moves 20 mm forward (the limit)
CALIBRATE_N = 5  # readings averaged for the user's resting distance
LOST_S = 1.0  # no face this long: look back at where the user was
OMEGA = 12.0  # spring stiffness (rad/s): ~0.3 s to settle, inside the 300 ms lag budget (F6)
SPEED_DPS = 150.0  # also mm/s for x
MAX_DT = 0.05  # a late tick integrates at most this much
POLL_S = 0.03  # the frame grab and the model set the real rate (~15-25 Hz)
FACE_TAU_S = 3.0  # facing the user follows their position this slowly (frames lag the head)
MAX_MIMIC_S = 300.0  # shortcut: mimic ends after 5 min like a dance does; raise if needed
STEERED = ("yaw", "pitch", "roll", "x", "body_yaw")


class OneEuro:
    """One-euro filter (Casiez et al. 2012): less jitter when still, less lag when moving."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self._min_cutoff, self._beta, self._d_cutoff = min_cutoff, beta, d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or t <= self._t:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self._d_cutoff, dt) * (dx - self._dx)
        cutoff = self._min_cutoff + self._beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        return self._x


def _clip(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


class Mimic:
    def __init__(
        self,
        motion: Motion,
        grab_frame: GrabFrame,
        make_estimator: Callable[[], Estimator],
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = POLL_S,
    ) -> None:
        self._motion = motion
        self._grab = grab_frame
        self._make_estimator = make_estimator
        self._clock = clock
        self._poll_s = poll_s
        self._task: asyncio.Task[None] | None = None
        self._reset()

    def _reset(self) -> None:
        self._target: Pose | None = None  # set from the start pose on the move's first tick
        self._pose: Pose = dict(NEUTRAL)
        self._speed: dict[str, float] = dict.fromkeys(STEERED, 0.0)
        self._t = 0.0
        self._filters = {k: OneEuro() for k in ("yaw", "pitch", "roll")}
        self._distances: list[float] = []
        self._rest_cm: float | None = None
        self._last_seen = -math.inf
        self._user: tuple[float, float] | None = None  # (yaw, pitch) of the user's face
        self._user_t = 0.0  # when `_user` was last updated

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> str | None:
        """Start mimicking; returns an error string instead when it can't."""
        if self.active:
            return None
        if self._motion.dancing:  # check before loading the model (about 1 s)
            return BUSY_LONG
        try:
            estimator = await asyncio.to_thread(self._make_estimator)
        except (OSError, ImportError, RuntimeError, ValueError) as e:
            log.warning("mimic unavailable: %s", e)
            return f"mimic_unavailable: {e}"
        self._reset()
        move = Move("mimic user", True, MAX_MIMIC_S, self._pose_at, dict(NEUTRAL))
        if error := self._motion.submit(move):
            estimator.close()
            return error
        self._task = asyncio.create_task(self._run(estimator))
        log.info("mimicking the user")
        return None

    def stop(self) -> None:
        """End mimic: the head eases back to neutral, then the model is freed."""
        if self.active:
            self._motion.stop()

    # --- the long move ------------------------------------------------------------------

    def _pose_at(self, t: float, start: Pose) -> Pose:
        if self._target is None:  # first tick: hold where the head is
            self._target, self._pose, self._t = dict(start), dict(start), t
        dt = min(max(t - self._t, 0.0), MAX_DT)
        self._t = t
        pose = dict(self._pose)
        for k in STEERED:  # critically damped spring, speed capped
            v = self._speed[k]
            a = OMEGA * OMEGA * (self._target[k] - pose[k]) - 2 * OMEGA * v
            v = _clip(v + a * dt, SPEED_DPS)
            self._speed[k] = v
            pose[k] += v * dt
        self._pose = pose
        return {k: pose[k] for k in AXES}

    # --- the loop -----------------------------------------------------------------------

    async def _run(self, estimator: Estimator) -> None:
        warned = False
        try:
            while self._motion.dancing:  # stop/close/an error end the move, and so this loop
                seen_from = dict(self._motion.pose)  # the head pose when the frame is taken
                try:
                    frame = await asyncio.to_thread(self._grab)
                    now = self._clock()
                    head = (
                        None
                        if frame is None
                        else await asyncio.to_thread(estimator.estimate, frame, now)
                    )
                except Exception as e:  # noqa: BLE001 - fail small: a bad frame skips a reading
                    if not warned:
                        log.warning("mimic frame failed: %s", e)
                        warned = True
                    head, now = None, self._clock()
                self.step(head, seen_from, now)
                await asyncio.sleep(self._poll_s)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(estimator.close)
            log.info("stopped mimicking")

    def step(self, head: HeadPose | None, seen_from: Pose, now: float) -> None:
        """Update the target from one head-pose reading taken while the head was at `seen_from`."""
        if self._target is None:
            return  # the move has not started yet
        if head is None:
            if now - self._last_seen > LOST_S and self._user is not None:
                self._aim(*self._user, roll=0.0, x=0.0)  # look back where the user was
            return
        self._last_seen = now
        f = self._filters
        # Where the user is, in robot coordinates: the camera turns with the head. Slowly, since
        # the frame is older than `seen_from` (see the module docstring).
        seen = (wrap(seen_from["yaw"] + head.bearing_yaw), seen_from["pitch"] + head.bearing_pitch)
        if self._user is None:
            user = seen
        else:
            k = min(max(now - self._user_t, 0.0) / FACE_TAU_S, 1.0)
            user = (
                wrap(self._user[0] + k * wrap(seen[0] - self._user[0])),
                self._user[1] + k * (seen[1] - self._user[1]),
            )
        self._user, self._user_t = user, now
        yaw, pitch = f["yaw"](head.yaw, now), f["pitch"](head.pitch, now)
        # The image roll turns with the camera: add the robot's roll back to get the user's.
        roll = f["roll"](head.roll + seen_from["roll"], now)
        x = 0.0
        if self._rest_cm is None:
            self._distances.append(head.distance_cm)
            if len(self._distances) >= CALIBRATE_N:
                self._rest_cm = statistics.median(self._distances)
        else:
            x = (self._rest_cm - head.distance_cm) * X_MM_PER_CM
        # Mirror: Reachy moves toward the same side of the room as the user. The user's turn to
        # their left (yaw +) is the robot's right (yaw -). A tilt to their left shoulder is
        # clockwise in the image (roll -) and the robot's right shoulder (roll -). Nods copy.
        self._aim(
            user[0] - _clip(yaw, MAX_YAW),
            user[1] + _clip(pitch, MAX_PITCH),
            roll=_clip(roll, MAX_ROLL),
            x=_clip(x, LIMITS["x"]),
        )

    def _aim(self, yaw: float, pitch: float, roll: float, x: float) -> None:
        assert self._target is not None
        reach = LIMITS["body_yaw"] + LIMITS["yaw"]
        plan = plan_turn(max(-reach, min(reach, wrap(yaw))), self._target["body_yaw"])
        assert plan is not None  # the clip above keeps it in reach
        body, head = plan
        self._target.update(
            yaw=head,
            body_yaw=body,
            pitch=_clip(pitch, LIMITS["pitch"]),
            roll=roll,
            x=x,
        )
