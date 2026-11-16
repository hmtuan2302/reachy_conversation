"""The only module that imports reachy_mini (docs/architecture/01-overview.md §4).

robot/dances.py imports the dances library, which imports the SDK in turn; it reads the move
catalog only and never talks to the robot.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from reachy_demo.robot.motion import NEUTRAL, Pose, SetPose

log = logging.getLogger(__name__)

WAKE_S = 2.0  # rest pose -> neutral at startup
REST_S = 2.0  # neutral -> rest pose at exit


@dataclass(frozen=True)
class Robot:
    set_pose: SetPose  # send one target, return at once; only robot/motion.py calls it
    grab_jpeg: Callable[[], bytes | None]  # current camera frame; None without a camera
    has_camera: bool


def _open(host: str, camera: bool) -> Any:
    from reachy_mini import ReachyMini  # lazy: the SDK is the optional `robot` extra

    if camera:
        # Camera streams over WebRTC (spike 2026-11-16: works from WSL2, first frame ~1.5 s).
        # It also opens an audio channel we don't use; the "No Reachy Mini Audio USB device"
        # error it logs on a laptop is harmless.
        try:
            return ReachyMini(host=host, connection_mode="network", media_backend="webrtc")
        except Exception as e:  # fail small: no camera, but the head still works
            log.warning("camera unavailable (%s); connecting without it", e)
    return ReachyMini(host=host, connection_mode="network", media_backend="no_media")


@contextmanager
def connect_robot(host: str, camera: bool = True) -> Iterator[Robot]:
    """Connect over Wi-Fi by IP; yields the pose sender and the camera.

    Poses: head position (mm), orientation and antennas (degrees). On entry the motors are
    enabled and the robot eases to neutral, so the motion loop starts from a known pose. On exit
    the head is lowered into the SDK's rest (sleep) pose, and only then are the motors disabled:
    the head already rests on its stops, so it doesn't drop.
    """
    from reachy_mini.reachy_mini import SLEEP_ANTENNAS_JOINT_POSITIONS, SLEEP_HEAD_POSE
    from reachy_mini.utils import create_head_pose

    def head(p: Pose) -> object:
        return create_head_pose(
            x=p["x"],
            y=p["y"],
            z=p["z"],
            roll=p["roll"],
            pitch=p["pitch"],
            yaw=p["yaw"],
            mm=True,
            degrees=True,
        )

    def antennas(p: Pose) -> list[float]:
        return [math.radians(p["antenna_0"]), math.radians(p["antenna_1"])]

    with contextlib.ExitStack() as stack:
        mini = stack.enter_context(_open(host, camera))
        has_camera = mini.media.camera is not None

        def set_pose(p: Pose) -> None:
            mini.set_target(head=head(p), antennas=antennas(p))

        def grab_jpeg() -> bytes | None:
            return mini.media.get_frame_jpeg() if has_camera else None

        try:
            # Torque-off from a previous exit stays on the robot; targets are ignored until
            # the motors are enabled. Enable first: enable_motors() pins targets to the
            # present pose (SDK set_target note), so moves must come after it.
            mini.enable_motors()
            mini.goto_target(head=head(NEUTRAL), antennas=antennas(NEUTRAL), duration=WAKE_S)
            yield Robot(set_pose, grab_jpeg, has_camera)
        finally:
            try:
                # Like ReachyMini.goto_sleep(), without its sound (media sound isn't used).
                mini.goto_target(
                    head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=REST_S
                )
            finally:
                mini.disable_motors()
