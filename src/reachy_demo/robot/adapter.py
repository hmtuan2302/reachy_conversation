"""The only module that imports reachy_mini (docs/architecture/01-overview.md §4).

robot/dances.py imports the dances library, which imports the SDK in turn; it reads the move
catalog only and never talks to the robot.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

from reachy_demo.robot.motion import NEUTRAL, Pose, SetPose

WAKE_S = 2.0  # rest pose -> neutral at startup
REST_S = 2.0  # neutral -> rest pose at exit


@contextmanager
def connect_head(host: str) -> Iterator[SetPose]:
    """Connect over Wi-Fi by IP and yield set_pose(pose): head position (mm), orientation and
    antennas (degrees). It sends one target and returns at once; only robot/motion.py calls it.

    On entry the motors are enabled and the robot eases to neutral, so the motion loop starts
    from a known pose. On exit the head is lowered into the SDK's rest (sleep) pose, and only
    then are the motors disabled: the head already rests on its stops, so it doesn't drop.
    """
    from reachy_mini import ReachyMini  # lazy: the SDK is the optional `robot` extra
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

    with ReachyMini(host=host, connection_mode="network", media_backend="no_media") as mini:

        def set_pose(p: Pose) -> None:
            mini.set_target(head=head(p), antennas=antennas(p))

        try:
            # Torque-off from a previous exit stays on the robot; targets are ignored until
            # the motors are enabled. Enable first: enable_motors() pins targets to the
            # present pose (SDK set_target note), so moves must come after it.
            mini.enable_motors()
            mini.goto_target(head=head(NEUTRAL), antennas=antennas(NEUTRAL), duration=WAKE_S)
            yield set_pose
        finally:
            try:
                # Like ReachyMini.goto_sleep(), without its sound (media is off here).
                mini.goto_target(
                    head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=REST_S
                )
            finally:
                mini.disable_motors()
