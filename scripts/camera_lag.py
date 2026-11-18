"""Measure CAMERA_LAG_S: time from a head command to the first camera frame that shows the move.

Run from the repo root, with the agent app stopped (one owner per head):
    uv run python scripts/camera_lag.py
Point the robot at a scene with texture (not a blank wall) and keep still. The head steps
left/right by 8 degrees 6 times. Put the printed median in .env as CAMERA_LAG_S.

It includes the motor response, which the object-tracking loop also has to wait for.
"""

import os
import statistics
import time

import numpy as np
from dotenv import find_dotenv, load_dotenv

from reachy_demo.robot.adapter import connect_robot
from reachy_demo.robot.motion import NEUTRAL

STEP_DEG = 8.0
TRIES = 6
TIMEOUT_S = 2.0


def small(frame: np.ndarray) -> np.ndarray:
    return frame[::8, ::8].mean(axis=2)


def moved_after(grab, ref: np.ndarray, noise: float, t0: float) -> float | None:
    """Seconds from t0 to the first frame that differs from `ref` well above the noise."""
    while (now := time.monotonic()) - t0 < TIMEOUT_S:
        frame = grab()
        if frame is not None and np.abs(small(frame) - ref).mean() > max(4 * noise, 3.0):
            return now - t0
    return None


def main() -> int:
    load_dotenv(find_dotenv(usecwd=True))
    with connect_robot(os.environ["ROBOT_HOST"]) as robot:
        if not robot.has_camera:
            print("no camera")
            return 1
        lags: list[float] = []
        for i in range(TRIES):
            yaw = STEP_DEG if i % 2 == 0 else 0.0
            time.sleep(1.5)  # settle
            frames = [small(f) for f in (robot.grab_frame() for _ in range(10)) if f is not None]
            ref = frames[-1]
            noise = float(np.mean([np.abs(f - ref).mean() for f in frames[:-1]]))
            t0 = time.monotonic()
            robot.set_pose({**NEUTRAL, "yaw": yaw})
            lag = moved_after(robot.grab_frame, ref, noise, t0)
            print(f"try {i + 1}: " + ("no change seen" if lag is None else f"{lag:.3f} s"))
            if lag is not None:
                lags.append(lag)
        robot.set_pose(dict(NEUTRAL))
        if not lags:
            print("no move seen: point the camera at something with texture")
            return 1
        print(f"CAMERA_LAG_S={statistics.median(lags):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
