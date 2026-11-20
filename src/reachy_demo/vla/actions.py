"""Actions from face boxes (plan §7.2): how far to turn the gaze, per frame, with hindsight.

Each face box becomes a direction in the room: the camera's gaze at the frame's capture time
(measured head pose, poses.jsonl) plus the box centre's angle in the image. Then per frame:

- **visible:** turn toward where the face will be LOOKAHEAD_S later (the robot acts that late),
  from a short centred average of directions (removes box jitter).
- **not visible, but seen again within HINDSIGHT_S:** turn toward where it is next seen. This
  teaches "body in view, head above: tilt up" and "left fast to the right: turn right", from the
  recording itself (the chase and body rules made the camera find the face again).
- **otherwise:** hold (0, 0).

Each row also carries where the face was last seen (relative to the current gaze) and how long
ago: the policy's memory input, kept at runtime from its own past outputs.

Angles in degrees: yaw + = left, pitch + = down (robot/motion.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from reachy_demo.perception.head_pose import TAN_HALF_H, TAN_HALF_V

LOOKAHEAD_S = 0.2  # plan §7.2: start with 200 ms, set from the measured latency (§12.1)
SMOOTH = 5  # frames in the centred average
HINDSIGHT_S = 3.0  # a face seen again later than this: the frame says "hold"
MAX_D = 20.0  # plan §3: actions clamped to ±20° (the motion loop repeats them)
MAX_LAST_D = 90.0
NEVER = -1.0  # since_seen_s when the face hasn't been seen yet in the episode

Floats = npt.NDArray[np.float64]


def image_angles(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    """(yaw, pitch) of pixel (u, v) from the image centre.

    shortcut: pinhole with the nominal field of view and no lens distortion; use the episode's
    calibration (calib_id, plan §6.7) once track_calibrate.py exists.
    """
    x = u / max(width - 1, 1) * 2 - 1
    y = v / max(height - 1, 1) * 2 - 1
    return -math.degrees(math.atan(x * TAN_HALF_H)), math.degrees(math.atan(y * TAN_HALF_V))


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


def actions(
    boxes: list[dict[str, Any]],
    t_arrival: list[float],
    poses: list[dict[str, float]],
    width: int,
    height: int,
    lag_s: float,
) -> list[dict[str, Any]]:
    """Rows for actions.jsonl, one per frame present in both `boxes` and `t_arrival`."""
    n = min(len(boxes), len(t_arrival))
    t = np.asarray(t_arrival[:n], np.float64) - lag_s  # estimated capture times
    pt = np.array([p["t"] for p in poses])
    gaze = np.stack(
        [
            np.interp(t, pt, [p["yaw"] for p in poses]),
            np.interp(t, pt, [p["pitch"] for p in poses]),
        ],
        1,
    )
    seen = np.array([bool(boxes[i]["present"]) for i in range(n)])
    room: Floats = np.full((n, 2), np.nan)
    for i in np.nonzero(seen)[0]:
        x1, y1, x2, y2 = boxes[i]["box_xyxy"]
        room[i] = gaze[i] + image_angles((x1 + x2) / 2, (y1 + y2) / 2, width, height)
    smooth = room.copy()
    half = SMOOTH // 2
    for i in np.nonzero(seen)[0]:
        window = room[max(0, i - half) : i + half + 1]
        smooth[i] = np.nanmean(window, 0)
    rows = []
    last = -1
    for i in range(n):
        if seen[i]:
            last = i
            j = min(int(np.searchsorted(t, t[i] + LOOKAHEAD_S)), n - 1)
            target, src = (smooth[j] if seen[j] else smooth[i]), "seen"
        else:
            later = np.nonzero(seen[i:] & (t[i:] - t[i] <= HINDSIGHT_S))[0]
            if len(later):
                target, src = smooth[i + int(later[0])], "hindsight"
            else:
                target, src = gaze[i], "none"
        d = target - gaze[i]
        ld = smooth[last] - gaze[i] if last >= 0 else np.zeros(2)
        rows.append(
            {
                "i": i,
                "t": round(float(t[i]), 4),
                "gaze_yaw": round(float(gaze[i][0]), 2),
                "gaze_pitch": round(float(gaze[i][1]), 2),
                "d_yaw": round(_clamp(float(d[0]), MAX_D), 2),
                "d_pitch": round(_clamp(float(d[1]), MAX_D), 2),
                "visible": int(seen[i]),
                "src": src,
                "last_d_yaw": round(_clamp(float(ld[0]), MAX_LAST_D), 2),
                "last_d_pitch": round(_clamp(float(ld[1]), MAX_LAST_D), 2),
                "since_seen_s": round(float(t[i] - t[last]), 3) if last >= 0 else NEVER,
            }
        )
    return rows
