"""The user's head pose from one camera frame (F6 mimic). The only module that imports mediapipe.

Two stages, because MediaPipe's own detector shrinks the frame to 128 px and misses a face more
than ~0.5 m away (checked 2026-11-17):

1. YuNet (the SDK's detector, injected from robot/adapter.py) finds the face box in a half-size
   frame. The box gives where the face is (`bearing_*`) and how far (`distance_cm`).
2. MediaPipe's face landmarker runs on a square crop around the box and returns a facial
   transformation matrix (x right, y up, the camera looks down -z). It gives how the head is
   turned **relative to the line of sight** (`yaw`, `pitch`): these don't change when the robot
   turns its own head (the camera turns with it), so mimicking them can't feed back on itself.
   `roll` is in the image, which turns with the camera; attention/mimic.py corrects it.

Signs (checked 2026-11-17 on MediaPipe's portrait.jpg, flipped and rotated):
yaw + = the user turned toward their own left (nose toward image right); pitch + = looking down;
roll + = head tilted counter-clockwise in the image; bearing_yaw + = face in the image's left
(the robot's left); bearing_pitch + = face below the image centre.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# tan(half field of view) of the wireless camera (SDK K: fx 2002 px, cx 1906 px; 16:9 frame).
# shortcut: pinhole without lens distortion; good to a few degrees, calibrate if mimic drifts.
TAN_HALF_H = 0.96
TAN_HALF_V = 0.54
FACE_CM = 15.0  # typical face width: YuNet box width -> distance
CROP = 2.4  # crop side / face box side: the landmarker wants the whole head in view
Box = tuple[float, float, float, float]  # x, y, w, h in pixels
Detect = Callable[[npt.NDArray[np.uint8]], list[Box]]  # BGR frame -> face boxes

DEFAULT_MODEL = Path("models/mediapipe/face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)


@dataclass(frozen=True)
class HeadPose:
    yaw: float  # degrees, see the module docstring for every sign
    pitch: float
    roll: float
    bearing_yaw: float
    bearing_pitch: float
    distance_cm: float


def _heading(v: npt.NDArray[np.float64]) -> tuple[float, float]:
    """(yaw toward +x, pitch toward -y) of a direction, in degrees."""
    return (
        math.degrees(math.atan2(v[0], v[2])),
        math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2]))),
    )


def pose_from_matrix(m: npt.ArrayLike) -> HeadPose:
    """HeadPose from a 4x4 facial transformation matrix."""
    m = np.asarray(m, dtype=np.float64)
    r, t = m[:3, :3], m[:3, 3]
    distance = float(np.linalg.norm(t))
    face_yaw, face_pitch = _heading(r[:, 2])  # where the face points
    look_yaw, look_pitch = _heading(-t / distance)  # facing the camera would point here
    return HeadPose(
        yaw=face_yaw - look_yaw,
        pitch=face_pitch - look_pitch,
        roll=math.degrees(math.atan2(r[1, 0], r[0, 0])),
        bearing_yaw=-math.degrees(math.atan2(t[0], -t[2])),
        bearing_pitch=-math.degrees(math.atan2(t[1], -t[2])),
        distance_cm=distance,
    )


class HeadPoseEstimator:
    """YuNet + MediaPipe face landmarker. Not thread-safe: call it from one thread at a time.
    Close it to free the model (CPU budget, rule 6)."""

    def __init__(self, detect: Detect, model: Path = DEFAULT_MODEL) -> None:
        if not model.is_file():
            raise FileNotFoundError(
                f"face model missing: {model}. Download it: curl -L -o {model} {MODEL_URL}"
            )
        from mediapipe.tasks.python import BaseOptions, vision  # lazy: the `perception` extra

        self._detect = detect
        self._landmarker: Any = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.IMAGE,  # each crop moves: no tracking between
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
        )
        self._last: tuple[float, float] | None = None  # centre of the face followed

    def _pick(self, boxes: list[Box]) -> Box:
        """The face nearest the one followed so far (else the largest): one person at a time."""
        if self._last is None:
            return max(boxes, key=lambda b: b[2] * b[3])
        lx, ly = self._last
        return min(boxes, key=lambda b: (b[0] + b[2] / 2 - lx) ** 2 + (b[1] + b[3] / 2 - ly) ** 2)

    def estimate(self, bgr: npt.NDArray[np.uint8], at_s: float) -> HeadPose | None:
        """Head pose in one BGR frame, or None without a face. `at_s` is unused (IMAGE mode)."""
        import mediapipe as mp

        height, width = bgr.shape[:2]
        boxes = self._detect(np.ascontiguousarray(bgr[::2, ::2]))  # half size: ~4x faster
        if not boxes:
            self._last = None
            return None
        x, y, w, h = (2 * v for v in self._pick(boxes))
        cx, cy = x + w / 2, y + h / 2
        self._last = (cx / 2, cy / 2)
        half = CROP * max(w, h) / 2
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, width))
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, height))
        rgb = np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1])
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.facial_transformation_matrixes:
            return None
        turn = pose_from_matrix(result.facial_transformation_matrixes[0])
        fx = width / 2 / TAN_HALF_H
        return HeadPose(
            yaw=turn.yaw,
            pitch=turn.pitch,
            roll=turn.roll,
            bearing_yaw=-math.degrees(math.atan((cx - width / 2) / fx)),
            bearing_pitch=math.degrees(math.atan((cy - height / 2) / fx)),
            distance_cm=FACE_CM * fx / max(w, 1.0),
        )

    def close(self) -> None:
        self._landmarker.close()
