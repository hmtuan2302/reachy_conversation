"""The real MediaPipe model on MediaPipe's sample portrait: signs match head_pose.py's docs."""

from pathlib import Path

import numpy as np
import pytest

from reachy_demo.perception.head_pose import DEFAULT_MODEL, HeadPose, HeadPoseEstimator

PORTRAIT = Path("models/mediapipe/portrait.jpg")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (DEFAULT_MODEL.is_file() and PORTRAIT.is_file()), reason="model/portrait missing"
    ),
]


def _pose(rotate_ccw: float = 0, flip: bool = False) -> HeadPose:
    pytest.importorskip("mediapipe")
    from PIL import Image

    img = Image.open(PORTRAIT).convert("RGB")
    if flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rotate_ccw:
        img = img.rotate(rotate_ccw, expand=True)
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    from reachy_demo.robot.adapter import face_detector

    est = HeadPoseEstimator(face_detector())
    try:
        pose = est.estimate(bgr, 1.0)
    finally:
        est.close()
    assert pose is not None
    return pose


def test_frontal_portrait_is_near_zero() -> None:
    p = _pose()
    assert abs(p.yaw) < 10 and abs(p.pitch) < 15 and abs(p.roll) < 5


def test_counter_clockwise_rotation_is_positive_roll() -> None:
    assert _pose(rotate_ccw=20).roll == pytest.approx(20, abs=5)
    assert _pose(rotate_ccw=-20).roll == pytest.approx(-20, abs=5)


def test_small_far_face_is_found() -> None:
    """A face ~1.5 m away in a 1280x720 frame: MediaPipe alone misses it."""
    pytest.importorskip("mediapipe")
    from PIL import Image

    from reachy_demo.robot.adapter import face_detector

    face = Image.open(PORTRAIT).convert("RGB").resize((164, 205))
    frame = Image.new("RGB", (1280, 720), (90, 90, 90))
    frame.paste(face, (900, 450))  # right of centre, low
    est = HeadPoseEstimator(face_detector())
    try:
        pose = est.estimate(np.ascontiguousarray(np.asarray(frame)[:, :, ::-1]), 1.0)
    finally:
        est.close()
    assert pose is not None
    assert pose.bearing_yaw < -15  # image right = robot's right
    assert pose.bearing_pitch > 0  # below centre
    assert 100 < pose.distance_cm < 400


def test_mirror_image_flips_yaw_and_roll() -> None:
    a, b = _pose(), _pose(flip=True)
    assert np.sign(a.yaw) == -np.sign(b.yaw)
