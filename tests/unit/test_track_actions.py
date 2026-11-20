import math
from typing import Any

import pytest

from reachy_demo.perception.head_pose import TAN_HALF_H
from reachy_demo.vla.actions import HINDSIGHT_S, MAX_D, NEVER, actions, image_angles

W, H = 1280, 720
FPS = 30.0
LAG = 0.1


def _box_at_yaw(rel_yaw: float) -> list[float]:
    """A face box whose centre is `rel_yaw` degrees left of the image centre, at mid height."""
    x = -math.tan(math.radians(rel_yaw)) / TAN_HALF_H
    u = (x + 1) / 2 * (W - 1)
    return [u - 40, (H - 1) / 2 - 50, u + 40, (H - 1) / 2 + 50]


def _episode(gaze_yaw: list[float], face_yaw: list[float | None]) -> Any:
    n = len(gaze_yaw)
    t_arrival = [i / FPS + LAG for i in range(n)]
    poses = [{"t": i / FPS, "yaw": gaze_yaw[i], "pitch": 0.0} for i in range(n)]
    boxes = [
        {"present": 0, "box_xyxy": None}
        if f is None
        else {"present": 1, "box_xyxy": _box_at_yaw(f - gaze_yaw[i])}
        for i, f in enumerate(face_yaw)
    ]
    return actions(boxes, t_arrival, poses, W, H, LAG)


def test_image_angle_signs() -> None:
    yaw, pitch = image_angles(W - 1, H - 1, W, H)  # bottom right corner
    assert yaw < 0 and pitch > 0  # right of centre = turn right (yaw -), below = look down


def test_a_still_face_gives_the_same_room_direction_while_the_head_turns() -> None:
    n = 60
    gaze = [i * 0.5 for i in range(n)]  # head turning left 15°/s
    rows = _episode(gaze, [10.0] * n)  # face fixed 10° left in the room
    for r in rows[5:-5]:
        assert r["src"] == "seen" and r["visible"] == 1
        assert r["d_yaw"] == pytest.approx(10.0 - r["gaze_yaw"], abs=0.05)
        assert r["since_seen_s"] == 0.0


def test_hindsight_points_where_the_face_shows_up_again() -> None:
    gone = range(30, 60)
    faces: list[float | None] = [
        None if i in gone else (-5.0 if i < 30 else 15.0) for i in range(90)
    ]
    rows = _episode([0.0] * 90, faces)
    r = rows[40]
    assert r["visible"] == 0 and r["src"] == "hindsight"
    assert r["d_yaw"] == pytest.approx(15.0, abs=0.05)  # where it reappears: turn left
    assert r["last_d_yaw"] == pytest.approx(-5.0, abs=0.05)  # where it was last
    assert r["since_seen_s"] == pytest.approx(11 / FPS, abs=1e-3)


def test_no_sighting_in_reach_holds_and_big_turns_are_clamped() -> None:
    n = int((HINDSIGHT_S + 1) * FPS) + 20
    faces: list[float | None] = [None] * (n - 10) + [60.0] * 10
    rows = _episode([0.0] * n, faces)
    assert rows[0]["src"] == "none" and rows[0]["d_yaw"] == 0.0
    assert rows[0]["since_seen_s"] == NEVER
    assert rows[-15]["src"] == "hindsight" and rows[-15]["d_yaw"] == MAX_D
