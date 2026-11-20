from typing import Any

import numpy as np

from reachy_demo.attention.hints import BEYOND, CHASE_S, Reading, chasing, with_body
from reachy_demo.perception.objects import ABOVE, Detection, PersonHeads


def _seq(*items: Reading | None) -> Any:
    it = iter(items)
    return lambda: next(it)


def test_body_fills_in_only_while_no_face_and_picks_the_body_nearest_the_last_face() -> None:
    heads = [(-0.6, -0.2, 1.0), (0.5, ABOVE, 1.0)]
    read = with_body(_seq((0.4, 0.0, 0.0), None, (0.1, 0.1, 2.0)), lambda: heads)
    assert read() == (0.4, 0.0, 0.0)
    assert read() == (0.5, ABOVE, 1.0)  # nearest the last face, head above the frame
    assert read() == (0.1, 0.1, 2.0)
    far = with_body(_seq((0.9, 0.0), None), lambda: [(-0.9, 0.0, 1.0)])
    far()
    assert far() is None  # someone else on the other side


def test_chase_keeps_turning_the_way_the_face_left_then_gives_up() -> None:
    now = [0.0]
    readings: list[Reading | None] = [(0.6, 0.2), (0.85, 0.2), None, None, None]
    it = iter(readings)
    read = chasing(lambda: next(it), clock=lambda: now[0])
    for t in (0.0, 0.1):
        now[0] = t
        read()
    now[0] = 0.2
    assert read() == (BEYOND, 0.2)  # left by the right edge
    now[0] = 0.1 + CHASE_S - 0.01
    assert read() == (BEYOND, 0.2)
    now[0] = 0.1 + CHASE_S + 0.01
    assert read() is None


def test_chase_follows_speed_over_position_and_holds_when_lost_in_the_middle() -> None:
    now = [0.0]
    it = iter([(0.3, 0.0), (0.1, 0.0), None])  # moving left fast (2 units/s), still right half
    read = chasing(lambda: next(it), clock=lambda: now[0])
    for t in (0.0, 0.1, 0.2):
        now[0] = t
        r = read()
    assert r == (-BEYOND, 0.0)
    it2 = iter([(0.1, 0.0), (0.1, 0.0), None])  # still, in the middle, then gone: turned away
    read2 = chasing(lambda: next(it2), clock=lambda: now[0])
    read2(), read2()
    assert read2() is None
    it3 = iter([(0.0, -0.7), None])  # left by the top (stood up)
    read3 = chasing(lambda: next(it3), clock=lambda: now[0])
    read3()
    assert read3() == (0.0, -BEYOND)


class FakePeople:
    def __init__(self, boxes: list[tuple[float, float, float, float]]) -> None:
        self.boxes = boxes

    def detect(self, bgr: Any, label: str | None = None) -> list[Detection]:
        assert label == "person" and bgr.shape == (360, 640, 3)  # half-size frame
        return [Detection("person", 0.9, b) for b in self.boxes]


def test_person_heads_aim_at_the_top_of_the_box_or_above_the_frame() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    people = FakePeople([(100, 100, 100, 200), (400, 0, 200, 360)])
    heads = PersonHeads(lambda: frame, lambda: people, clock=lambda: 5.0, lag_s=0.3).read()
    (x1, y1, t1), (x2, y2, _) = heads
    assert t1 == 4.7
    assert round(x1, 2) == round(150 / 639 * 2 - 1, 2)
    assert -1 < y1 < 0 and abs(y1 - (100 / 359 * 2 - 1 + 0.2 * 200 / 359)) < 1e-6
    assert y2 == ABOVE  # box cut off at the top: the head is above the frame
