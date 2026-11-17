"""The real YOLOX-nano on YOLOX's sample image (dog, bicycle, car)."""

from pathlib import Path

import pytest

from reachy_demo.perception.objects import DEFAULT_MODEL, ObjectDetector

SAMPLE = Path("models/yolox/dog.jpg")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (DEFAULT_MODEL.is_file() and SAMPLE.is_file()), reason="model/sample missing"
    ),
]


def test_detects_the_sample_objects() -> None:
    cv2 = pytest.importorskip("cv2")
    found = ObjectDetector().detect(cv2.imread(str(SAMPLE)))
    labels = {d.label for d in found}
    assert {"dog", "bicycle"} <= labels
    dog = next(d for d in found if d.label == "dog")
    x, y, w, h = dog.box
    assert 100 < x < 170 and 180 < y < 240 and w > 150 and h > 300


def test_label_filter() -> None:
    cv2 = pytest.importorskip("cv2")
    found = ObjectDetector().detect(cv2.imread(str(SAMPLE)), "dog")
    assert [d.label for d in found] == ["dog"]
