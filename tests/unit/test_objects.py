import asyncio
import json
from collections.abc import Callable

import numpy as np
import pytest

from reachy_demo.agent.tools.track import NAME, NOT_OWNER, OBJECT, TrackTools
from reachy_demo.attention.follow import Follow
from reachy_demo.memory.store import Speaker
from reachy_demo.perception.objects import (
    COCO,
    SIZE,
    Box,
    Detection,
    ObjectDetector,
    ObjectSource,
    decode,
    iou,
)
from reachy_demo.robot.motion import BUSY_LONG, Motion, Pose, dance

pytest.importorskip("cv2")


def test_coco_has_80_classes_in_yolox_order() -> None:
    assert len(COCO) == 80
    assert COCO.index("cell phone") == 67 and COCO.index("cup") == 41


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_decode_one_cell_into_frame_pixels() -> None:
    n = sum((SIZE // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 85), dtype=np.float32)
    raw[:, 2:4] = -10  # tiny boxes everywhere, zero scores
    cell = 10 * 52 + 20  # stride 8 grid: x=20, y=10
    raw[cell, :4] = [0.5, 0.5, np.log(4), np.log(2)]  # centre (164, 84), 32x16 px
    raw[cell, 4] = 0.9
    raw[cell, 5 + COCO.index("cup")] = 0.9
    [d] = decode(raw, scale=0.5, label=None)
    assert d.label == "cup" and d.score == pytest.approx(0.81)
    assert d.box == pytest.approx((296, 152, 64, 32))  # /0.5 back to frame pixels
    assert decode(raw, 0.5, "cell phone") == []


# --- ObjectSource with the real CSRT tracker on a synthetic scene -------------------------


class Scene:
    """A white square on a textured background; `x` moves it."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.bg = rng.integers(0, 80, (720, 1280, 3), dtype=np.uint8)
        self.x, self.y, self.size = 300, 300, 120

    def frame(self) -> np.ndarray:
        f = self.bg.copy()
        f[self.y : self.y + self.size, self.x : self.x + self.size] = 255
        f[self.y + 40 : self.y + 60, self.x + 30 : self.x + 90] = (0, 0, 200)  # texture
        return f

    def half_box(self) -> Box:
        return (self.x / 2, self.y / 2, self.size / 2, self.size / 2)


class FakeDetector(ObjectDetector):
    def __init__(self, found: Callable[[], list[Detection]]) -> None:
        self.found = found
        self.calls = 0

    def detect(self, bgr: np.ndarray, label: str | None = None) -> list[Detection]:
        self.calls += 1
        return [d for d in self.found() if label is None or d.label == label]


def _source(
    found: Callable[[], list[Detection]], scene: Scene, ask: object = None
) -> tuple[ObjectSource, FakeDetector]:
    det = FakeDetector(found)
    return ObjectSource(scene.frame, lambda: det, ask), det  # type: ignore[arg-type]


async def test_unsupported_and_not_visible() -> None:
    scene = Scene()
    src, _ = _source(lambda: [], scene)
    assert (await src.acquire("my keys", "keys") or "").startswith("unsupported_object")
    assert await src.acquire("my phone", "cell phone") == "not_visible: no cell phone in view"
    assert src.read() is None


async def test_several_candidates_ask_the_vision_model() -> None:
    scene = Scene()
    other = Detection("cell phone", 0.9, (500.0, 50.0, 40.0, 60.0))
    mine = Detection("cell phone", 0.5, scene.half_box())
    questions: list[str] = []

    async def ask(jpeg: bytes, q: str) -> str:
        questions.append(q)
        return "Box 2."

    src, _ = _source(lambda: [other, mine], scene, ask)
    assert await src.acquire("my iPhone", "cell phone") is None
    assert "my iPhone" in questions[0]
    x, y = src.read() or (9.0, 9.0)
    # square centre (360, 360) in a 1280x720 frame
    assert x == pytest.approx(360 / 640 - 1, abs=0.05) and y == pytest.approx(0, abs=0.05)


async def test_vision_model_failure_falls_back_to_the_best_score() -> None:
    scene = Scene()
    best = Detection("cup", 0.9, scene.half_box())
    other = Detection("cup", 0.4, (500.0, 50.0, 40.0, 60.0))

    async def ask(jpeg: bytes, q: str) -> str:
        raise RuntimeError("vision down")

    src, _ = _source(lambda: [best, other], scene, ask)
    assert await src.acquire("the cup", "cup") is None
    assert src._box == pytest.approx(best.box)


async def test_tracker_follows_a_moving_object() -> None:
    scene = Scene()
    src, _ = _source(lambda: [Detection("cup", 0.9, scene.half_box())], scene)
    assert await src.acquire("the cup", "cup") is None
    for _ in range(10):
        scene.x += 20
        reading = src.read()
    assert reading is not None
    assert reading[0] == pytest.approx((scene.x + 60) / 640 - 1, abs=0.06)


async def test_lost_object_reads_none_and_redetection_recovers() -> None:
    scene = Scene()
    visible = [True]
    src, _ = _source(lambda: [Detection("cup", 0.9, scene.half_box())] if visible[0] else [], scene)
    assert await src.acquire("the cup", "cup") is None
    visible[0] = False
    scene.size = 0  # gone from the frame
    for _ in range(3):
        src.read()
    scene.size, visible[0] = 120, True
    src._checked = -1e9  # force the next re-check
    assert src.read() is not None


# --- Follow give-up and the tool ----------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


async def _until(cond: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not cond():  # noqa: ASYNC110 - test polling
            await asyncio.sleep(0.001)


async def test_follow_gives_up_on_a_lost_target() -> None:
    lost: list[int] = []
    switch: list[bool] = []
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())

    def turn(on: bool) -> bool:
        switch.append(on)
        return True

    follow = Follow(
        motion,
        lambda: None,
        turn,
        clock=FakeClock(),
        poll_s=0.001,
        give_up_s=1.0,
        on_lost=lambda: lost.append(1),
    )
    assert await follow.start() == (None, True)
    await _until(lambda: not follow.active)
    assert lost == [1] and switch == [True, False]


def _tools(scene: Scene, guest: bool = False) -> tuple[TrackTools, Follow, Motion]:
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    src, _ = _source(lambda: [Detection("cell phone", 0.9, scene.half_box())], scene)
    obj = Follow(motion, src.read, src.switch, poll_s=0.001)
    face = Follow(motion, lambda: None, lambda on: True, poll_s=0.001)
    speaker = (lambda: Speaker(2, "guest")) if guest else (lambda: Speaker(1, "owner"))
    return TrackTools(face, speaker=speaker, objects=(obj, src)), obj, motion


async def _call(tools: TrackTools, name: str, **args: object) -> dict[str, object]:
    return await tools.call(name, json.dumps(args))


async def test_track_object_tool_starts_and_track_face_false_stops_it() -> None:
    tools, obj, _ = _tools(Scene())
    assert OBJECT in [s["function"]["name"] for s in tools.specs()]
    result = await _call(tools, OBJECT, target="my iPhone", coco_class="cell phone")
    assert result == {"status": "tracking", "object": "cell phone"}
    assert obj.active and obj.name == "my iPhone"
    assert await _call(tools, NAME, enabled=False) == {"status": "stopped"}
    await _until(lambda: not obj.active)


async def test_track_object_refusals() -> None:
    tools, obj, motion = _tools(Scene(), guest=True)
    args = {"target": "my iPhone", "coco_class": "cell phone"}
    assert await _call(tools, OBJECT, **args) == NOT_OWNER
    tools, obj, motion = _tools(Scene())
    assert str((await _call(tools, OBJECT, target="x"))["error"]).startswith("bad_arguments")
    assert motion.submit(dance("d", 4.0, lambda t: {})) is None
    assert await _call(tools, OBJECT, **args) == {"error": BUSY_LONG}
    motion.stop()


async def test_none_class_is_refused_without_loading_the_detector() -> None:
    tools, obj, _ = _tools(Scene())
    spec = next(s for s in tools.specs() if s["function"]["name"] == OBJECT)
    assert spec["function"]["parameters"]["properties"]["coco_class"]["enum"][-1] == "none"
    result = await _call(tools, OBJECT, target="my keys", coco_class="none")
    assert result == {"error": "unsupported_object: my keys is not one I can recognise"}
    assert not obj.active and tools._objects is not None
    assert tools._objects[1]._detector is None


async def test_no_camera_no_object_tool() -> None:
    motion = Motion(lambda p: None, hz=1000)
    tools = TrackTools(Follow(motion, lambda: None, lambda on: True))
    assert [s["function"]["name"] for s in tools.specs()] == [NAME]
