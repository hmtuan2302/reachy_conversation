import asyncio
import json
from collections.abc import Callable

import numpy as np
import pytest

from reachy_demo.agent.tools.track import NAME, NOT_OWNER, OBJECT, TrackTools
from reachy_demo.attention.follow import Follow
from reachy_demo.memory.store import Speaker
from reachy_demo.perception.objects import (
    CAMERA_LAG_S,
    COCO,
    MARK_STALE_S,
    MAX_MISSES,
    SIZE,
    Box,
    BoxesDetector,
    BoxTracker,
    CenterFilter,
    Detection,
    ObjectDetector,
    ObjectSource,
    centre,
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


# --- ObjectSource: tracking by detection on a synthetic scene -----------------------------


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


async def test_unsupported_object_is_refused() -> None:
    src, _ = _source(lambda: [], Scene())
    assert (await src.acquire("my keys", "keys") or "").startswith("unsupported_object")


async def test_not_in_view_waits_then_locks_on_when_it_appears() -> None:
    scene, seen = Scene(), [False]
    src, _ = _source(lambda: [Detection("cup", 0.9, scene.half_box())] if seen[0] else [], scene)
    assert await src.acquire("the cup", "cup") is None  # starts anyway
    assert src.waiting and src.switch(True)
    assert src.read() is None
    seen[0] = True
    reading = src.read()
    assert reading is not None and not src.waiting
    assert reading[0] == pytest.approx(360 / 640 - 1, abs=0.05)


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
    x, y, _ = src.read() or (9.0, 9.0, 0.0)
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
    truth = (scene.x + 60) / 640 - 1
    assert reading[0] < truth  # smoothed: trails a moving target
    for _ in range(10):  # still: the filter settles within its dead zone
        reading = src.read()
    assert reading is not None
    assert reading[0] == pytest.approx(truth, abs=0.03)


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
    assert src.read() is not None


def _det(x: float, y: float, size: float = 60.0) -> Detection:
    return Detection("cup", 0.9, (x, y, size, size))


def test_box_tracker_keeps_the_nearest_and_ignores_jumps() -> None:
    tracker = BoxTracker(centre((100.0, 100.0, 60.0, 60.0), 640, 360))
    near, far = _det(110, 100), _det(500, 250, 120)
    assert tracker.select([far, near], 640, 360) is near  # nearest, not largest
    assert tracker.select([far], 640, 360) is None  # too far a jump: a miss
    assert tracker.has_target


def test_box_tracker_drops_after_misses_then_picks_nearest_the_last_sighting() -> None:
    tracker = BoxTracker(centre((100.0, 100.0, 60.0, 60.0), 640, 360))
    for _ in range(MAX_MISSES + 1):
        assert tracker.select([], 640, 360) is None
    assert not tracker.has_target
    assert tracker.select([_det(100, 100, 2)], 640, 360) is None  # a tiny box is not a new target
    mine, bigger = _det(400, 100), _det(500, 250, 120)  # both further than MAX_JUMP
    assert tracker.select([bigger, mine], 640, 360) is mine  # nearest, not largest
    assert tracker.has_target


def test_box_tracker_without_a_seed_picks_the_largest() -> None:
    small, big = _det(100, 100), _det(500, 250, 120)
    assert BoxTracker().select([small, big], 640, 360) is big


def test_box_tracker_bridge_is_bounded_by_misses() -> None:
    tracker = BoxTracker((0.0, 0.0))
    assert not tracker.bridge((0.9, 0.9))  # too far: not ours
    for _ in range(MAX_MISSES):
        tracker.select([], 640, 360)
        assert tracker.bridge((0.01, 0.0))
    tracker.select([], 640, 360)  # one miss too many: dropped
    assert not tracker.bridge((0.01, 0.0))


async def test_csrt_bridges_frames_the_detector_misses() -> None:
    scene = Scene()
    seen = [True]
    src, _ = _source(lambda: [Detection("cup", 0.9, scene.half_box())] if seen[0] else [], scene)
    assert await src.acquire("the cup", "cup") is None
    seen[0] = False  # e.g. a hand over it: the detector misses, the object is still there
    for _ in range(5):
        scene.x += 10
        assert src.read() is not None
    assert src._box is not None
    assert src._box[0] == pytest.approx(scene.x / 2, abs=8)


def test_center_filter_smooths_jitter_and_holds_in_dead_zone() -> None:
    f = CenterFilter()
    assert f.update((0.0, 0.0)) == (0.0, 0.0)
    assert f.update((0.01, 0.0)) == (0.0, 0.0)  # dead zone
    x, _ = f.update((0.1, 0.0))
    assert x == pytest.approx(0.03)  # slow alpha for a small move
    x, _ = f.update((0.5, 0.0))
    assert x == pytest.approx(0.03 + 0.6 * 0.47)  # fast alpha for a big move


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


async def test_follow_waits_longer_for_a_first_sighting() -> None:
    now = [0.0]
    motion = Motion(lambda p: None, hz=1000, clock=FakeClock())
    seen: list[tuple[float, float] | None] = [None]
    follow = Follow(
        motion, lambda: seen[0], lambda on: True, clock=lambda: now[0], poll_s=0.001,
        give_up_s=1.0, first_wait_s=5.0,
    )  # fmt: skip
    assert await follow.start() == (None, True)
    now[0] = 3.0  # past give_up_s, but nothing was ever seen: keep waiting
    await asyncio.sleep(0.05)
    assert follow.active
    now[0] = 6.0  # past first_wait_s
    await _until(lambda: not follow.active)


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


async def test_track_object_not_in_view_starts_waiting() -> None:
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    src, _ = _source(lambda: [], Scene())
    obj = Follow(motion, src.read, src.switch, poll_s=0.001)
    face = Follow(motion, lambda: None, lambda on: True, poll_s=0.001)
    tools = TrackTools(face, objects=(obj, src))
    result = await _call(tools, OBJECT, target="my phone", coco_class="cell phone")
    assert result == {"status": "waiting_for_object", "object": "cell phone"}
    assert obj.active
    obj.stop()
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


async def test_mark_reports_box_score_and_source_then_goes_stale() -> None:
    scene, now, seen = Scene(), [0.0], [True]
    det = FakeDetector(lambda: [Detection("cup", 0.8, scene.half_box())] if seen[0] else [])
    src = ObjectSource(scene.frame, lambda: det, clock=lambda: now[0])
    assert src.mark() is None
    assert await src.acquire("the cup", "cup") is None
    src.read()
    mark = src.mark()
    assert mark is not None and (mark.label, mark.source) == ("cup", "yolox")
    assert mark.score == pytest.approx(0.8)
    assert mark.box == pytest.approx((scene.x, scene.y, scene.size, scene.size))  # full frame
    seen[0] = False
    src.read()
    mark = src.mark()
    assert mark is not None and mark.source == "csrt" and mark.score == pytest.approx(0.8)
    now[0] += MARK_STALE_S + 0.1
    assert src.mark() is None  # stale
    src.release()
    assert src.mark() is None


async def test_read_stamps_the_frame_time_minus_the_camera_lag() -> None:
    """Follow applies the offset to the head pose of that moment; without it a laggy camera
    makes the head swing around a still object (to the wrong side from ~0.3 s)."""
    scene, now = Scene(), [10.0]
    det = FakeDetector(lambda: [Detection("cup", 0.9, scene.half_box())])
    src = ObjectSource(scene.frame, lambda: det, clock=lambda: now[0], lag_s=0.3)
    assert await src.acquire("the cup", "cup") is None
    reading = src.read()
    assert reading is not None and len(reading) == 3
    assert reading[2] == pytest.approx(9.7)


def test_camera_lag_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_demo.agent.cli import camera_lag_from_env

    monkeypatch.delenv("CAMERA_LAG_S", raising=False)
    assert camera_lag_from_env() == CAMERA_LAG_S
    monkeypatch.setenv("CAMERA_LAG_S", "0.45")
    assert camera_lag_from_env() == 0.45
    for bad in ("fast", "-0.1", "5"):
        monkeypatch.setenv("CAMERA_LAG_S", bad)
        assert camera_lag_from_env() is None


def test_boxes_detector_wraps_a_scored_face_detector() -> None:
    det = BoxesDetector("face", lambda bgr: [((1.0, 2.0, 3.0, 4.0), 0.9)])
    assert det.detect(np.zeros((4, 4, 3), np.uint8)) == [Detection("face", 0.9, (1, 2, 3, 4))]


def test_begin_tracks_the_largest_face_with_the_same_pipeline() -> None:
    scene = Scene()
    small = ((10.0, 10.0, 20.0, 20.0), 0.9)
    src = ObjectSource(
        scene.frame, lambda: BoxesDetector("face", lambda f: [small, (scene.half_box(), 0.8)])
    )
    assert src.begin("face") and src.waiting
    reading = src.read()
    assert reading is not None and not src.waiting
    assert reading[0] == pytest.approx(360 / 640 - 1, abs=0.05)  # the big box, not the small one
    mark = src.mark()
    assert mark is not None and mark.label == "face"
    assert not src.switch(False)


def test_begin_fails_small_when_the_detector_cannot_load() -> None:
    def broken() -> BoxesDetector:
        raise OSError("no model")

    assert not ObjectSource(Scene().frame, broken).begin("face")
