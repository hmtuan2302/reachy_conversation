import asyncio
import json
import math
from collections.abc import Callable

import numpy as np
import pytest

from reachy_demo.agent.tools.find import NAME, NOT_OWNER, FindTools
from reachy_demo.attention.search import SWEEP_S, Search, Sweep, height, path
from reachy_demo.memory.store import Speaker
from reachy_demo.perception.head_pose import TAN_HALF_H
from reachy_demo.perception.objects import Detection, ObjectDetector
from reachy_demo.robot.motion import BUSY_LONG, NEUTRAL, Motion, Pose, dance

pytest.importorskip("cv2")

W, H = 1280, 720


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


class Room(ObjectDetector):
    """Cups at world yaws `at` (eye level), drawn where a pinhole camera on the head sees them."""

    def __init__(
        self, motion: Motion, at: float | None | list[float], score: float = 0.9, n: int = 1
    ):
        self.motion, self.score, self.n = motion, score, n
        self.at = [] if at is None else [at] if isinstance(at, float) else at
        self.loads = 0

    def detect(self, bgr: np.ndarray, label: str | None = None) -> list[Detection]:
        if bgr.shape[1] != W or label != "cup":  # tiles see nothing extra
            return []
        out = []
        for at in self.at:
            off = at - self.motion.pose["yaw"]
            if abs(off) > 40:  # outside the camera's view
                continue
            cx = W / 2 - (W / 2 / TAN_HALF_H) * math.tan(math.radians(off))
            out += [
                Detection("cup", self.score, (cx - 30 + 150 * i, 320.0, 60.0, 80.0))
                for i in range(self.n)
            ]
        return out


def _search(
    at: float | None | list[float],
    ask: Callable[..., object] | None = None,
    clock: Callable[[], float] | None = None,
    **room: float,
) -> tuple[Search, Motion, list[Pose], Room]:
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    det = Room(motion, at, **room)  # type: ignore[arg-type]

    def make() -> Room:
        det.loads += 1
        return det

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    s = Search(
        motion,
        lambda: frame,
        make,
        ask,  # type: ignore[arg-type]
        clock=clock or FakeClock(),
        settle_s=0.0,
        brake_s=0.0,
    )
    return s, motion, poses, det


def test_sweep_path_reaches_both_sides_within_limits_and_starts_and_ends_still() -> None:
    ts = np.linspace(0, SWEEP_S, 4001)
    poses = [path(float(t)) for t in ts]
    yaws = np.array([p["yaw"] for p in poses])
    assert yaws.max() == pytest.approx(160, abs=0.5) and yaws.min() == pytest.approx(-160, abs=0.5)
    assert all(abs(p["body_yaw"]) <= 120 + 1e-6 for p in poses)
    assert all(abs(p["yaw"] - p["body_yaw"]) <= 40 + 1e-6 for p in poses)
    speed = np.abs(np.diff(yaws)) / (ts[1] - ts[0])
    assert speed[0] < 1 and speed[-1] < 1  # starts and ends at rest
    assert speed.max() < 80  # deg/s
    assert height(25) == "low, below my eye level" and height(0) == "at my eye level"


def test_sweep_length_scales_speed_and_is_validated() -> None:
    slow = [path(float(t), 40.0)["yaw"] for t in np.linspace(0, 40.0, 4001)]
    speed = np.abs(np.diff(slow)) / 0.01
    assert speed.max() < 40 and max(slow) == pytest.approx(160, abs=0.5)
    sweep = Sweep(40.0)
    sweep.time = 39.9
    assert not sweep.done
    sweep.time = 40.0
    assert sweep.done
    motion = Motion(lambda p: None, hz=1000)
    with pytest.raises(ValueError, match="8-120"):
        Search(motion, lambda: None, ObjectDetector, sweep_s=5.0)


def test_sweep_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_demo.agent.cli import search_sweep_from_env

    monkeypatch.delenv("SEARCH_SWEEP_S", raising=False)
    assert search_sweep_from_env() == SWEEP_S
    monkeypatch.setenv("SEARCH_SWEEP_S", "30")
    assert search_sweep_from_env() == 30.0
    for bad in ("3", "abc", "500"):
        monkeypatch.setenv("SEARCH_SWEEP_S", bad)
        assert search_sweep_from_env() is None


def test_brake_slows_smoothly_and_stops_short() -> None:
    sweep = Sweep()
    move = sweep.move()
    t, yaws = 0.0, []
    while t < 5.0:
        yaws.append(move.pose_at(t, dict(NEUTRAL))["yaw"])
        t += 0.02
    sweep.brake()
    for _ in range(60):
        yaws.append(move.pose_at(t, dict(NEUTRAL))["yaw"])
        t += 0.02
    steps = np.abs(np.diff(yaws))
    braking = steps[len(steps) - 60 :]
    assert braking[-1] < 0.01  # at rest
    assert all(b <= a + 1e-9 for a, b in zip(braking, braking[1:], strict=False))  # no jolt
    assert steps[-61] - braking[0] < 0.3  # speed continuous when the brake starts


async def test_finds_in_front_without_asking_and_centres_it() -> None:
    s, motion, poses, _ = _search(0.0)
    result = await s.find("the cup", "cup")
    assert result["status"] == "found"
    assert result["where"] == "in front of me"


async def test_finds_behind_to_the_left_with_the_body() -> None:
    s, motion, poses, _ = _search(110.0)
    result = await s.find("the cup", "cup")
    assert result["status"] == "found"
    assert "to my left, behind me" in str(result["where"])
    assert motion.pose["body_yaw"] > 50
    assert all(abs(p["yaw"] - p["body_yaw"]) <= 45 + 1e-6 for p in poses)
    yaws = np.array([p["yaw"] for p in poses])
    assert np.abs(np.diff(yaws)).max() < 5  # <= 90°/s at 50 ms fake-clock ticks: no jumps


async def test_not_found_returns_to_neutral() -> None:
    s, motion, poses, det = _search(None)
    result = await s.find("the cup", "cup")
    assert result["status"] == "not_found"
    await motion.settled()
    assert poses[-1] == pytest.approx(NEUTRAL)


async def test_my_object_and_several_candidates_ask_the_vision_model() -> None:
    questions: list[str] = []

    async def ask(jpeg: bytes, q: str) -> str:
        questions.append(q)
        return "a red cup on the desk" if q.startswith("Describe") else "2"

    s, *_ = _search(0.0, ask, n=2)
    result = await s.find("my cup", "cup")
    assert result["status"] == "found"
    assert result["description"] == "a red cup on the desk"
    assert "my cup" in questions[0]


async def test_rejected_look_alike_is_skipped_and_the_sweep_resumes() -> None:
    asked: list[int] = []

    async def ask(jpeg: bytes, q: str) -> str:
        asked.append(1)
        return "0"  # a look-alike

    s, motion, poses, _ = _search(0.0, ask, score=0.4)
    assert (await s.find("the cup", "cup"))["status"] == "not_found"
    assert len(asked) == 1  # its direction is skipped afterwards
    yaws = [p["yaw"] for p in poses]
    assert max(yaws) > 150 and min(yaws) < -150  # the sweep went on, both sides


async def _describing(jpeg: bytes, q: str) -> str:
    return "a cup" if q.startswith("Describe") else "1"


def _passes(poses: list[Pose]) -> int:
    """Times the head reached the far left."""
    yaws = [p["yaw"] for p in poses]
    return sum(1 for a, b in zip(yaws, yaws[1:], strict=False) if a < 150 <= b)


async def test_no_resumes_past_the_wrong_object_to_the_next() -> None:
    s, motion, poses, _ = _search([0.0, 110.0], _describing)
    first = await s.find("my cup", "cup")
    assert first["status"] == "found" and first["where"] == "in front of me"
    second = await s.find("my cup", "cup", resume=True)  # "no, not that one"
    assert second["status"] == "found"
    assert "to my left, behind me" in str(second["where"])


async def test_two_sweeps_then_not_found() -> None:
    s, motion, poses, _ = _search(None)
    result = await s.find("the cup", "cup")
    assert result["status"] == "not_found" and result["sweeps"] == 2
    assert _passes(poses) == 2


async def test_resume_counts_toward_the_two_sweeps() -> None:
    s, motion, poses, _ = _search([110.0], _describing)
    assert (await s.find("my cup", "cup"))["status"] == "found"
    result = await s.find("my cup", "cup", resume=True)
    assert result["status"] == "not_found" and result["sweeps"] == 2
    assert _passes(poses) == 2  # the first sweep's left side, then one more sweep


async def test_resume_after_a_minute_or_for_another_object_starts_over() -> None:
    t = [0.0]
    s, motion, poses, _ = _search([0.0], _describing, clock=lambda: t[0])
    assert (await s.find("my cup", "cup"))["status"] == "found"
    t[0] += 61
    again = await s.find("my cup", "cup", resume=True)  # too late: fresh search
    assert again["status"] == "found" and again["where"] == "in front of me"
    assert (await s.find("the bottle", "bottle", resume=True))["status"] == "not_found"


async def test_refusals() -> None:
    s, motion, _, det = _search(0.0)
    assert str((await s.find("keys", "keys"))["error"]).startswith("unsupported_object")
    assert motion.submit(dance("d", 4.0, lambda t: {})) is None
    assert await s.find("the cup", "cup") == {"error": BUSY_LONG}
    assert det.loads == 0  # refused before loading the detector
    motion.stop()


async def test_missing_model_is_reported() -> None:
    motion = Motion(lambda p: None, hz=1000, clock=FakeClock())

    def make() -> ObjectDetector:
        raise FileNotFoundError("object model missing")

    s = Search(motion, lambda: None, make)
    assert str((await s.find("the cup", "cup"))["error"]).startswith("find_unavailable")


async def test_stop_cancels_the_scan_and_faces_front() -> None:
    s, motion, poses, _ = _search(None)
    task = asyncio.create_task(s.find("the cup", "cup"))
    async with asyncio.timeout(5):
        while abs(motion.pose["yaw"]) < 30:  # noqa: ASYNC110 - test polling
            await asyncio.sleep(0.001)
    task.cancel()  # what barge-in does to the running answer
    with pytest.raises(asyncio.CancelledError):
        await task
    await motion.settled()
    assert poses[-1] == pytest.approx(NEUTRAL)


async def _call(tools: FindTools, **args: object) -> dict[str, object]:
    return await tools.call(NAME, json.dumps(args))


async def test_tool() -> None:
    s, *_ = _search(0.0)
    tools = FindTools(s)
    assert (await _call(tools, target="the cup", coco_class="cup"))["status"] == "found"
    spec = tools.specs()[0]["function"]["parameters"]["properties"]
    assert spec["resume"]["type"] == "boolean"
    none = await _call(tools, target="my keys", coco_class="none")
    assert none == {"error": "unsupported_object: my keys is not one I can recognise"}
    assert str((await _call(tools, target="x"))["error"]).startswith("bad_arguments")
    guest = FindTools(s, speaker=lambda: Speaker(2, "guest"))
    assert await _call(guest, target="the cup", coco_class="cup") == NOT_OWNER
