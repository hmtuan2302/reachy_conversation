import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from reachy_demo.robot.motion import LIMITS, NEUTRAL, clamp
from reachy_demo.vla.record import (
    FACE_FACTORS,
    FACTORS,
    EpisodeWriter,
    jolted,
    load_objects,
    next_episode_dir,
    next_scenario,
    recorded,
    sweep_move,
)

OBJECTS = [
    {"id": "mug_white", "phrasings": ["track the white mug", "follow the mug"], "coco": "cup"},
    {"id": "tape_blue", "phrasings": ["watch the blue tape"], "coco": None},
    {"id": "bottle_green", "phrasings": ["track the green bottle"], "coco": "bottle"},
    {"id": "duck", "phrasings": ["follow the duck"], "held_out": True},
]


def _objects(tmp_path: Path, objects: Any = OBJECTS) -> list[dict[str, Any]]:
    path = tmp_path / "objects.json"
    path.write_text(json.dumps(objects))
    return load_objects(path)


def test_load_objects_rejects_bad_lists(tmp_path: Path) -> None:
    for bad in ([], [{"id": "a"}], [{"id": "a", "phrasings": ["x"], "coco": "unicorn"}]):
        with pytest.raises(ValueError):
            _objects(tmp_path, bad)
    with pytest.raises(ValueError, match="duplicate"):
        _objects(tmp_path, [{"id": "a", "phrasings": ["x"]}] * 2)


def test_sampler_balances_every_factor_and_keeps_held_out_apart(tmp_path: Path) -> None:
    objects = _objects(tmp_path)
    rng = random.Random(0)
    metas: list[dict[str, Any]] = []
    for _ in range(120):
        metas.append(next_scenario(objects, metas, rng))
    for factor, values in FACTORS.items():
        counts = Counter(m[factor] for m in metas)
        assert max(counts.values()) - min(counts[v] for v in values) <= 1, factor
    assert {m["object_id"] for m in metas} == {"mug_white", "tape_blue", "bottle_green"}
    negatives = [m for m in metas if m["negative"]]
    assert len(negatives) == 12 and not metas[0]["negative"]
    assert all(m["head_mode"] == "sweep" for m in negatives)
    assert all(m["head_mode"] == "sweep" for m in metas if m["object_id"] == "tape_blue")
    coco = [m for m in metas if m["coco"] and not m["negative"]]
    share = sum(m["head_mode"] == "baseline_jolt" for m in coco) / len(coco)
    assert 0.5 < share < 0.7
    assert next_scenario(objects, metas, rng, held_out=True)["object_id"] == "duck"


PEOPLE = [
    {"id": "a", "phrasings": ["look at me"], "specific": ["look at the person in blue"]},
    {"id": "b", "phrasings": ["follow me"]},
]


def test_face_targets_use_face_factors_and_specific_phrasings(tmp_path: Path) -> None:
    people = _objects(tmp_path, PEOPLE) and load_objects(tmp_path / "objects.json", face=True)
    assert {p["coco"] for p in people} == {"face"}
    rng = random.Random(3)
    metas: list[dict[str, Any]] = []
    for _ in range(60):
        metas.append(next_scenario(people, metas, rng, factors=FACE_FACTORS))
    for factor, values in FACE_FACTORS.items():
        counts = Counter(m[factor] for m in metas)
        assert max(counts.values()) - min(counts[v] for v in values) <= 1, factor
    assert "mover" not in metas[0]
    for m in metas:
        if m["object_id"] == "a":
            want = "look at the person in blue" if m["distractor"] != "none" else "look at me"
            assert m["task_text"] == want
    assert any(m["head_mode"] == "baseline_jolt" for m in metas)


def test_jolt_shifts_the_reading_for_a_while_then_stops() -> None:
    now = [0.0]
    jr = jolted(lambda: (0.1, 0.0, now[0]), random.Random(1), clock=lambda: now[0])
    shifted = []
    for i in range(200):  # 10 s at 20 Hz
        now[0] = i * 0.05
        r = jr()
        assert r is not None
        shifted.append(abs(r[0] - 0.1) > 1e-9 or abs(r[1]) > 1e-9)
    kicks = sum(1 for a, b in zip([False, *shifted], shifted, strict=False) if b and not a)
    assert 2 <= kicks <= 5 and sum(shifted) < len(shifted) / 2
    assert jolted(lambda: None, random.Random(1))() is None


def test_sweep_stays_inside_the_limits_and_starts_where_the_head_is() -> None:
    for around in (False, True):
        move = sweep_move(around, random.Random(2))
        start = {**NEUTRAL, "yaw": 20.0}
        assert move.pose_at(0.0, start) == pytest.approx(start)
        for i in range(600):
            pose = move.pose_at(i * 0.1, start)
            assert clamp(pose) == pytest.approx(pose), (around, i)
            assert abs(pose["yaw"] - pose["body_yaw"]) <= LIMITS["yaw"]
        bodies = [move.pose_at(i * 0.1, start)["body_yaw"] for i in range(600)]
        assert (max(bodies) > 45) == around


class FakeVideo:
    def __init__(self) -> None:
        self.frames = 0
        self.released = False

    def write(self, img: Any) -> None:
        self.frames += 1

    def release(self) -> None:
        self.released = True


class FakeImg:
    shape = (720, 1280, 3)


def test_writer_layout_and_only_finished_episodes_count(tmp_path: Path) -> None:
    session = tmp_path / "session_x"
    videos: list[FakeVideo] = []

    def open_video(path: Path, w: int, h: int) -> FakeVideo:
        assert (w, h) == (1280, 720) and path.name == "frames.mp4"
        videos.append(FakeVideo())
        return videos[-1]

    ep = EpisodeWriter(next_episode_dir(session), open_video)
    assert ep.path.name == "ep_0001"
    for t in (1.0, 1.04):
        ep.frame(FakeImg(), t)
    ep.pose({"t": 1.0, "yaw": 2.0})
    unfinished = EpisodeWriter(next_episode_dir(session), open_video)
    assert unfinished.path.name == "ep_0002"
    ep.close({"object_id": "mug_white"})
    assert videos[0].frames == 2 and videos[0].released
    lines = (ep.path / "frames.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [
        {"i": 0, "t_arrival": 1.0},
        {"i": 1, "t_arrival": 1.04},
    ]
    assert json.loads((ep.path / "poses.jsonl").read_text()) == {"t": 1.0, "yaw": 2.0}
    assert recorded(tmp_path) == [
        {"object_id": "mug_white", "frames": 2, "frame_size": [1280, 720]}
    ]
    assert next_episode_dir(session).name == "ep_0003"
