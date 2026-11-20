"""Raw episodes for the VLA tracking dataset (pipeline doc §6; run by scripts/track_record.py).

What is here has no SDK or camera import, so tests use fakes:

- the scenario sampler: picks each episode's object, motion, distractor... as the value used
  least so far over every session already on disk, so coverage evens out (§6.4);
- the two head modes: the object tracker with random jolts, and slow sweeps (§6.2);
- the episode writer: the raw layout of §6.3, written as-is.

Teleop (mouse) is not built yet: non-COCO objects get sweeps only.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from reachy_demo.attention.hints import Reading as HintReading
from reachy_demo.perception.head_pose import TAN_HALF_H
from reachy_demo.perception.objects import COCO
from reachy_demo.robot.motion import NEUTRAL, Move, Pose

RAW_DIR = Path("data/track/raw")
FACE_RAW_DIR = Path("data/track_face/raw")
Factors = dict[str, tuple[str, ...]]
# Factors that change per episode; lighting and background are set per session (§6.4).
FACTORS: Factors = {
    "motion": ("still", "slow_slide", "fast_pass", "circle", "leave_return", "carried"),
    "mover": ("hand", "string", "toy_car", "turntable"),
    "distractor": ("none", "similar_object", "person_behind"),
    "start": ("centred", "off_centre", "half_out"),
    "range": ("head", "around"),
}
# Face tracking: the target is a person (docs/usage/track-record-face.md).
FACE_FACTORS: Factors = {
    "motion": (
        "still",
        "small_moves",
        "walk_across",
        "approach_retreat",
        "sit_stand",
        "leave_return",
        "turn_away",
        "fast_exit",
    ),
    "distance": ("near", "mid", "far"),
    "appearance": ("as_is", "glasses", "hat", "hand_on_face"),
    "distractor": ("none", "second_person", "face_picture"),
    "start": ("centred", "off_centre", "half_out", "body_only"),
    "range": ("head", "around"),
}
LIGHTING = ("daylight", "lamp", "backlit", "dim")
BACKGROUND = ("clean_desk", "cluttered_desk", "other_room")
FACE_BACKGROUND = ("plain_wall", "cluttered_room", "window", "other_room")
# Target shares of head modes for COCO objects (§6.2, without the 20% teleop not built yet).
HEAD_SHARES = {"baseline_jolt": 0.6, "sweep": 0.4}
NEGATIVE_SHARE = 0.1  # episodes where the named object is not there (present = 0)
JOLT_DEG = 10.0  # §6.2: a ±10° kick every 2-5 s
JOLT_EVERY_S = (2.0, 5.0)
JOLT_S = 0.6
SWEEP_EASE_S = 1.0
VIDEO_FPS = 30  # nominal only: frames.jsonl has each frame's real time


def load_objects(path: Path, face: bool = False) -> list[dict[str, Any]]:
    """The target list: [{"id", "phrasings": [...], "specific": [...], "coco": label or null,
    "held_out": bool}]. `specific`: phrasings that tell it from a look-alike, used when the
    episode has a distractor. `face`: the targets are people, tracked by the face detector."""
    objects = json.loads(path.read_text())
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"{path}: expected a non-empty JSON list of objects")
    ids = set()
    for o in objects:
        if not isinstance(o, dict) or not isinstance(o.get("id"), str) or not o["id"]:
            raise ValueError(f"{path}: every object needs a string id: {o!r}")
        if o["id"] in ids:
            raise ValueError(f"{path}: duplicate id {o['id']}")
        ids.add(o["id"])
        phrasings = o.get("phrasings")
        if not isinstance(phrasings, list) or not phrasings:
            raise ValueError(f"{o['id']}: needs a non-empty phrasings list")
        if not isinstance(o.get("specific", []), list):
            raise ValueError(f"{o['id']}: specific must be a list")
        if face:
            o["coco"] = "face"  # BoxesDetector's label: every person is trackable
        elif o.get("coco") is not None and o["coco"] not in COCO:
            raise ValueError(f"{o['id']}: coco {o['coco']!r} is not a COCO class")
        o.setdefault("coco", None)
        o["held_out"] = bool(o.get("held_out", False))
    return objects


def recorded(raw_dir: Path) -> list[dict[str, Any]]:
    """The meta.json of every episode on disk, all sessions."""
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*/ep_*/meta.json"))]


def _least(
    values: tuple[str, ...] | list[str],
    used: Counter[str],
    rng: random.Random,
    shares: dict[str, float] | None = None,
) -> str:
    """The value furthest below its share so far (equal shares by default); ties at random."""

    def lag(v: str) -> float:
        return used[v] / (shares[v] if shares else 1.0)

    low = min(lag(v) for v in values)
    return rng.choice([v for v in values if lag(v) == low])


def next_scenario(
    objects: list[dict[str, Any]],
    metas: list[dict[str, Any]],
    rng: random.Random,
    held_out: bool = False,
    factors: Factors = FACTORS,
) -> dict[str, Any]:
    """The next episode's scenario fields, balanced against the episodes in `metas`.

    `held_out` picks only held-out objects (evaluation sessions); otherwise they are never picked.
    """
    pool = [o for o in objects if o["held_out"] == held_out]
    if not pool:
        raise ValueError("no " + ("held-out" if held_out else "training") + " objects in the list")
    counts: dict[str, Counter[str]] = {
        k: Counter(str(m.get(k)) for m in metas)
        for k in (*factors, "object_id", "head_mode", "task_text")
    }
    pick = _least([o["id"] for o in pool], counts["object_id"], rng)
    obj = next(o for o in pool if o["id"] == pick)
    scenario: dict[str, Any] = {k: _least(v, counts[k], rng) for k, v in factors.items()}
    negatives = sum(bool(m.get("negative")) for m in metas)
    negative = negatives + 1 <= NEGATIVE_SHARE * (len(metas) + 1)  # 1 in 10, never the first
    if negative or obj["coco"] is None:
        head_mode = "sweep"  # the tracker needs a COCO class and an object in view
    else:
        coco = _coco_ids(objects)
        mode_counts = Counter(
            str(m.get("head_mode"))
            for m in metas
            if m.get("object_id") in coco and not m.get("negative")
        )
        head_mode = _least(list(HEAD_SHARES), mode_counts, rng, HEAD_SHARES)
    # With a look-alike in view, "the mug" / "look at me" is ambiguous: use a specific phrasing.
    specific = obj.get("specific") or []
    phrasings = specific if specific and scenario["distractor"] != "none" else obj["phrasings"]
    scenario.update(
        object_id=obj["id"],
        coco=obj["coco"],
        task_text=_least(phrasings, counts["task_text"], rng),
        head_mode=head_mode,
        negative=negative,
        held_out=held_out,
    )
    return scenario


def _coco_ids(objects: list[dict[str, Any]]) -> set[str]:
    return {o["id"] for o in objects if o["coco"] is not None}


Reading = HintReading | None  # (x, y in [-1, 1], optional frame time) as Follow reads


def jolted(
    read: Callable[[], Reading],
    rng: random.Random,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[], Reading]:
    """`read`, but every 2-5 s the target is reported JOLT_DEG off for JOLT_S.

    The tracker steers toward the false spot, then recovers when it ends: a kick plus a
    recovery, with the tracker's own lag correction kept.
    """
    state = {"next": clock() + rng.uniform(*JOLT_EVERY_S), "dx": 0.0, "dy": 0.0}
    shift = math.tan(math.radians(JOLT_DEG)) / TAN_HALF_H  # degrees -> image units (approx.)

    def read_jolted() -> Reading:
        r = read()
        now = clock()
        if now >= state["next"]:
            angle = rng.uniform(0, 2 * math.pi)
            state.update(dx=shift * math.cos(angle), dy=shift * math.sin(angle) / 2)
            state["end"] = now + JOLT_S
            state["next"] = now + rng.uniform(*JOLT_EVERY_S)
        if r is None or now >= state.get("end", 0.0):
            return r
        return (r[0] + state["dx"], r[1] + state["dy"], *r[2:])

    return read_jolted


def sweep_move(around: bool, rng: random.Random, max_s: float = 600.0) -> Move:
    """A slow Lissajous sweep of the gaze, eased in from the start pose; stopped from outside.

    `around`: the body turns too (±90°), so objects pass beyond the head's reach (§6.4 range).
    """
    ty, tp, tb = rng.uniform(8, 14), rng.uniform(5, 9), rng.uniform(18, 26)  # periods, s
    phase = rng.uniform(0, 2 * math.pi)

    def pose_at(t: float, start: Pose) -> Pose:
        body = 90.0 * math.sin(2 * math.pi * t / tb) if around else 0.0
        target = {
            **NEUTRAL,
            "body_yaw": body,
            "yaw": body + 35.0 * math.sin(2 * math.pi * t / ty + phase),  # yaw is the world gaze
            "pitch": 15.0 * math.sin(2 * math.pi * t / tp),
        }
        f = min(t / SWEEP_EASE_S, 1.0)
        return {k: start[k] + (target[k] - start[k]) * f for k in target}

    return Move("track sweep", True, max_s, pose_at, dict(NEUTRAL))


class Video(Protocol):
    def write(self, img: Any) -> None: ...
    def release(self) -> None: ...


OpenVideo = Callable[[Path, int, int], Video]  # (path, width, height) -> writer


def next_episode_dir(session_dir: Path) -> Path:
    taken = [int(p.name[3:]) for p in session_dir.glob("ep_*") if p.name[3:].isdigit()]
    return session_dir / f"ep_{max(taken, default=0) + 1:04d}"


class EpisodeWriter:
    """One episode folder: frames.mp4 + frames.jsonl, poses.jsonl, then meta.json on close.

    `frame` and `pose` are called from their own threads. A folder without meta.json is an
    unfinished episode (crash, Ctrl+C) and is not counted by `recorded`.
    """

    def __init__(self, path: Path, open_video: OpenVideo) -> None:
        self.path = path
        path.mkdir(parents=True)
        self._open_video = open_video
        self._video: Video | None = None
        self._frames = (path / "frames.jsonl").open("w")
        self._poses = (path / "poses.jsonl").open("w")
        self._pose_lock = threading.Lock()
        self.n_frames = 0
        self.size: tuple[int, int] | None = None  # (width, height)

    def frame(self, img: Any, t: float) -> None:
        if self._video is None:
            h, w = img.shape[:2]
            self.size = (w, h)
            self._video = self._open_video(self.path / "frames.mp4", w, h)
        self._video.write(img)
        self._frames.write(json.dumps({"i": self.n_frames, "t_arrival": t}) + "\n")
        self.n_frames += 1

    def pose(self, row: dict[str, float]) -> None:
        with self._pose_lock:
            self._poses.write(json.dumps(row) + "\n")

    def close(self, meta: dict[str, Any]) -> None:
        """Finish the files and write meta.json (the episode then counts)."""
        if self._video is not None:
            self._video.release()
        self._frames.close()
        self._poses.close()
        meta = {**meta, "frames": self.n_frames, "frame_size": self.size}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
