from typing import Any

import numpy as np

from reachy_demo.vla.label import (
    EMBED_EVERY,
    GAP_FRAMES,
    Faces,
    gallery_vector,
    label,
    tracks,
)

N = 90
WIDTH = 1280


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


ME, OTHER = _vec(1), _vec(2)


def _faces(people: list[tuple[np.ndarray, float, set[int]]]) -> Faces:
    """people: (identity vector, x start, frames missing). Each walks right 4 px a frame."""
    frame, box, score, emb = [], [], [], []
    for i in range(N):
        for vec, x0, missing in people:
            if i in missing:
                continue
            x = x0 + 4 * i
            frame.append(i)
            box.append([x, 200, x + 120, 340])
            score.append(0.9)
            noisy = vec + 0.02 * _vec(1000 + i)
            due = i % EMBED_EVERY == 0 or i - 1 in missing  # as track_label.detect does
            emb.append(noisy / np.linalg.norm(noisy) if due else [np.nan] * 512)
    return Faces(
        np.array(frame), np.array(box, np.float32), np.array(score, np.float32),
        np.array(emb, np.float32), N, WIDTH,
    )  # fmt: skip


def _meta(**kw: Any) -> dict[str, Any]:
    return {"object_id": "a", "distractor": "none", "negative": False, **kw}


def test_one_track_per_person_survives_short_misses() -> None:
    faces = _faces([(ME, 100, set(range(30, 30 + GAP_FRAMES))), (OTHER, 700, set())])
    assert sorted(len(t.rows) for t in tracks(faces)) == [N - GAP_FRAMES, N]


def test_named_person_is_picked_over_a_second_person_and_gaps_are_filled() -> None:
    gallery = gallery_vector(_faces([(ME, 100, set())]))
    assert gallery is not None and float(gallery @ ME) > 0.9
    faces = _faces([(OTHER, 100, set()), (ME, 700, {40, 41, 42})])
    rows, summary = label(faces, _meta(distractor="second_person"), gallery)
    assert all(r["present"] for r in rows) and summary["flags"] == []
    assert rows[10]["box_xyxy"][0] == 700 + 40 and rows[10]["src"] == "det"
    assert [rows[i]["src"] for i in (40, 41, 42)] == ["interp"] * 3
    assert rows[41]["box_xyxy"][0] == 700 + 4 * 41


def test_long_absence_is_not_present() -> None:
    gallery = gallery_vector(_faces([(ME, 100, set())]))
    gone = set(range(10, 85))  # present 15 of 90 frames
    rows, summary = label(_faces([(ME, 100, gone)]), _meta(), gallery)
    assert [r["present"] for r in rows].count(1) == N - len(gone)
    assert "mostly_absent" in summary["flags"]


def test_negatives_are_absent_and_flag_only_the_named_person() -> None:
    gallery = gallery_vector(_faces([(ME, 100, set())]))
    rows, summary = label(_faces([(OTHER, 100, set())]), _meta(negative=True), gallery)
    assert not any(r["present"] for r in rows) and summary["flags"] == []
    rows, summary = label(_faces([(ME, 100, set())]), _meta(negative=True), gallery)
    assert not any(r["present"] for r in rows) and summary["flags"] == ["negative_seen"]


def test_without_a_gallery_the_longest_track_is_used_and_flagged() -> None:
    rows, summary = label(_faces([(ME, 100, set())]), _meta(), None)
    assert all(r["present"] for r in rows) and summary["flags"] == ["no_gallery"]
    faces = _faces([(ME, 100, set()), (OTHER, 700, set(range(50)))])
    _, summary = label(faces, _meta(distractor="second_person"), None)
    assert summary["flags"] == ["no_gallery", "identity_unsure"]


def test_a_teleport_is_flagged() -> None:
    gallery = gallery_vector(_faces([(ME, 100, set())]))
    # the person's track ends at the left, a new one of them starts at the right next frame
    left = _faces([(ME, 100, set(range(45, N)))])
    right = _faces([(ME, 900, set(range(45)))])
    both = Faces(
        np.concatenate([left.frame, right.frame]), np.concatenate([left.box, right.box]),
        np.concatenate([left.score, right.score]), np.concatenate([left.emb, right.emb]),
        N, WIDTH,
    )  # fmt: skip
    _, summary = label(both, _meta(), gallery)
    assert "jump" in summary["flags"]
