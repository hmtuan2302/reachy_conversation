"""Face labels for recorded episodes (plan §7.1, faces): which face is the named person, per frame.

Input per episode: every face SCRFD found (`faces.npz`: frame index, box, score, and an ArcFace
vector on every EMBED_EVERY-th frame). No models here, so tests use made-up detections.

1. **Tracks:** faces are linked frame to frame by box overlap (a track survives GAP_FRAMES
   missed frames), so one identity check covers the whole track.
2. **Who:** a person's gallery is the mean ArcFace vector of their track in their own
   episodes with nobody else in view (distractor "none", not negative). In every episode, the
   track most like the named person's gallery (cosine >= MIN_SIM) is the target, per frame.
   With no gallery yet, the longest track is taken and the episode is flagged.
3. **Gaps:** short misses (<= GAP_FRAMES) inside a track are filled by interpolation; other
   frames are present = 0. Negative episodes are present = 0 throughout.
4. **Flags:** anything a person should look at (§7.3).

Output rows (`boxes.jsonl`): {i, box_xyxy or null, det_score, sim, present, src} with src
"det" (detected), "interp" (filled gap) or "none".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

EMBED_EVERY = 10  # frames between ArcFace runs (~90 ms per face on CPU)
GAP_FRAMES = 6  # ~0.2 s at 30 fps: longer misses are "not present"
MIN_IOU = 0.3
MIN_SIM = 0.3  # ArcFace cosine: same person ~0.4-0.9, others < 0.2
UNSURE_SIM = 0.1  # rival within this of the target's similarity: flag
MIN_GALLERY_FRAMES = 30  # a gallery track must be this long (1 s)
JUMP_FRAC = 0.25  # centre moves more than this share of the frame width in one frame: flag
MIN_PRESENT = 0.25  # a normal episode with the target in fewer frames than this: flag
NEGATIVE_SEEN = 0.05  # a negative episode with the target in more frames than this: flag

Floats = npt.NDArray[np.float32]


@dataclass
class Faces:
    """Every face found in one episode, as flat arrays (one row per face)."""

    frame: npt.NDArray[np.int64]  # (N,)
    box: Floats  # (N, 4) x1, y1, x2, y2
    score: Floats  # (N,)
    emb: Floats  # (N, 512), NaN rows where no vector was computed
    n_frames: int
    width: int

    @classmethod
    def empty(cls, n_frames: int, width: int) -> Faces:
        f32 = np.float32
        return cls(
            np.zeros(0, np.int64), np.zeros((0, 4), f32), np.zeros(0, f32),
            np.zeros((0, 512), f32), n_frames, width,
        )  # fmt: skip


@dataclass
class Track:
    rows: list[int] = field(default_factory=list)  # indices into Faces, in frame order

    def last(self, faces: Faces) -> int:
        return int(faces.frame[self.rows[-1]])


def iou(a: Floats, b: Floats) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = float(w * h)
    area = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area - inter)


def tracks(faces: Faces) -> list[Track]:
    """Link faces into tracks: greedy best-overlap match to tracks seen in the last GAP_FRAMES."""
    done: list[Track] = []
    live: list[Track] = []
    for i in range(faces.n_frames):
        rows = np.nonzero(faces.frame == i)[0]
        done += [t for t in live if i - t.last(faces) > GAP_FRAMES + 1]
        live = [t for t in live if i - t.last(faces) <= GAP_FRAMES + 1]
        pairs = sorted(
            ((iou(faces.box[t.rows[-1]], faces.box[r]), k, int(r))
             for k, t in enumerate(live) for r in rows),
            reverse=True,
        )  # fmt: skip
        used_t: set[int] = set()
        used_r: set[int] = set()
        for overlap, k, r in pairs:
            if overlap < MIN_IOU or k in used_t or r in used_r:
                continue
            live[k].rows.append(r)
            used_t.add(k)
            used_r.add(r)
        live += [Track([int(r)]) for r in rows if int(r) not in used_r]
    return sorted(done + live, key=lambda t: t.last(faces))


def track_vector(faces: Faces, t: Track) -> Floats | None:
    """Mean ArcFace vector of a track (unit length), None if none was computed on it."""
    e = faces.emb[t.rows]
    e = e[~np.isnan(e[:, 0])]
    if not len(e):
        return None
    v = e.mean(0)
    out: Floats = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def gallery_vector(faces: Faces) -> Floats | None:
    """The person's vector from an episode where only they are in view: their longest track."""
    ts = [t for t in tracks(faces) if len(t.rows) >= MIN_GALLERY_FRAMES]
    if not ts:
        return None
    return track_vector(faces, max(ts, key=lambda t: len(t.rows)))


def mean_vector(vectors: list[Floats]) -> Floats | None:
    if not vectors:
        return None
    v = np.mean(vectors, 0)
    out: Floats = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def label(
    faces: Faces, meta: dict[str, Any], gallery: Floats | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame rows for boxes.jsonl, and a summary with flags for label.json."""
    ts = tracks(faces)
    sims = [None if (v := track_vector(faces, t)) is None or gallery is None else float(v @ gallery)
            for t in ts]  # fmt: skip
    flags: list[str] = []
    target: dict[int, tuple[int, float | None]] = {}  # frame -> (face row, similarity)
    if gallery is None:
        flags.append("no_gallery")
        if ts:
            best = max(range(len(ts)), key=lambda k: float(faces.score[ts[k].rows].sum()))
            target = {int(faces.frame[r]): (r, None) for r in ts[best].rows}
            if len(ts) > 1 and meta.get("distractor") != "none":
                flags.append("identity_unsure")
    else:
        unsure = False
        for k in sorted(range(len(ts)), key=lambda k: sims[k] or -1.0):  # best last: it wins
            s = sims[k]
            if s is None or s < MIN_SIM:
                continue
            for r in ts[k].rows:
                i = int(faces.frame[r])
                if i in target and (other := target[i][1]) is not None and s - other < UNSURE_SIM:
                    unsure = True
                target[i] = (r, s)
        if unsure:
            flags.append("identity_unsure")
        if any(s is not None and MIN_SIM - UNSURE_SIM <= s < MIN_SIM for s in sims):
            flags.append("near_threshold")
    if meta.get("negative"):
        seen = len(target) / max(faces.n_frames, 1)
        if gallery is not None and seen > NEGATIVE_SEEN:  # without one, any face may be the other
            flags.append("negative_seen")  # the named person was in view: re-record or delete
        target = {}
    rows = _rows(faces, target)
    present = sum(r["present"] for r in rows) / max(len(rows), 1)
    if not meta.get("negative") and present < MIN_PRESENT:
        flags.append("mostly_absent")
    if _jumps(rows, faces.width):
        flags.append("jump")
    summary = {
        "present_frac": round(present, 3),
        "tracks": len(ts),
        "target_sim": max((s for s in sims if s is not None), default=None),
        "flags": flags,
    }
    return rows, summary


def _rows(faces: Faces, target: dict[int, tuple[int, float | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(faces.n_frames):
        if i in target:
            r, s = target[i]
            box = [round(float(v), 1) for v in faces.box[r]]
            sim = None if s is None else round(s, 3)
            rows.append({"i": i, "box_xyxy": box, "det_score": round(float(faces.score[r]), 3),
                         "sim": sim, "present": 1, "src": "det"})  # fmt: skip
        else:
            rows.append({"i": i, "box_xyxy": None, "det_score": None, "sim": None, "present": 0,
                         "src": "none"})  # fmt: skip
    # fill short gaps between two detections of the target
    seen = [i for i in range(faces.n_frames) if i in target]
    for a, b in zip(seen, seen[1:], strict=False):
        if 1 < b - a <= GAP_FRAMES + 1:
            ba, bb = np.array(rows[a]["box_xyxy"]), np.array(rows[b]["box_xyxy"])
            for i in range(a + 1, b):
                f = (i - a) / (b - a)
                rows[i].update(box_xyxy=[round(float(v), 1) for v in ba + (bb - ba) * f],
                               present=1, src="interp")  # fmt: skip
    return rows


def _jumps(rows: list[dict[str, Any]], width: int) -> bool:
    prev = None
    for r in rows:
        if not r["present"]:
            prev = None
            continue
        x = (r["box_xyxy"][0] + r["box_xyxy"][2]) / 2
        y = (r["box_xyxy"][1] + r["box_xyxy"][3]) / 2
        if prev is not None and np.hypot(x - prev[0], y - prev[1]) > JUMP_FRAC * width:
            return True
        prev = (x, y)
    return False
