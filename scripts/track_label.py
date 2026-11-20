"""Label recorded face episodes (docs/usage/track-label.md): boxes.jsonl + actions.jsonl.

Run from the repo root after a recording session (CPU, about 1 min per 25 s episode):
    uv run python scripts/track_label.py
Pass 1 finds every face in every new episode (SCRFD; ArcFace every 10th frame and whenever the
number of faces changes) and caches it in faces.npz. Pass 2 builds each person's face vector
from their episodes with nobody else in view, then labels every episode: the named person's box
per frame, present 0/1, and flags. Rerunning is quick: only new episodes are detected again
(--redetect forces it).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from reachy_demo.vla.actions import actions
from reachy_demo.vla.label import EMBED_EVERY, Faces, gallery_vector, label, mean_vector
from reachy_demo.vla.record import FACE_RAW_DIR

LABEL_VERSION = 2  # 2: actions.jsonl with hindsight and last-seen


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def write_actions(ep: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """actions.jsonl from the boxes and the head poses; a summary for label.json."""
    frames, poses = jsonl(ep / "frames.jsonl"), jsonl(ep / "poses.jsonl")
    if not frames or not poses or not meta.get("frame_size"):
        return {"actions": "skipped: no frames.jsonl / poses.jsonl / frame_size"}
    w, h = meta["frame_size"]
    acts = actions(
        rows, [f["t_arrival"] for f in frames], poses, w, h, float(meta.get("camera_lag_s", 0.0))
    )
    with (ep / "actions.jsonl").open("w") as f:
        f.writelines(json.dumps(a) + "\n" for a in acts)
    src = Counter(a["src"] for a in acts)
    return {k: round(src[k] / max(len(acts), 1), 3) for k in ("seen", "hindsight", "none")}


def episodes(raw: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for m in sorted(raw.glob("*/ep_*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("target", "face") == "face":
            out.append((m.parent, meta))
    return out


def detect(ep: Path, detector: Any, embedder: Any) -> Faces:
    """Every face in frames.mp4, with an ArcFace vector on every EMBED_EVERY-th frame."""
    import cv2

    video = cv2.VideoCapture(str(ep / "frames.mp4"))
    frame, box, score, emb = [], [], [], []
    i = width = before = 0
    while True:
        ok, img = video.read()
        if not ok:
            break
        width = img.shape[1]
        found = detector.detect(img)
        # also when the face count changes: a new track gets its identity checked at once
        due = i % EMBED_EVERY == 0 or len(found) != before
        vecs = embedder.embed(img, found) if due else None
        before = len(found)
        for k, f in enumerate(found):
            frame.append(i)
            box.append(f.box)
            score.append(f.score)
            emb.append(vecs[k] if vecs is not None else np.full(512, np.nan, np.float32))
        i += 1
    video.release()
    if not frame:
        return Faces.empty(i, width)
    return Faces(
        np.array(frame, np.int64),
        np.array(box, np.float32),
        np.array(score, np.float32),
        np.array(emb, np.float32),
        i,
        width,
    )


def save(path: Path, f: Faces) -> None:
    np.savez_compressed(
        path, frame=f.frame, box=f.box, score=f.score, emb=f.emb, size=[f.n_frames, f.width]
    )


def load(path: Path) -> Faces:
    z = np.load(path)
    n, w = (int(v) for v in z["size"])
    return Faces(z["frame"], z["box"], z["score"], z["emb"], n, w)


def preview(ep: Path, rows: list[dict[str, Any]], faces: Faces) -> None:
    """label_preview.mp4: every face grey, the target green (yellow where filled in)."""
    import cv2

    video = cv2.VideoCapture(str(ep / "frames.mp4"))
    out = None
    for r in rows:
        ok, img = video.read()
        if not ok:
            break
        for b in faces.box[faces.frame == r["i"]]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 1)
        if r["present"]:
            b = r["box_xyxy"]
            colour = (0, 200, 0) if r["src"] == "det" else (0, 220, 255)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colour, 3)
        else:
            cv2.putText(img, "not present", (20, 40), 0, 1.2, (0, 0, 255), 2)
        if out is None:
            h, w = img.shape[:2]
            out = cv2.VideoWriter(
                str(ep / "label_preview.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 30, (w, h)
            )
        out.write(img)
    video.release()
    if out is not None:
        out.release()


def main(args: argparse.Namespace) -> int:
    eps = episodes(args.raw)
    if not eps:
        print(f"no face episodes under {args.raw}")
        return 1
    todo = [ep for ep, _ in eps if args.redetect or not (ep / "faces.npz").is_file()]
    if todo:
        from reachy_demo.perception.faces import FaceDetector, FaceEmbedder

        detector, embedder = FaceDetector(min_score=args.min_score), FaceEmbedder()
        for n, ep in enumerate(todo, 1):
            t = time.monotonic()
            faces = detect(ep, detector, embedder)
            save(ep / "faces.npz", faces)
            print(f"[{n}/{len(todo)}] {ep.parent.name}/{ep.name}: {len(faces.frame)} faces "
                  f"in {faces.n_frames} frames ({time.monotonic() - t:.0f} s)")  # fmt: skip
    all_faces = {ep: load(ep / "faces.npz") for ep, _ in eps}

    # Pass 2: each person's vector, from episodes where only they are in view
    vectors: dict[str, list[Any]] = {}
    for ep, meta in eps:
        if meta.get("distractor") == "none" and not meta.get("negative"):
            v = gallery_vector(all_faces[ep])
            if v is not None:
                vectors.setdefault(meta["object_id"], []).append(v)
    galleries = {p: mean_vector(vs) for p, vs in vectors.items()}
    for person, vs in sorted(vectors.items()):
        print(f"gallery {person}: {len(vs)} episodes")

    flagged = []
    counts: Counter[str] = Counter()
    for ep, meta in eps:
        faces = all_faces[ep]
        rows, summary = label(faces, meta, galleries.get(meta["object_id"]))
        summary = {"label_version": LABEL_VERSION, **summary, **write_actions(ep, meta, rows)}
        with (ep / "boxes.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        (ep / "label.json").write_text(json.dumps(summary, indent=2) + "\n")
        counts.update(summary["flags"])
        name = f"{ep.parent.name}/{ep.name}"
        if summary["flags"]:
            flagged.append(f"{name}  {meta['object_id']}  {' '.join(summary['flags'])}")
        if args.preview == "all" or (args.preview == "flagged" and summary["flags"]):
            preview(ep, rows, faces)
    (args.raw / "flagged.txt").write_text("".join(line + "\n" for line in flagged))
    print(f"\nlabelled {len(eps)} episodes, {len(flagged)} flagged ({args.raw / 'flagged.txt'})")
    for flag, n in counts.most_common():
        print(f"  {flag}: {n}")
    return 0


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", type=Path, default=FACE_RAW_DIR, help=f"default {FACE_RAW_DIR}")
    p.add_argument("--redetect", action="store_true", help="run the detector again on all")
    p.add_argument("--min-score", type=float, default=0.5, help="SCRFD confidence (default 0.5)")
    p.add_argument(
        "--preview",
        choices=("flagged", "all", "none"),
        default="flagged",
        help="write label_preview.mp4 with boxes drawn (default: flagged episodes)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main(parse()))
