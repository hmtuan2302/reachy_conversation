"""Offline face models for labelling (scripts/track_label.py): SCRFD finds faces, ArcFace whose.

Both come from InsightFace's `buffalo_l` pack (SCRFD-10GF `det_10g.onnx`, ArcFace R50
`w600k_r50.onnx`) and run on onnxruntime, so no new package is needed. SCRFD is far stronger than
YuNet on small, dark and side-on faces; it is too slow for live tracking on the laptop's CPU
(~50 ms a frame) but fine offline. The models are for non-commercial research use (InsightFace
licence).
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

MODEL_DIR = Path("models/insightface")
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DETECTOR = "det_10g.onnx"
EMBEDDER = "w600k_r50.onnx"
SIZE = 640  # SCRFD's output shapes are fixed to a 640x640 input
STRIDES = (8, 16, 32)
ANCHORS = 2  # per grid cell
NMS_IOU = 0.4
# ArcFace's 5-point template (eyes, nose, mouth corners) in its 112x112 input
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)  # fmt: skip

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    kps: npt.NDArray[np.float32]  # (5, 2): eyes, nose, mouth corners


def fetch_models(model_dir: Path = MODEL_DIR) -> None:
    """Download buffalo_l once (289 MB) and keep only the two models used."""
    if all((model_dir / m).is_file() for m in (DETECTOR, EMBEDDER)):
        return
    import urllib.request

    model_dir.mkdir(parents=True, exist_ok=True)
    part = model_dir / "buffalo_l.zip.part"
    log.info("downloading face models from %s", PACK_URL)
    try:
        urllib.request.urlretrieve(PACK_URL, part)  # noqa: S310 - a fixed https URL
        with zipfile.ZipFile(part) as z:
            for m in (DETECTOR, EMBEDDER):
                name = next(n for n in z.namelist() if n.endswith(m))
                (model_dir / m).write_bytes(z.read(name))
    except (OSError, zipfile.BadZipFile, StopIteration) as e:
        raise FileNotFoundError(f"face model download failed: {e}") from e
    finally:
        part.unlink(missing_ok=True)


def _session(path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # ArcFace's file declares batch 1; a batch warns every run
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def decode(outputs: list[npt.NDArray[np.float32]], scale: float, min_score: float) -> list[Face]:
    """SCRFD outputs (scores x3, boxes x3, keypoints x3, one per stride) -> faces after NMS."""
    import cv2

    boxes, scores, kps = [], [], []
    for k, stride in enumerate(STRIDES):
        n = SIZE // stride
        ys, xs = np.mgrid[:n, :n]
        centres = np.repeat(np.stack([xs, ys], -1).reshape(-1, 2) * stride, ANCHORS, 0)
        score = outputs[k][:, 0]
        keep = score >= min_score
        c = centres[keep].astype(np.float32)
        d = outputs[k + 3][keep] * stride  # distances to left, top, right, bottom
        p = outputs[k + 6][keep].reshape(-1, 5, 2) * stride
        boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], 1))
        kps.append(p + c[:, None, :])
        scores.append(score[keep])
    box = np.concatenate(boxes) / scale
    score = np.concatenate(scores)
    points = np.concatenate(kps) / scale
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in box]
    picked = cv2.dnn.NMSBoxes(xywh, score.tolist(), min_score, NMS_IOU) if len(xywh) else []
    return [
        Face(
            (float(box[i][0]), float(box[i][1]), float(box[i][2]), float(box[i][3])),
            float(score[i]),
            points[i].astype(np.float32),
        )
        for i in np.asarray(picked, dtype=int).reshape(-1)
    ]


class FaceDetector:
    """SCRFD-10GF: BGR frame -> faces with 5 keypoints."""

    def __init__(self, model_dir: Path = MODEL_DIR, min_score: float = 0.5) -> None:
        fetch_models(model_dir)
        self._session = _session(model_dir / DETECTOR)
        self._input = self._session.get_inputs()[0].name
        self._min_score = min_score

    def detect(self, bgr: Image) -> list[Face]:
        import cv2

        h, w = bgr.shape[:2]
        scale = min(SIZE / h, SIZE / w)
        padded = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        padded[: int(h * scale), : int(w * scale)] = cv2.resize(
            bgr, (int(w * scale), int(h * scale))
        )
        blob = cv2.dnn.blobFromImage(padded, 1 / 128, (SIZE, SIZE), (127.5,) * 3, swapRB=True)
        return decode(self._session.run(None, {self._input: blob}), scale, self._min_score)


class FaceEmbedder:
    """ArcFace R50: aligned face -> unit 512-d vector; same person ~0.4-0.8 cosine, others < 0.2."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        fetch_models(model_dir)
        self._session = _session(model_dir / EMBEDDER)
        self._input = self._session.get_inputs()[0].name

    def embed(self, bgr: Image, faces: list[Face]) -> npt.NDArray[np.float32]:
        import cv2

        if not faces:
            return np.zeros((0, 512), np.float32)
        crops = []
        for f in faces:
            m, _ = cv2.estimateAffinePartial2D(f.kps, TEMPLATE, method=cv2.LMEDS)
            crops.append(cv2.warpAffine(bgr, m, (112, 112), borderValue=0.0))
        blob = cv2.dnn.blobFromImages(crops, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        v = self._session.run(None, {self._input: blob})[0]
        out: npt.NDArray[np.float32] = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(
            np.float32
        )
        return out
