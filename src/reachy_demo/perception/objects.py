"""Find a named object once, then follow it fast (F4 "track my phone").

- `ObjectDetector`: YOLOX-nano (Apache-2.0, Megvii), ONNX on onnxruntime, 80 COCO classes,
  416x416 input, about 40-80 ms on the laptop CPU.
- `ObjectSource`: picks the object (asks the vision model when several of that class are in
  view), then an OpenCV CSRT tracker follows its box every frame (~15 ms). The detector checks
  again every RECHECK_S and corrects the tracker's drift (phones are smooth and shiny).

`read()` returns the box centre like the face reader: x right, y down, each in [-1, 1], so
attention/follow.py steers toward it exactly as it does toward a face.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

DEFAULT_MODEL = Path("models/yolox/yolox_nano.onnx")
MODEL_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
)
COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip
SIZE = 416  # yolox_nano / yolox_tiny; read from the model, so s/m/l (640) work too
STRIDES = (8, 16, 32)
MIN_SCORE = 0.3  # shortcut: one threshold for every class; tune per object on the robot
RECHECK_S = 1.0
MATCH_IOU = 0.1  # a re-detection this close to the tracked box is the same object
NEAR = 0.25  # ...or its centre within this share of the frame width (after a fast move)

Frame = npt.NDArray[np.uint8]
Box = tuple[float, float, float, float]  # x, y, w, h in pixels
GrabFrame = Callable[[], Frame | None]
Ask = Callable[[bytes, str], Awaitable[str]]  # (jpeg, question) -> answer


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: Box


def iou(a: Box, b: Box) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    h = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = w * h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _grids(size: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    cells, strides = [], []
    for s in STRIDES:
        n = size // s
        ys, xs = np.mgrid[0:n, 0:n]
        cells.append(np.stack([xs.ravel(), ys.ravel()], 1))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(cells).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode(
    raw: npt.NDArray[np.float32], scale: float, label: str | None, size: int = SIZE
) -> list[Detection]:
    """YOLOX head output (N, 85) on the letterboxed input -> detections in frame pixels."""
    import cv2

    grid, stride = _grids(size)
    xy = (raw[:, :2] + grid) * stride
    wh = np.exp(raw[:, 2:4]) * stride
    classes = raw[:, 5:] * raw[:, 4:5]
    if label is not None:  # only the class asked for
        k = COCO.index(label)
        cls, score = np.full(len(raw), k), classes[:, k]
    else:
        cls, score = classes.argmax(1), classes.max(1)
    keep = score >= MIN_SCORE
    boxes = np.concatenate([xy - wh / 2, wh], 1)[keep] / scale
    score, cls = score[keep], cls[keep]
    picked = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), MIN_SCORE, 0.45)
    return [
        Detection(COCO[int(cls[i])], float(score[i]), tuple(map(float, boxes[i])))  # type: ignore[arg-type]
        for i in np.asarray(picked).ravel()
    ]


class ObjectDetector:
    """YOLOX-nano on one CPU thread (leaves the rest to speech and the motion loop)."""

    def __init__(self, model: Path = DEFAULT_MODEL) -> None:
        if not model.is_file():
            raise FileNotFoundError(
                f"object model missing: {model}. Download it: curl -L -o {model} {MODEL_URL}"
            )
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self._size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else SIZE
        n = sum((self._size // s) ** 2 for s in STRIDES)
        out = self._session.get_outputs()[0].shape
        if list(out[1:]) != [n, 5 + len(COCO)]:
            raise RuntimeError(
                f"{model.name} is not a YOLOX COCO model (output {out}, expected [1, {n}, 85])"
            )

    def detect(self, bgr: Frame, label: str | None = None) -> list[Detection]:
        import cv2

        h, w = bgr.shape[:2]
        size = self._size
        scale = min(size / h, size / w)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[: int(h * scale), : int(w * scale)] = cv2.resize(
            bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        x = padded.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0-255, as YOLOX trains
        raw = self._session.run(None, {self._input: x})[0][0]
        return decode(raw, scale, label, size)


def detect_tiled(detector: ObjectDetector, bgr: Frame, label: str) -> list[Detection]:
    """Detect on the whole frame and on 3x3 overlapping tiles (each 40% of the frame), so small
    far objects reach the detector big enough. 10 detector runs, ~130 ms on the laptop.

    Checked 2026-11-17 (YOLOX-nano, 1280x720): the whole frame alone missed objects under
    ~80x140 px; 2x2 tiles found them from ~67x117 px, 3x3 from ~48x84 px.
    """
    import cv2

    h, w = bgr.shape[:2]
    th, tw = int(h * 0.4), int(w * 0.4)
    found = list(detector.detect(bgr, label))
    for y0 in np.linspace(0, h - th, 3).astype(int):
        for x0 in np.linspace(0, w - tw, 3).astype(int):
            tile = np.ascontiguousarray(bgr[y0 : y0 + th, x0 : x0 + tw])
            for d in detector.detect(tile, label):
                x, y, bw, bh = d.box
                found.append(Detection(d.label, d.score, (x + x0, y + y0, bw, bh)))
    if not found:
        return []
    keep = cv2.dnn.NMSBoxes([list(d.box) for d in found], [d.score for d in found], 0.0, 0.45)
    return [found[i] for i in np.asarray(keep).ravel()]


async def choose(ask: Ask, frame: Frame, found: list[Detection], target: str) -> Detection | None:
    """Ask the vision model which numbered box is `target`; None if none is, or it can't tell.

    With one box this is a yes/no check, which weeds out look-alikes (a remote for a phone).
    """
    import cv2

    marked = frame.copy()
    for i, d in enumerate(found, 1):
        x, y, w, h = map(int, d.box)
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(marked, str(i), (x + 4, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    ok, jpeg = cv2.imencode(".jpg", marked)
    if not ok:
        return None
    question = (
        f"Green numbered boxes mark objects. Which box contains {target}? "
        "Answer with the number only, or 0 if none."
    )
    try:
        answer = await ask(jpeg.tobytes(), question)
    except Exception as e:  # noqa: BLE001 - fail small: the caller falls back
        log.warning("vision pick failed: %s", e)
        return None
    m = re.search(r"\d+", answer)
    n = int(m.group()) if m else 0
    return found[n - 1] if 1 <= n <= len(found) else None


async def describe_box(ask: Ask, frame: Frame, d: Detection) -> str | None:
    """One short sentence about the object in `d`'s box, or None if the vision model fails."""
    import cv2

    marked = frame.copy()
    x, y, w, h = map(int, d.box)
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 3)
    ok, jpeg = cv2.imencode(".jpg", marked)
    if not ok:
        return None
    question = (
        "Describe the object in the green box in one short sentence: what it is, its colour, "
        "and what it is on or next to. Do not mention the box."
    )
    try:
        answer = (await ask(jpeg.tobytes(), question)).strip()
    except Exception as e:  # noqa: BLE001 - fail small: report without a description
        log.warning("vision describe failed: %s", e)
        return None
    return answer or None


class ObjectSource:
    """One object: picked once, then tracked. `read` runs in a worker thread, one at a time."""

    def __init__(
        self,
        grab: GrabFrame,
        make_detector: Callable[[], ObjectDetector],
        ask: Ask | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._grab = grab
        self._make_detector = make_detector
        self._ask = ask
        self._clock = clock
        self._detector: ObjectDetector | None = None
        self._tracker: Any = None
        self._label = ""
        self._box: Box | None = None
        self._checked = -np.inf
        self._width = 1

    async def acquire(self, target: str, label: str) -> str | None:
        """Find `target` (of COCO class `label`) in view; returns an error string if it can't."""
        import asyncio

        if label not in COCO:
            return f"unsupported_object: can only track these: {', '.join(COCO)}"
        if self._detector is None:
            try:
                self._detector = await asyncio.to_thread(self._make_detector)
            except (OSError, ImportError, RuntimeError) as e:
                log.warning("object detector unavailable: %s", e)
                return f"tracking_unavailable: {e}"
        frame = await asyncio.to_thread(self._half)
        if frame is None:
            return "no camera frame (camera off or not connected)"
        found = await asyncio.to_thread(self._detector.detect, frame, label)
        if not found:
            self.release()
            return f"not_visible: no {label} in view"
        pick = max(found, key=lambda d: d.score)
        if len(found) > 1 and self._ask is not None:
            pick = await choose(self._ask, frame, found, target) or pick
        self._label, self._width = label, frame.shape[1]
        await asyncio.to_thread(self._start_tracker, frame, pick.box)
        log.info("tracking %s (%s, %d candidates)", target, label, len(found))
        return None

    def _half(self) -> Frame | None:
        """Half-size frame: the tracker runs 2x faster, the detector input is 416 px anyway."""
        frame = self._grab()
        return None if frame is None else np.ascontiguousarray(frame[::2, ::2])

    def _start_tracker(self, frame: Frame, box: Box) -> None:
        import cv2

        self._tracker = cv2.TrackerCSRT.create()
        self._tracker.init(frame, tuple(int(v) for v in box))
        self._box = box
        self._checked = self._clock()

    def switch(self, on: bool) -> bool:
        """Follow's detector switch: on means acquire() already ran; off frees the models."""
        if not on:
            self.release()
        return self._tracker is not None

    def release(self) -> None:
        self._detector = None
        self._tracker = None
        self._box = None

    def read(self) -> tuple[float, float] | None:
        """The tracked object's centre (x right, y down, in [-1, 1]), or None when lost."""
        if self._tracker is None or self._detector is None:
            return None
        frame = self._half()
        if frame is None:
            return None
        ok, box = self._tracker.update(frame)
        box = tuple(map(float, box)) if ok else None
        now = self._clock()
        if box is None or now - self._checked >= RECHECK_S:
            self._checked = now
            fixed = self._match(self._detector.detect(frame, self._label), box or self._box)
            if fixed is not None:
                self._start_tracker(frame, fixed)
                box = fixed
        if box is None:
            return None
        self._box = box
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        return (x + bw / 2) / (w / 2) - 1.0, (y + bh / 2) / (h / 2) - 1.0

    def _match(self, found: list[Detection], near: Box | None) -> Box | None:
        """The detection that is the tracked object: most overlap, else nearest centre."""
        if not found or near is None:
            return None
        best = max(found, key=lambda d: iou(d.box, near))
        if iou(best.box, near) >= MATCH_IOU:
            return best.box

        def dist(d: Detection) -> float:
            return float(
                np.hypot(
                    d.box[0] + d.box[2] / 2 - near[0] - near[2] / 2,
                    d.box[1] + d.box[3] / 2 - near[1] - near[3] / 2,
                )
            )

        closest = min(found, key=dist)
        return closest.box if dist(closest) <= NEAR * self._width else None
