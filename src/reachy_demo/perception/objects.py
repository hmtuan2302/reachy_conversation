"""Find a named object once, then follow it fast (F4 "track my phone").

- `ObjectDetector`: YOLOX-nano (Apache-2.0, Megvii), ONNX on onnxruntime, 80 COCO classes,
  416x416 input, about 40-80 ms on the laptop CPU.
- `ObjectSource`: picks the object (asks the vision model when several of that class are in
  view), then tracks it by detection with the robot daemon's face-tracker algorithm (reachy_mini
  `vision/face_tracking.py`): the detector runs on every frame, `BoxTracker` keeps the box
  nearest the last one, and `CenterFilter` smooths its centre. Two changes for objects, which
  YOLOX misses far more often than YuNet misses faces: an OpenCV CSRT tracker bridges frames
  the detector misses (a hand over the phone), and a lost object is found again near where it
  was last seen, not as the largest box (that could be someone else's phone).

Face tracking uses the same `ObjectSource` with the SDK's YuNet wrapped in `BoxesDetector`
(`begin("face")`: no pick, the largest face wins, as on the robot).

`read()` returns the box centre like the face reader: x right, y down, each in [-1, 1], plus
the time the frame shows (grab time minus `lag_s`, the camera + motor delay). attention/follow.py
applies the offset to the head pose of that moment; without it a ~0.3 s Wi-Fi camera makes the
head swing around a still object, to the wrong side (checked in simulation 2026-11-18).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
# Tracker gates, as in reachy_mini vision/face_tracking.py `Tracker`.
MIN_AREA_FRAC = 0.003  # a new target must cover this share of the frame
MAX_JUMP = 0.5  # centre moves further than this (in [-1, 1] units) per frame: not ours
MAX_MISSES = 20  # frames without a match before the target is dropped and re-picked

Frame = npt.NDArray[np.uint8]
Box = tuple[float, float, float, float]  # x, y, w, h in pixels
GrabFrame = Callable[[], Frame | None]
Ask = Callable[[bytes, str], Awaitable[str]]  # (jpeg, question) -> answer


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: Box


@dataclass(frozen=True)
class Mark:
    """What object tracking saw last, for the debug view."""

    box: Box  # in full-frame pixels
    score: float  # the detector's; a CSRT box keeps the last detection's score
    label: str
    source: str  # "yolox" (detected this frame) or "csrt" (bridging a missed frame)
    aim: tuple[float, float]  # smoothed centre the head steers by, in [-1, 1]
    at: float  # clock time of the reading


MARK_STALE_S = 0.5  # an older mark counts as absent (stale perception)
CAMERA_LAG_S = 0.3  # [verify] head command -> seen in a grabbed frame; scripts/camera_lag.py


class Detects(Protocol):
    def detect(self, bgr: Frame, label: str | None = None) -> list[Detection]: ...


class BoxesDetector:
    """A one-class detector that gives boxes and scores (the SDK's YuNet) as a `Detects`."""

    def __init__(self, label: str, find: Callable[[Frame], list[tuple[Box, float]]]) -> None:
        self._label = label
        self._find = find

    def detect(self, bgr: Frame, label: str | None = None) -> list[Detection]:
        return [Detection(self._label, score, box) for box, score in self._find(bgr)]


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


def centre(box: Box, width: int, height: int) -> tuple[float, float]:
    """Box centre, x right and y down, each in [-1, 1] (pixel 0 to width - 1, as the SDK)."""
    x, y, w, h = box
    return (x + w / 2) / max(width - 1, 1) * 2 - 1, (y + h / 2) / max(height - 1, 1) * 2 - 1


class BoxTracker:
    """Track one box: associate nearest, drop after misses, re-pick nearest the last sighting.

    Port of reachy_mini `vision/face_tracking.py` `Tracker` (Apache-2.0, Pollen Robotics) for
    detector boxes. `seed`: the centre of the box picked at acquire. The SDK re-picks the largest
    face after a loss; here the box nearest the last sighting wins (largest only without one)."""

    def __init__(self, seed: tuple[float, float] | None = None) -> None:
        self._centre = seed
        self._last = seed  # last sighting, kept after the target is dropped
        self._misses = 0

    def select(self, found: list[Detection], width: int, height: int) -> Detection | None:
        """The detection to aim at, or None when no plausible target is in this frame."""
        if not found:
            self._miss()
            return None
        last = self._centre or self._last
        if last is None:
            pick = max(found, key=lambda d: d.box[2] * d.box[3])
        else:
            pick = min(found, key=lambda d: _dist2(centre(d.box, width, height), last))
            if self._centre is not None and _dist2(centre(pick.box, width, height), last) > (
                MAX_JUMP**2
            ):
                self._miss()
                return None
        if self._centre is None and pick.box[2] * pick.box[3] < MIN_AREA_FRAC * width * height:
            self._miss()  # a new target must be big enough to be real
            return None
        self._centre = self._last = centre(pick.box, width, height)
        self._misses = 0
        return pick

    def bridge(self, point: tuple[float, float]) -> bool:
        """Accept a box-tracker centre for a frame the detector missed. The miss still counts,
        so bridging lasts at most MAX_MISSES frames: CSRT alone drifts onto the background."""
        if self._centre is None or _dist2(point, self._centre) > MAX_JUMP**2:
            return False
        self._centre = self._last = point
        return True

    def _miss(self) -> None:
        self._misses += 1
        if self._misses > MAX_MISSES:
            self._centre = None

    @property
    def has_target(self) -> bool:
        return self._centre is not None


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class CenterFilter:
    """Smooth the target centre: slow when it jitters, faster when it moves, still in a dead zone.

    Port of reachy_mini `vision/face_tracking.py` `_AdaptiveCenterFilter` (same constants)."""

    ALPHA = 0.3
    FAST_ALPHA = 0.6
    MOVEMENT_THRESHOLD = 0.15
    DEAD_ZONE = 0.02

    def __init__(self) -> None:
        self._value: npt.NDArray[np.float64] | None = None
        self._previous: npt.NDArray[np.float64] | None = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        current = np.asarray(point, dtype=np.float64)
        if self._value is None or self._previous is None:
            self._value, self._previous = current.copy(), current.copy()
            return point
        movement = float(np.linalg.norm(current - self._previous))
        self._previous = current.copy()
        delta = current - self._value
        if float(np.linalg.norm(delta)) >= self.DEAD_ZONE:
            alpha = self.FAST_ALPHA if movement > self.MOVEMENT_THRESHOLD else self.ALPHA
            self._value += alpha * delta
        return float(self._value[0]), float(self._value[1])

    def reset(self) -> None:
        self._value = self._previous = None


class ObjectSource:
    """One object: picked once, then tracked. `read` runs in a worker thread, one at a time."""

    def __init__(
        self,
        grab: GrabFrame,
        make_detector: Callable[[], Detects],
        ask: Ask | None = None,
        clock: Callable[[], float] = time.monotonic,
        lag_s: float = CAMERA_LAG_S,
    ) -> None:
        """`lag_s`: how old a grabbed frame is, from head command to image (CAMERA_LAG_S)."""
        self._grab = grab
        self._clock = clock
        self._lag_s = lag_s
        self._mark: Mark | None = None
        self._score = 0.0
        self._make_detector = make_detector
        self._ask = ask
        self._detector: Detects | None = None
        self._tracker: BoxTracker | None = None
        self._csrt: Any = None  # runs only on frames the detector misses
        self._seen: tuple[Frame, Box] | None = None  # last frame the detector found it in
        self._filter = CenterFilter()
        self._label = ""
        self._box: Box | None = None

    async def acquire(self, target: str, label: str) -> str | None:
        """Start tracking `target` (of COCO class `label`); returns an error string if it can't.

        Not in view yet: it still starts, and `waiting` stays true until the object shows up.
        The first one seen then wins (the largest, as the face tracker picks faces)."""
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
        self._label, self._csrt, self._seen = label, None, None
        self._filter.reset()
        if not found:
            self._tracker, self._box = BoxTracker(), None  # no seed: the largest one that appears
            log.info("waiting for %s (%s) to come into view", target, label)
            return None
        pick = max(found, key=lambda d: d.score)
        if len(found) > 1 and self._ask is not None:
            pick = await choose(self._ask, frame, found, target) or pick
        h, w = frame.shape[:2]
        self._box = pick.box
        self._tracker = BoxTracker(centre(pick.box, w, h))
        self._seen = (frame, pick.box)
        self._score = pick.score
        log.info("tracking %s (%s, %d candidates)", target, label, len(found))
        return None

    def _half(self) -> Frame | None:
        """Half-size frame: cheaper to grab and resize, the detector input is 416 px anyway."""
        frame = self._grab()
        return None if frame is None else np.ascontiguousarray(frame[::2, ::2])

    def _start_csrt(self, frame: Frame, box: Box) -> None:
        import cv2

        self._csrt = cv2.TrackerCSRT.create()
        self._csrt.init(frame, tuple(int(v) for v in box))

    def _follow(self, frame: Frame) -> Box | None:
        """CSRT's box in `frame`, started from the last frame the detector found the object in."""
        if self._csrt is None:
            if self._seen is None:
                return None
            self._start_csrt(*self._seen)
            self._seen = None
        ok, box = self._csrt.update(frame)
        if not ok:
            self._csrt = None
            return None
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    def begin(self, label: str) -> bool:
        """Start tracking `label` with no pick: the largest one in view wins (face tracking).
        Blocking (the first call loads, maybe downloads, the model); False if it can't load."""
        if self._detector is None:
            try:
                self._detector = self._make_detector()
            except Exception as e:  # noqa: BLE001 - fail small: the caller follows the voice
                log.warning("%s detector unavailable: %s", label, e)
                return False
        self._label, self._csrt, self._seen, self._box = label, None, None, None
        self._tracker = BoxTracker()
        self._filter.reset()
        return True

    @property
    def waiting(self) -> bool:
        """Tracking started, but the object has not been seen yet (or was lost and dropped)."""
        return self._tracker is not None and not self._tracker.has_target

    def switch(self, on: bool) -> bool:
        """Follow's detector switch: on means acquire() already ran; off frees the models."""
        if not on:
            self.release()
        return self._tracker is not None

    def release(self) -> None:
        self._detector = None
        self._tracker = None
        self._csrt = self._seen = None
        self._box = None
        self._mark = None

    def mark(self) -> Mark | None:
        """The latest reading for the debug view; None when lost, stopped or stale."""
        mark = self._mark
        if mark is None or self._clock() - mark.at > MARK_STALE_S:
            return None
        return mark

    def read(self) -> tuple[float, float, float] | None:
        """The tracked object's smoothed centre (x right, y down, in [-1, 1]) and the clock time
        the frame shows, or None when it is not seen in this frame."""
        if self._tracker is None or self._detector is None:
            return None
        shown_at = self._clock() - self._lag_s
        frame = self._half()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        pick = self._tracker.select(self._detector.detect(frame, self._label), w, h)
        if pick is not None:
            box, source, self._score = pick.box, "yolox", pick.score
            self._csrt, self._seen = None, (frame, box)  # CSRT costs ~30 ms: only when needed
        elif (follow := self._follow(frame)) is not None and self._tracker.bridge(
            centre(follow, w, h)
        ):
            box, source = follow, "csrt"  # detector missed it this frame; CSRT still sees it
        else:
            if not self._tracker.has_target:
                self._filter.reset()
            self._mark = None
            return None
        self._box = box
        aim = self._filter.update(centre(box, w, h))
        full = (box[0] * 2, box[1] * 2, box[2] * 2, box[3] * 2)  # half-size frame -> full
        self._mark = Mark(full, self._score, self._label, source, aim, self._clock())
        return aim[0], aim[1], shown_at
