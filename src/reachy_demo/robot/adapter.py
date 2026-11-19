"""The only module that imports reachy_mini (docs/architecture/01-overview.md §4).

robot/dances.py imports the dances library, which imports the SDK in turn; it reads the move
catalog only and never talks to the robot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from reachy_demo.robot.motion import NEUTRAL, Pose, SetPose

log = logging.getLogger(__name__)

WAKE_S = 2.0  # rest pose -> neutral at startup
REST_S = 2.0  # neutral -> rest pose at exit
DAEMON_PORT = 8000  # the daemon's REST API (ReachyMini's default port)
DOA_TIMEOUT_S = 0.5
FRAME_WAIT_S = 1.0  # a caller waits this long for a fresh frame before getting None
FRAME_STALE_S = 0.5  # an older frame counts as absent (stale perception)
# The daemon's face tracker steers the head itself with this blend weight. 1 = it owns the head
# orientation while on (as the reference conversation app does: on-robot, no Wi-Fi lag); our
# targets still set the body yaw. 0 would pause its detector.
TRACKING_WEIGHT = 1.0


def doa_reader(host: str) -> Callable[[], tuple[float, bool] | None]:
    """Read the mic array's direction of arrival from the daemon (`GET /api/state/doa`).

    The daemon caches DoA at 10 Hz and answers null without a mic array. Network errors raise
    OSError; callers skip that reading.
    """
    url = f"http://{host}:{DAEMON_PORT}/api/state/doa"

    def read() -> tuple[float, bool] | None:
        with urllib.request.urlopen(url, timeout=DOA_TIMEOUT_S) as r:  # noqa: S310 (fixed http URL)
            doa = json.loads(r.read())
        if not isinstance(doa, dict):
            return None
        return float(doa["angle"]), bool(doa["speech_detected"])

    return read


def _daemon_json(host: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """GET (or POST `body`) one daemon REST path; network errors raise OSError."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(  # noqa: S310 (fixed http URL)
        f"http://{host}:{DAEMON_PORT}/api/{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=DOA_TIMEOUT_S) as r:  # noqa: S310
        return json.loads(r.read())


FaceReading = tuple[float, float] | tuple[float, float, float]


def face_reader(
    host: str, clock: Callable[[], float] = time.monotonic
) -> Callable[[], FaceReading | None]:
    """The face the daemon's detector (YuNet, on the robot) tracks now.

    Returns its centre in the camera frame, x right and y down, each in [-1, 1], and when the
    daemon stamps it, the time it was detected on `clock`; None when no face is seen or detection
    is off. The daemon's `ts` is the robot's monotonic clock: the smallest (now - ts) seen so far
    is taken as the clocks' offset.
    """
    offset = [math.inf]  # shortcut: the offset includes the fastest reply's latency (~20 ms)

    def read() -> FaceReading | None:
        face = _daemon_json(host, "media/tracking/face")["face_target"]
        now = clock()
        if not face.get("detected") or face.get("x") is None or face.get("y") is None:
            return None
        x, y = float(face["x"]), float(face["y"])
        if face.get("ts") is None:
            return x, y
        offset[0] = min(offset[0], now - float(face["ts"]))
        return x, y, float(face["ts"]) + offset[0]

    return read


def face_tracking_switch(host: str) -> Callable[[bool], bool]:
    """Start or stop the daemon's face detector; True when it runs (False: no camera)."""

    def switch(on: bool) -> bool:
        if on:
            reply = _daemon_json(host, "media/tracking/enable", {"weight": TRACKING_WEIGHT})
        else:
            reply = _daemon_json(host, "media/tracking/disable", {})
        return bool(reply.get("enabled"))

    return switch


def face_detector() -> Callable[[Any], list[tuple[float, float, float, float]]]:
    """The SDK's YuNet face detector (ONNX, CPU, one thread): BGR frame -> (x, y, w, h) boxes.

    The first call downloads the model (pinned revision) from Hugging Face.
    """
    from reachy_mini.vision.face_detector import FaceDetector  # lazy: the `robot` extra

    detector = FaceDetector()
    return lambda bgr: [f.bbox for f in detector.detect(bgr)]


def scored_face_detector(
    min_score: float = 0.6,
) -> Callable[[Any], list[tuple[tuple[float, float, float, float], float]]]:
    """Like `face_detector`, with YuNet's confidence (0-1) per box; NaN if it can't be matched.

    The SDK's `Face` has no score, so the score of each candidate box is recomputed from the
    same model outputs (as `FaceDetector._decode` does) and matched to the kept boxes. Boxes
    under `min_score` are dropped (the SDK's default is 0.6).
    """
    import numpy as np
    from reachy_mini.vision import face_detector as yunet  # lazy: the `robot` extra

    scores: dict[tuple[float, float, float, float], float] = {}

    # shortcut: mirrors the SDK's private decode maths; scores turn NaN if the SDK changes it
    class Scored(yunet.FaceDetector):  # type: ignore[misc]
        def _decode(self, outputs: dict[str, Any], width: int) -> Any:
            scores.clear()
            for stride in yunet._STRIDES:
                cls, obj = outputs[f"cls_{stride}"][0, :, 0], outputs[f"obj_{stride}"][0, :, 0]
                score = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
                idx = np.nonzero(score >= self._score_threshold)[0]
                bbox = outputs[f"bbox_{stride}"][0][idx]
                cols = width // stride
                cx = ((idx % cols).astype(np.float32) + bbox[:, 0]) * stride
                cy = ((idx // cols).astype(np.float32) + bbox[:, 1]) * stride
                w, h = np.exp(bbox[:, 2]) * stride, np.exp(bbox[:, 3]) * stride
                for k in range(idx.size):
                    box = (
                        float(cx[k] - w[k] / 2),
                        float(cy[k] - h[k] / 2),
                        float(w[k]),
                        float(h[k]),
                    )
                    scores[box] = float(score[idx[k]])
            return super()._decode(outputs, width)

    detector = Scored(score_threshold=min_score)
    return lambda bgr: [(f.bbox, scores.get(f.bbox, math.nan)) for f in detector.detect(bgr)]


@dataclass(frozen=True)
class Robot:
    set_pose: SetPose  # send one target, return at once; only robot/motion.py calls it
    grab_jpeg: Callable[[], bytes | None]  # current camera frame; None without a camera
    grab_frame: Callable[[], Any]  # current BGR frame (numpy HxWx3 uint8); None without a camera
    has_camera: bool
    # (angle rad: 0 = left, pi/2 = front, pi = right; speech) or None. Blocks up to DOA_TIMEOUT_S.
    read_doa: Callable[[], tuple[float, bool] | None]
    # Face centre (x right, y down, in [-1, 1]), + its detection time (monotonic) when known,
    # or None. Blocks up to DOA_TIMEOUT_S.
    read_face: Callable[[], FaceReading | None]
    set_face_tracking: Callable[[bool], bool]  # daemon face detector on/off


class LatestFrame:
    """The camera's one reader: a thread keeps the newest frame, and every caller gets it.

    The SDK's video sink holds one frame, and a pull takes it away. With several readers (the
    search, the debug view) each got an uneven share of the frames, and the view stuttered.
    Callers share the array: draw on a copy.
    """

    def __init__(self, read: Callable[[], Any], clock: Callable[[], float] = time.monotonic):
        self._read = read  # blocks up to ~20 ms (SDK), None when no frame came
        self._clock = clock
        self._frame: tuple[float, Any] | None = None  # (clock time, frame)
        self._new = threading.Condition()
        self._stop = threading.Event()

    def run(self) -> None:
        """Pull frames until `stop()`; run it in its own thread."""
        while not self._stop.is_set():
            try:
                frame = self._read()
            except Exception as e:  # noqa: BLE001 - fail small: a bad pull skips one frame
                log.warning("camera read failed: %s", e)
                self._stop.wait(0.1)
                continue
            if frame is not None:
                with self._new:
                    self._frame = (self._clock(), frame)
                    self._new.notify_all()

    def stop(self) -> None:
        self._stop.set()

    def get(self, wait_s: float = FRAME_WAIT_S, stale_s: float = FRAME_STALE_S) -> Any:
        """The newest frame; waits up to `wait_s` for one under `stale_s` old, else None."""

        def fresh() -> bool:
            return self._frame is not None and self._clock() - self._frame[0] <= stale_s

        with self._new:
            if not self._new.wait_for(fresh, timeout=wait_s):
                return None
            assert self._frame is not None
            return self._frame[1]


def _open(host: str, camera: bool) -> Any:
    from reachy_mini import ReachyMini  # lazy: the SDK is the optional `robot` extra

    if camera:
        # Camera streams over WebRTC (spike 2026-11-16: works from WSL2, first frame ~1.5 s).
        # It also opens an audio channel we don't use; the "No Reachy Mini Audio USB device"
        # error it logs on a laptop is harmless.
        try:
            return ReachyMini(host=host, connection_mode="network", media_backend="webrtc")
        except Exception as e:  # fail small: no camera, but the head still works
            log.warning("camera unavailable (%s); connecting without it", e)
    return ReachyMini(host=host, connection_mode="network", media_backend="no_media")


NO_NVDEC = ",".join(
    f"nv{c}dec:NONE"
    for c in ("h264", "h265", "vp8", "vp9", "av1", "jpeg", "mpeg2video", "mpeg4video")
)


@contextmanager
def connect_robot(host: str, camera: bool = True) -> Iterator[Robot]:
    """Connect over Wi-Fi by IP; yields the pose sender and the camera.

    Poses: head position (mm), orientation and antennas (degrees). On entry the motors are
    enabled and the robot eases to neutral, so the motion loop starts from a known pose. On exit
    the head is lowered into the SDK's rest (sleep) pose, and only then are the motors disabled:
    the head already rests on its stops, so it doesn't drop.
    """
    # NVDEC decoding makes CUDA calls from GStreamer's thread, which break PyTorch's CUDA graph
    # capture (VieNeu on GPU: "Offset increment outside graph capture"). Decode video on the CPU.
    # Read once, by the SDK's first Gst.init during connect.
    os.environ.setdefault("GST_PLUGIN_FEATURE_RANK", NO_NVDEC)
    from reachy_mini.reachy_mini import SLEEP_ANTENNAS_JOINT_POSITIONS, SLEEP_HEAD_POSE
    from reachy_mini.utils import create_head_pose

    def head(p: Pose) -> object:
        return create_head_pose(
            x=p["x"],
            y=p["y"],
            z=p["z"],
            roll=p["roll"],
            pitch=p["pitch"],
            yaw=p["yaw"],
            mm=True,
            degrees=True,
        )

    def antennas(p: Pose) -> list[float]:
        return [math.radians(p["antenna_0"]), math.radians(p["antenna_1"])]

    with contextlib.ExitStack() as stack:
        mini = stack.enter_context(_open(host, camera))
        has_camera = mini.media.camera is not None

        def set_pose(p: Pose) -> None:
            mini.set_target(
                head=head(p), antennas=antennas(p), body_yaw=math.radians(p["body_yaw"])
            )

        frames = LatestFrame(mini.media.get_frame)
        if has_camera:
            reader = threading.Thread(target=frames.run, name="camera", daemon=True)
            reader.start()
            stack.callback(reader.join, 1.0)
            stack.callback(frames.stop)

        def grab_frame() -> Any:
            return frames.get() if has_camera else None

        def grab_jpeg() -> bytes | None:
            if not has_camera:
                return None
            try:
                import cv2  # the `perception` extra
            except ImportError:  # the SDK's encoder pulls its own frame from the shared sink
                jpeg: bytes | None = mini.media.get_frame_jpeg()
                return jpeg
            frame = frames.get()
            if frame is None:
                return None
            ok, jpg = cv2.imencode(".jpg", frame)
            return bytes(jpg.tobytes()) if ok else None

        try:
            # Torque-off from a previous exit stays on the robot; targets are ignored until
            # the motors are enabled. Enable first: enable_motors() pins targets to the
            # present pose (SDK set_target note), so moves must come after it.
            mini.enable_motors()
            mini.goto_target(head=head(NEUTRAL), antennas=antennas(NEUTRAL), duration=WAKE_S)
            yield Robot(
                set_pose,
                grab_jpeg,
                grab_frame,
                has_camera,
                doa_reader(host),
                face_reader(host),
                face_tracking_switch(host),
            )
        finally:
            with contextlib.suppress(OSError):  # a detector left on keeps using robot CPU
                face_tracking_switch(host)(False)
            try:
                # Like ReachyMini.goto_sleep(), without its sound (media sound isn't used).
                mini.goto_target(
                    head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=REST_S
                )
            finally:
                mini.disable_motors()
