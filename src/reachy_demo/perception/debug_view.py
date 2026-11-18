"""Live camera page with face boxes, for debugging face tracking (`agent --robot --debug-view`).

Serves an MJPEG stream at http://127.0.0.1:PORT/ with the stdlib server (opencv-headless has no
windows). Each frame shows:

- green boxes: YuNet run here on the laptop on this very frame, so they never lag the image,
  each with its confidence (0-1);
- red cross: the daemon's tracked face (`/api/media/tracking/face`), what attention/follow.py
  steers by. It shows only while tracking runs ("track me"), since only then is the detector on;
- grey cross: the image centre, where tracking tries to put the face;
- object tracking ("track my phone"): its box with class and detector score, blue when YOLOX
  found it on this read, yellow when CSRT bridges a frame YOLOX missed, and a cross of the same
  colour at the smoothed aim point the head steers by.

The annotated frames are also recorded to `record_dir` (data/recordings/view-<time>.mp4, plus
the chat as view-<time>.txt) and saved on exit. Frames are repeated to a fixed RECORD_FPS so the
video plays at real speed, lag included.

Beside it, the conversation ("you:" / "bot:" lines only), polled from /chat once a second.

A red cross that trails the green box while you move is daemon/Wi-Fi lag. The head still
lagging after the red cross is on you is follow.py's gain and speed limit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reachy_demo.perception.objects import Mark

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]  # x, y (top left), w, h in pixels
Scored = tuple[Box, float]  # a box and its confidence, 0-1 (NaN: unknown)
RECORD_FPS = 15
LABEL_SCALE = 1.2  # confidence text size (cv2 font scale)
OBJECT_COLOURS = {"yolox": (255, 128, 0), "csrt": (0, 255, 255)}  # BGR: blue, yellow
CHAT_LINES = 200  # shortcut: older lines drop off the page; the terminal keeps them all
PAGE = b"""<!doctype html><meta charset=utf-8><title>Reachy debug view</title>
<style>
body{margin:0;background:#111;color:#ccc;font:14px sans-serif;display:flex;flex-wrap:wrap}
#cam{flex:2 1 480px;padding:8px}#cam img{width:100%}
#chat{flex:1 1 280px;height:95vh;overflow-y:auto;padding:8px;white-space:pre-wrap}
.you{color:#8cf}.bot{color:#eee}
</style>
<div id=cam><img src=/stream><p>green: laptop YuNet (this frame) &middot;
red: daemon face target (only while tracking) &middot; grey: centre &middot;
blue/yellow: tracked object, YOLOX/CSRT (only while tracking an object)</p></div>
<div id=chat></div>
<script>
const chat = document.getElementById("chat");
let shown = -1;
async function poll() {
  try {
    const r = await (await fetch("/chat")).json();
    if (r.n !== shown) {
      shown = r.n;
      chat.replaceChildren(...r.lines.map(([who, text]) => {
        const p = document.createElement("div");
        p.className = who;
        p.textContent = who + ": " + text;
        return p;
      }));
      chat.scrollTop = chat.scrollHeight;
    }
  } catch (e) {}
  setTimeout(poll, 1000);
}
poll();
</script>
"""


def annotate(
    frame: Any,
    boxes: list[Scored],
    face: tuple[float, ...] | None,
    note: str,
    obj: Mark | None = None,
) -> Any:
    """Draw the overlay on a copy of a BGR frame."""
    import cv2  # lazy: the `perception` extra

    img = frame.copy()
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (160, 160, 160), cv2.MARKER_CROSS, 30, 1)
    for (x, y, bw, bh), score in boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
        label = (int(x), max(int(y) - 10, 35))
        cv2.putText(
            img, f"{score:.2f}", label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (0, 255, 0), 3
        )
    if face is not None:  # daemon's normalised [-1, 1] centre, as in robot/adapter.py
        u, v = int((face[0] + 1) / 2 * (w - 1)), int((face[1] + 1) / 2 * (h - 1))
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    if obj is not None:
        colour = OBJECT_COLOURS.get(obj.source, (255, 255, 255))
        x, y, bw, bh = (int(v) for v in obj.box)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, 3)
        text = f"{obj.label} {obj.score:.2f}" + (" csrt" if obj.source == "csrt" else "")
        cv2.putText(
            img, text, (x, max(y - 10, 35)), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, colour, 3
        )
        u, v = int((obj.aim[0] + 1) / 2 * (w - 1)), int((obj.aim[1] + 1) / 2 * (h - 1))
        cv2.drawMarker(img, (u, v), colour, cv2.MARKER_CROSS, 40, 3)
    cv2.putText(img, note, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def to_jpeg(img: Any) -> bytes:
    import cv2

    ok, jpg = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("JPEG encoding failed")
    return bytes(jpg.tobytes())


class Recorder:
    """Writes frames to an MP4 at RECORD_FPS, repeating frames so playback keeps real time."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.path = path
        self._clock = clock
        self._writer: Any = None
        self._start = 0.0
        self._written = 0

    def write(self, img: Any) -> None:
        import cv2

        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, RECORD_FPS, (w, h))
            if not self._writer.isOpened():
                raise OSError(f"cannot write video {self.path}")
            self._start = self._clock()
        due = int((self._clock() - self._start) * RECORD_FPS) + 1  # frames owed by now
        for _ in range(max(due - self._written, 0)):
            self._writer.write(img)
        self._written = max(due, self._written)

    def close(self) -> bool:
        """Finish the file; True when anything was recorded."""
        if self._writer is None:
            return False
        self._writer.release()
        return True


class DebugView:
    def __init__(
        self,
        grab_frame: Callable[[], Any],
        make_detector: Callable[[], Callable[[Any], list[Scored]]],
        read_face: Callable[[], tuple[float, ...] | None],
        record_dir: Path | None = None,
        read_object: Callable[[], Mark | None] = lambda: None,
    ) -> None:
        """`read_object`: object tracking's latest reading (None while it doesn't run)."""
        self._read_object = read_object
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.recorder = Recorder(record_dir / f"view-{stamp}.mp4") if record_dir else None
        self._grab = grab_frame
        self._make_detector = make_detector
        self._read_face = read_face
        self._jpeg: bytes | None = None
        self._new = threading.Condition()
        self._stop = threading.Event()
        self._lines: deque[tuple[str, str]] = deque(maxlen=CHAT_LINES)
        self._n = 0  # lines added so far: the page redraws when it changes

    def add_line(self, who: str, text: str) -> None:
        """One conversation line for the chat panel; `who` is "you" or "bot"."""
        self._lines.append((who, text))
        self._n += 1
        if self.recorder:  # the whole chat, kept beside the video
            with self.recorder.path.with_suffix(".txt").open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {who}: {text}\n")

    def _produce(self) -> None:
        try:
            self._loop()
        finally:  # in this thread: the writer is only touched here
            if self.recorder and self.recorder.close():
                log.info("debug video saved: %s", self.recorder.path)

    def _loop(self) -> None:
        detect = self._make_detector()  # in this thread: the first call may download the model
        last = time.monotonic()
        fps = 0.0
        while not self._stop.is_set():
            frame = self._grab()
            if frame is None:
                time.sleep(0.05)
                continue
            boxes = detect(frame)
            t0 = time.monotonic()
            try:
                face = self._read_face()
            except OSError:  # one failed read over Wi-Fi: draw no red cross this frame
                face = None
            read_ms = (time.monotonic() - t0) * 1000
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            note = f"{fps:4.1f} fps  daemon read {read_ms:3.0f} ms  faces {len(boxes)}"
            img = annotate(frame, boxes, face, note, self._read_object())
            if self.recorder:
                try:
                    self.recorder.write(img)
                except OSError as e:  # fail small: keep the live view, drop the recording
                    log.warning("debug video off: %s", e)
                    self.recorder = None
            jpeg = to_jpeg(img)
            with self._new:
                self._jpeg = jpeg
                self._new.notify_all()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (stdlib name)
                if self.path == "/chat":
                    body = json.dumps({"n": view._n, "lines": list(view._lines)}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                with contextlib.suppress(OSError):  # browser tab closed
                    while not view._stop.is_set():
                        with view._new:
                            view._new.wait(1.0)
                            jpeg = view._jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg + b"\r\n")

            def log_message(self, *args: Any) -> None:
                pass  # keep the agent's terminal clean

        return Handler

    @contextlib.contextmanager
    def serve(self, port: int) -> Iterator[str]:
        """Run the view on 127.0.0.1:`port` (0 = any free port); yields its URL."""
        server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        server.daemon_threads = True
        threads = [
            threading.Thread(target=self._produce, name="debug-view", daemon=True),
            threading.Thread(target=server.serve_forever, name="debug-http", daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/"
        finally:
            self._stop.set()
            threads[0].join(timeout=5)  # let it finish the video file before the app exits
            server.shutdown()
            server.server_close()
