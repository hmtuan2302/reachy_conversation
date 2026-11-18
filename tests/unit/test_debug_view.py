import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from reachy_demo.perception.debug_view import (  # noqa: E402
    OBJECT_COLOURS,
    RECORD_FPS,
    DebugView,
    Recorder,
    annotate,
    to_jpeg,
)
from reachy_demo.perception.objects import Mark  # noqa: E402


def test_annotate_draws_box_and_daemon_cross() -> None:
    frame = np.zeros((120, 160, 3), np.uint8)
    img = annotate(frame, [((10, 10, 40, 40), 0.87)], (0.5, -0.5), "note")
    assert img.any() and to_jpeg(img)[:2] == b"\xff\xd8"
    assert not frame.any()  # the caller's frame is left untouched


def test_annotate_draws_the_tracked_object_in_its_source_colour() -> None:
    frame = np.zeros((120, 160, 3), np.uint8)
    for source in ("yolox", "csrt"):
        mark = Mark((20, 40, 40, 40), 0.71, "cup", source, (-0.5, 0.0), 0.0)
        img = annotate(frame, [], None, "", mark)
        assert tuple(img[60, 20]) == OBJECT_COLOURS[source]  # left edge of the box


def test_stream_serves_annotated_frames() -> None:
    frame = np.zeros((120, 160, 3), np.uint8)
    view = DebugView(lambda: frame, lambda: lambda _f: [((1, 1, 5, 5), 0.9)], lambda: None)
    with view.serve(0) as url:
        assert b"/stream" in urllib.request.urlopen(url, timeout=5).read()
        with urllib.request.urlopen(url + "stream", timeout=5) as r:
            head = r.read(200)
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8")


def test_chat_lists_conversation_lines() -> None:
    view = DebugView(lambda: None, lambda: lambda _f: [], lambda: None)
    view.add_line("you", "Hello.")
    view.add_line("bot", "Hello! What can I do for you?")
    with view.serve(0) as url:
        chat = json.loads(urllib.request.urlopen(url + "chat", timeout=5).read())
    assert chat == {
        "n": 2,
        "lines": [["you", "Hello."], ["bot", "Hello! What can I do for you?"]],
    }


def test_recorder_repeats_frames_to_keep_real_time(tmp_path: Path) -> None:
    import cv2

    now = [0.0]
    rec = Recorder(tmp_path / "v.mp4", clock=lambda: now[0])
    frame = np.zeros((120, 160, 3), np.uint8)
    rec.write(frame)
    now[0] = 1.0  # one slow frame a second later: the gap is filled at RECORD_FPS
    rec.write(frame)
    assert rec.close()
    video = cv2.VideoCapture(str(tmp_path / "v.mp4"))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == RECORD_FPS + 1


def test_serve_saves_video_and_chat_on_exit(tmp_path: Path) -> None:
    frame = np.zeros((120, 160, 3), np.uint8)
    view = DebugView(lambda: frame, lambda: lambda _f: [], lambda: None, record_dir=tmp_path)
    view.add_line("you", "Hello.")
    with view.serve(0) as url:
        urllib.request.urlopen(url, timeout=5).read()
        time.sleep(0.3)
    assert view.recorder is not None
    assert view.recorder.path.stat().st_size > 0
    assert view.recorder.path.with_suffix(".txt").read_text().endswith("you: Hello.\n")
