import threading

from reachy_demo.robot.adapter import LatestFrame


def _running(read: object) -> tuple[LatestFrame, threading.Thread]:
    frames = LatestFrame(read)  # type: ignore[arg-type]
    t = threading.Thread(target=frames.run, daemon=True)
    t.start()
    return frames, t


def test_every_caller_gets_the_newest_frame() -> None:
    n = iter(range(10**9))
    frames, t = _running(lambda: next(n))
    a, b = frames.get(), frames.get()
    frames.stop()
    t.join(1)
    assert a is not None and b is not None and b >= a  # a pull no longer steals it


def test_no_frame_or_a_stale_one_is_none() -> None:
    frames, t = _running(lambda: None)
    assert frames.get(wait_s=0.05) is None
    frames.stop()
    t.join(1)
    old = LatestFrame(lambda: None, clock=lambda: 10.0)
    old._frame = (0.0, "frame")
    assert old.get(wait_s=0.0) is None  # 10 s old
