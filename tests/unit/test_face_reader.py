import pytest

from reachy_demo.robot import adapter


def test_daemon_time_maps_to_the_laptop_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            ({"detected": True, "x": 0.1, "y": -0.2, "ts": 50.0}, 1000.3),  # slow reply
            ({"detected": True, "x": 0.1, "y": -0.2, "ts": 51.0}, 1001.02),  # fast reply
            ({"detected": True, "x": 0.2, "y": 0.0, "ts": 52.0}, 1002.4),
            ({"detected": False, "ts": 53.0}, 1003.0),
            ({"detected": True, "x": 0.2, "y": 0.0, "ts": None}, 1004.0),
        ]
    )
    now = [0.0]

    def daemon(host: str, path: str) -> dict[str, object]:
        face, now[0] = next(replies)
        return {"face_target": face}

    monkeypatch.setattr(adapter, "_daemon_json", daemon)
    read = adapter.face_reader("robot", clock=lambda: now[0])
    assert read() == pytest.approx((0.1, -0.2, 1000.3))
    assert read() == pytest.approx((0.1, -0.2, 1001.02))
    assert read() == pytest.approx((0.2, 0.0, 1002.02))  # 0.38 s late, by the best offset
    assert read() is None
    assert read() == (0.2, 0.0)  # an older daemon without ts
