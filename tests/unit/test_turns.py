import numpy as np

from reachy_demo.voice.turns import FRAME, Resampler24to16, TurnDetector, TurnSettings

FRAME_MS = 32


def _frames(ms: int, prob: float) -> list[tuple[np.ndarray, float]]:
    return [(np.zeros(FRAME, np.float32), prob) for _ in range(ms // FRAME_MS)]


def speech(ms: int) -> list[tuple[np.ndarray, float]]:
    return _frames(ms, 0.9)


def silence(ms: int) -> list[tuple[np.ndarray, float]]:
    return _frames(ms, 0.0)


def run(det: TurnDetector, frames: list[tuple[np.ndarray, float]]) -> list[np.ndarray]:
    return [a for f, p in frames if (a := det.feed(f, p)) is not None]


def test_complete_turn_ends_at_pause() -> None:
    calls: list[int] = []
    det = TurnDetector(TurnSettings(), lambda a: calls.append(len(a)) or 0.9)
    out = run(det, speech(1000) + silence(224))
    assert len(out) == 1 and len(calls) == 1


def test_incomplete_turn_waits_then_resumes_same_turn() -> None:
    det = TurnDetector(TurnSettings(), lambda a: 0.1)
    assert run(det, speech(800) + silence(500) + speech(800)) == []
    out = run(det, silence(2016))
    assert len(out) == 1 and len(out[0]) >= 16 * 1600  # both speech parts in one turn


def test_max_silence_ends_turn_when_smart_turn_says_no() -> None:
    det = TurnDetector(TurnSettings(), lambda a: 0.0)
    assert run(det, speech(1000) + silence(1984)) == []
    assert len(run(det, silence(64))) == 1


def test_smart_turn_asked_once_per_pause() -> None:
    calls: list[int] = []
    det = TurnDetector(TurnSettings(), lambda a: calls.append(1) or 0.0)
    run(det, speech(1000) + silence(1500))
    assert len(calls) == 1


def test_lead_in_kept() -> None:
    det = TurnDetector(TurnSettings(), lambda a: 1.0)
    out = run(det, silence(1000) + speech(500) + silence(224))
    assert len(out[0]) >= 16 * (300 + 500)


def test_short_blip_is_not_a_turn() -> None:
    calls: list[int] = []
    det = TurnDetector(TurnSettings(), lambda a: calls.append(1) or 1.0)
    assert run(det, speech(160) + silence(2500)) == [] and calls == []


def test_flush_returns_open_turn() -> None:
    det = TurnDetector(TurnSettings(), lambda a: 0.0)
    run(det, speech(1000))
    assert det.flush() is not None and det.flush() is None


def test_smart_turn_gets_at_most_8s() -> None:
    seen: list[int] = []
    det = TurnDetector(TurnSettings(), lambda a: seen.append(len(a)) or 1.0)
    run(det, speech(12000) + silence(224))
    assert seen == [8 * 16000]


def test_resampler_ratio_across_odd_chunks() -> None:
    r = Resampler24to16()
    pcm = (np.ones(24000, np.int16) * 1000).tobytes()
    total = sum(len(r.feed(pcm[i : i + 1001])) for i in range(0, len(pcm), 1001))
    assert total == 16000


def test_resampler_scales_to_unit_range() -> None:
    out = Resampler24to16().feed((np.ones(3, np.int16) * 16384).tobytes())
    assert np.allclose(out, 0.5)


def test_settings_from_env() -> None:
    s = TurnSettings.from_env(
        {
            "STT_PAUSE_MS": "300",
            "STT_TURN_THRESHOLD": "0.7",
            "STT_MAX_SILENCE_MS": "",
            "VAD_THRESHOLD": "0.4",
        }
    )
    got = (s.pause_ms, s.turn_threshold, s.max_silence_ms, s.vad_threshold)
    assert got == (300, 0.7, 2000, 0.4)
