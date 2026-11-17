import numpy as np
import pytest

from reachy_demo.voice.speaker_id import KEEP_S, SpeakerGate, to_16k, voiceprint

RATE = 24_000
POS_1S = b"\x10\x00" * RATE  # first sample > 0
NEG_1S = b"\xf0\xff" * RATE  # first sample < 0
PRINT = np.array([1.0, 0.0])
seen: list[int] = []


def fake_embed(x: np.ndarray, rate: int) -> np.ndarray:
    assert rate == RATE
    seen.append(len(x))
    return np.array([1.0, 0.0]) if x[0] > 0 else np.array([0.6, 0.8])  # cos 1.0 / 0.6


def _take(gate: SpeakerGate) -> tuple[str, float | None]:
    verdict, score = gate.take()
    return verdict, None if score is None else round(score, 6)


def test_owner_guest_unsure() -> None:
    gate = SpeakerGate(fake_embed, PRINT, high=0.7, low=0.4)
    gate.feed(POS_1S * 2, speaking=True)
    assert _take(gate) == ("owner", 1.0)
    gate.feed(NEG_1S * 2, speaking=True)
    assert _take(gate) == ("unsure", 0.6)
    strict = SpeakerGate(fake_embed, PRINT, high=0.9, low=0.7)
    strict.feed(NEG_1S * 2, speaking=True)
    assert _take(strict) == ("guest", 0.6)


def test_too_short_is_unsure() -> None:
    seen.clear()
    gate = SpeakerGate(fake_embed, PRINT, high=0.7, low=0.4)
    gate.feed(POS_1S[:RATE], speaking=True)  # 0.5 s
    assert gate.take() == ("unsure", None)
    assert seen == []


def test_silence_not_kept() -> None:
    gate = SpeakerGate(fake_embed, PRINT, high=0.7, low=0.4)
    gate.feed(POS_1S * 3, speaking=False)
    assert gate.take() == ("unsure", None)


def test_take_clears() -> None:
    gate = SpeakerGate(fake_embed, PRINT, high=0.7, low=0.4)
    gate.feed(POS_1S * 2, speaking=True)
    assert gate.take()[0] == "owner"
    assert gate.take() == ("unsure", None)


def test_buffer_keeps_last_seconds() -> None:
    seen.clear()
    gate = SpeakerGate(fake_embed, PRINT, high=0.7, low=0.4)
    for _ in range(10):
        gate.feed(POS_1S, speaking=True)
    gate.take()
    assert seen == [int(KEEP_S * RATE)]


def test_voiceprint_is_unit_mean() -> None:
    v = voiceprint([np.array([2.0, 0.0]), np.array([0.0, 3.0])])
    assert v == pytest.approx([0.70710678, 0.70710678])


def test_to_16k_keeps_a_speech_tone() -> None:
    for rate in (24_000, 48_000):
        x = np.sin(2 * np.pi * 1000 * np.arange(rate) / rate).astype(np.float32)  # 1 s, 1 kHz
        y = to_16k(x, rate)
        assert y is not None
        assert len(y) == 16_000
        assert np.sqrt(np.mean(y[1000:-1000] ** 2)) == pytest.approx(0.7071, abs=0.01)


def test_to_16k_skips_slow_ratios() -> None:
    # 44.1 kHz needs a 14k-tap filter here: None means "let sherpa resample it"
    assert to_16k(np.zeros(44_100, dtype=np.float32), 44_100) is None


def test_own_voice_window() -> None:
    gate = SpeakerGate(fake_embed, PRINT, high=0.7, low=0.4)
    assert not gate.own_voice(10.0)
    gate.mark_own_voice(12.0)
    assert gate.own_voice(12.4) and not gate.own_voice(12.6)  # 0.5 s echo tail
