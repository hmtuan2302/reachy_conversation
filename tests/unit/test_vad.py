import numpy as np

from reachy_demo.audio.vad import EnergyVAD


def _pcm(amp: int, ms: int) -> bytes:
    return np.full(24 * ms, amp, dtype="<i2").tobytes()


def test_quiet_is_not_speech() -> None:
    assert not EnergyVAD().feed(_pcm(100, 200))


def test_loud_150ms_starts_speech() -> None:
    v = EnergyVAD()
    assert not v.feed(_pcm(2000, 100))
    assert v.feed(_pcm(2000, 100))
    assert v.speaking


def test_short_click_is_ignored() -> None:
    v = EnergyVAD()
    v.feed(_pcm(2000, 100))
    assert not v.feed(_pcm(0, 200))
    assert not v.feed(_pcm(2000, 100))  # the quiet chunk reset the onset count


def test_hangover_keeps_speech_then_ends() -> None:
    v = EnergyVAD()
    v.feed(_pcm(2000, 200))
    assert v.feed(_pcm(0, 400))
    assert not v.feed(_pcm(0, 100))


def test_empty_chunk_keeps_state() -> None:
    v = EnergyVAD()
    v.feed(_pcm(2000, 200))
    assert v.feed(b"")
