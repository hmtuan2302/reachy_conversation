from typing import Any

import numpy as np
import pytest

from reachy_demo.voice import vieneu_tts
from reachy_demo.voice.tts import TTSError


def test_to_pcm24k_halves_rate_and_scales() -> None:
    pcm = vieneu_tts.to_pcm24k(np.array([1.0, 1.0, -0.5, -0.5, 0.2], dtype=np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [32767, -16383]


async def test_stream_converts_chunks_and_passes_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Fake:
        def infer_stream(self, text: str, **kw: Any) -> Any:
            seen.update(kw, text=text)
            yield np.zeros(4, dtype=np.float32)
            yield np.ones(2, dtype=np.float32)

    monkeypatch.setattr(vieneu_tts, "_model", Fake)
    out = [c async for c in vieneu_tts.vieneu_stream("xin chào", voice="Mai Anh")]
    assert out == [b"\x00\x00\x00\x00", b"\xff\x7f"]
    assert seen == {"text": "xin chào", "voice": "Mai Anh"}


async def test_sdk_error_becomes_tts_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def infer_stream(self, text: str, **kw: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(vieneu_tts, "_model", Fake)
    with pytest.raises(TTSError, match="boom"):
        [c async for c in vieneu_tts.vieneu_stream("hi")]


async def test_ref_audio_clones_instead_of_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    seen: dict[str, Any] = {}

    class Fake:
        def infer_stream(self, text: str, **kw: Any) -> Any:
            seen.update(kw)
            yield np.zeros(2, dtype=np.float32)

    monkeypatch.setattr(vieneu_tts, "_model", Fake)
    _ = [c async for c in vieneu_tts.vieneu_stream("hi", voice="Mai Anh", ref_audio=Path("a.wav"))]
    assert seen == {"ref_audio": "a.wav"}


async def test_keeps_generating_while_consumer_is_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import threading

    made = threading.Event()

    class Fake:
        def infer_stream(self, text: str, **kw: Any) -> Any:
            yield np.zeros(2, dtype=np.float32)
            yield np.zeros(2, dtype=np.float32)
            made.set()  # second chunk generated before the consumer asked for it

    monkeypatch.setattr(vieneu_tts, "_model", Fake)
    stream = vieneu_tts.vieneu_stream("hi")
    await anext(stream)
    assert await asyncio.to_thread(made.wait, 2.0)
    assert [c async for c in stream] == [b"\x00\x00"]


def test_buffer_needed_covers_the_shortfall() -> None:
    # 7 s to say, 1 s made, model at 0.8x real time: it falls 0.25 s behind per second left
    assert vieneu_tts.buffer_needed(7.0, 1.0, 0.8) == pytest.approx(6 * 0.25 + vieneu_tts.MARGIN_S)


def test_buffer_needed_fast_model_needs_only_the_margin() -> None:
    assert vieneu_tts.buffer_needed(7.0, 0.3, 1.5) == vieneu_tts.MARGIN_S
    assert (
        vieneu_tts.buffer_needed(2.0, 3.0, 0.5) == vieneu_tts.MARGIN_S
    )  # text was longer than guessed


async def test_slow_model_holds_audio_until_buffered(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    chunk = np.zeros(480, dtype=np.float32)  # 0.01 s of 48 kHz audio
    yielded_at: list[float] = []

    class Fake:  # 0.01 s of audio every 0.02 s: half real time, ~0.5 s of speech
        def infer_stream(self, text: str, **kw: Any) -> Any:
            for _ in range(50):
                time.sleep(0.02)
                yield chunk

    monkeypatch.setattr(vieneu_tts, "_model", Fake)
    t0 = time.monotonic()
    async for _ in vieneu_tts.vieneu_stream("x" * 9):  # guessed 0.5 s of speech
        yielded_at.append(time.monotonic() - t0)
    assert len(yielded_at) == 50  # nothing lost
    assert yielded_at[0] > 0.5  # held back until the buffer could cover the slow model


@pytest.mark.parametrize(
    ("raw", "want"), [("", "auto"), ("cpu", "cpu"), ("GPU", "cuda"), (" cuda ", "cuda")]
)
def test_device_from_env(monkeypatch: pytest.MonkeyPatch, raw: str, want: str) -> None:
    monkeypatch.setenv("TTS_DEVICE", raw)
    assert vieneu_tts.device_from_env() == want


def test_device_from_env_rejects_typo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_DEVICE", "gpuu")
    with pytest.raises(TTSError, match="cpu or gpu"):
        vieneu_tts.device_from_env()
