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
