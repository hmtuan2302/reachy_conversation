import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from reachy_demo.voice import local_stt
from reachy_demo.voice.local_stt import (
    LocalModels,
    SttError,
    load_models,
    pick_language,
    transcribe_local,
)
from reachy_demo.voice.turns import TurnSettings

RATE24 = 24_000


def speech_pcm(ms: int) -> bytes:
    return (np.ones(RATE24 * ms // 1000, np.int16) * 10_000).tobytes()


def silence_pcm(ms: int) -> bytes:
    return np.zeros(RATE24 * ms // 1000, np.int16).tobytes()


async def chunks(pcm: bytes, size: int = 4800) -> AsyncIterator[bytes]:
    for i in range(0, len(pcm), size):
        yield pcm[i : i + size]


async def collect(lines: AsyncIterator[str]) -> list[str]:
    return [line async for line in lines]


def fake(score: float = 1.0, text: str = "xin chào", delay: float = 0.0) -> LocalModels:
    def transcribe(audio: np.ndarray) -> str:
        time.sleep(delay)
        return text

    return LocalModels(
        vad=lambda f: 0.9 if np.abs(f).max() > 0.1 else 0.0,
        turn=lambda a: score,
        transcribe=transcribe,
    )


async def test_transcribe_local_yields_one_text_per_turn() -> None:
    audio = speech_pcm(1000) + silence_pcm(400) + speech_pcm(1000) + silence_pcm(400)
    out = await collect(transcribe_local(chunks(audio), models=fake(), settings=TurnSettings()))
    assert out == ["xin chào", "xin chào"]


async def test_transcribe_local_flushes_at_end() -> None:
    out = await collect(
        transcribe_local(chunks(speech_pcm(1000)), models=fake(0.0), settings=TurnSettings())
    )
    assert out == ["xin chào"]


async def test_empty_transcript_is_skipped() -> None:
    audio = speech_pcm(1000) + silence_pcm(400)
    out = await collect(
        transcribe_local(chunks(audio), models=fake(text="  "), settings=TurnSettings())
    )
    assert out == []


async def test_transcribe_runs_off_loop() -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    audio = speech_pcm(1000) + silence_pcm(400)
    await collect(transcribe_local(chunks(audio), models=fake(delay=0.3), settings=TurnSettings()))
    task.cancel()
    assert ticks >= 15


def test_pick_language_limits_to_vi_en() -> None:
    assert pick_language([("ja", 0.5), ("vi", 0.3), ("en", 0.2)]) == "vi"
    assert pick_language([("en", 0.6), ("vi", 0.1)]) == "en"
    assert pick_language([("ja", 0.9)]) == "en"


def test_gpu_load_failure_message(monkeypatch: pytest.MonkeyPatch) -> None:
    faster_whisper = pytest.importorskip("faster_whisper")

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("libcudnn_ops.so.9: cannot open shared object file")

    monkeypatch.setattr(faster_whisper, "WhisperModel", broken)
    monkeypatch.setattr(local_stt, "_load_vad", lambda: lambda f: 0.0)
    monkeypatch.setattr(local_stt, "_load_smart_turn", lambda: lambda a: 0.0)
    with pytest.raises(SttError, match="STT_DEVICE=cpu"):
        load_models(device="gpu")


def test_gpu_failure_inside_lazy_segments_is_caught_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faster_whisper = pytest.importorskip("faster_whisper")

    class LazyBroken:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[Any, None]:
            def segments() -> Any:
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
                yield

            return segments(), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", LazyBroken)
    monkeypatch.setattr(local_stt, "_load_vad", lambda: lambda f: 0.0)
    monkeypatch.setattr(local_stt, "_load_smart_turn", lambda: lambda a: 0.0)
    with pytest.raises(SttError, match="libcublas"):
        load_models(device="gpu")
