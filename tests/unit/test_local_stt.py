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


def _fake_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_stt, "_load_vad", lambda: lambda f: 0.0)
    monkeypatch.setattr(local_stt, "_load_smart_turn", lambda: lambda a: 0.0)


class _RecordingWhisper:
    devices: list[str] = []

    def __init__(self, model: str, device: str, compute_type: str) -> None:
        _RecordingWhisper.devices.append(device)

    def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
        return [], None


def test_auto_device_falls_back_to_cpu_without_cudnn(monkeypatch: pytest.MonkeyPatch) -> None:
    faster_whisper = pytest.importorskip("faster_whisper")
    import ctranslate2

    _fake_parts(monkeypatch)
    _RecordingWhisper.devices = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", _RecordingWhisper)
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(local_stt, "_cuda_libs_ok", lambda: False)
    load_models(device="")
    assert _RecordingWhisper.devices == ["cpu"]


def test_explicit_gpu_without_cudnn_is_stt_error(monkeypatch: pytest.MonkeyPatch) -> None:
    faster_whisper = pytest.importorskip("faster_whisper")
    _fake_parts(monkeypatch)
    monkeypatch.setattr(faster_whisper, "WhisperModel", _RecordingWhisper)
    monkeypatch.setattr(local_stt, "_cuda_libs_ok", lambda: False)
    with pytest.raises(SttError, match="cuDNN.*STT_DEVICE=cpu"):
        load_models(device="gpu")


def test_missing_extra_is_stt_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_module() -> Any:
        raise ModuleNotFoundError("No module named 'faster_whisper'")

    monkeypatch.setattr(local_stt, "_load_vad", no_module)
    with pytest.raises(SttError, match="uv sync --extra stt"):
        load_models()


def test_offline_download_is_stt_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline() -> Any:
        raise OSError("Cannot reach huggingface.co")

    monkeypatch.setattr(local_stt, "_load_vad", lambda: lambda f: 0.0)
    monkeypatch.setattr(local_stt, "_load_smart_turn", offline)
    with pytest.raises(SttError, match="network"):
        load_models()


def test_cpu_load_failure_does_not_suggest_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    faster_whisper = pytest.importorskip("faster_whisper")

    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("model.bin not found")

    _fake_parts(monkeypatch)
    monkeypatch.setattr(faster_whisper, "WhisperModel", broken)
    with pytest.raises(SttError) as err:
        load_models(device="cpu")
    assert "STT_DEVICE=cpu" not in str(err.value)


async def test_smart_turn_runs_off_loop() -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    def slow_turn(audio: np.ndarray) -> float:
        time.sleep(0.3)
        return 1.0

    models = fake()
    models.turn = slow_turn
    task = asyncio.create_task(ticker())
    audio = speech_pcm(1000) + silence_pcm(400)
    await collect(transcribe_local(chunks(audio), models=models, settings=TurnSettings()))
    task.cancel()
    assert ticks >= 15


async def test_source_keeps_flowing_while_whisper_runs() -> None:
    pulls: list[float] = []

    async def timed(pcm: bytes) -> AsyncIterator[bytes]:
        async for c in chunks(pcm):
            pulls.append(time.monotonic())
            await asyncio.sleep(0.02)
            yield c

    audio = speech_pcm(1000) + silence_pcm(400) + silence_pcm(1000)
    start = time.monotonic()
    await collect(transcribe_local(timed(audio), models=fake(delay=0.5), settings=TurnSettings()))
    # all 25 chunks pulled at ~20 ms pace, not stalled behind the 0.5 s transcription
    assert pulls[-1] - start < 0.9
