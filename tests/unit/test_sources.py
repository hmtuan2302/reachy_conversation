import io
import sys
import types
import wave
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from reachy_demo.audio.sources import (
    Recording,
    mic_chunks,
    stdin_chunks,
    to_24k_mono_pcm16,
    wav_chunks,
)


def _wav(path: Path, samples: np.ndarray, rate: int, width: int = 2, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def test_to_24k_length() -> None:
    assert len(to_24k_mono_pcm16(np.zeros(16000, dtype=np.int16), 16000)) == 48000


def test_stereo_is_downmixed() -> None:
    stereo = np.array([[1000, 3000]] * 24000, dtype=np.int16)
    out = np.frombuffer(to_24k_mono_pcm16(stereo, 24000), dtype=np.int16)
    assert len(out) == 24000 and int(out[0]) == 2000


async def test_wav_chunks_total_bytes(tmp_path: Path) -> None:
    sine = (np.sin(np.arange(16000) / 5) * 8000).astype(np.int16)
    chunks = [c async for c in wav_chunks(_wav(tmp_path / "a.wav", sine, 16000), realtime=False)]
    assert sum(map(len, chunks)) == 48000
    assert all(len(c) <= 9600 for c in chunks)


async def test_empty_wav_yields_nothing(tmp_path: Path) -> None:
    path = _wav(tmp_path / "e.wav", np.zeros(0, dtype=np.int16), 16000)
    assert [c async for c in wav_chunks(path, realtime=False)] == []


async def test_8bit_wav_rejected(tmp_path: Path) -> None:
    path = _wav(tmp_path / "b.wav", np.zeros(100, dtype=np.uint8), 16000, width=1)
    with pytest.raises(ValueError, match="16-bit"):
        [c async for c in wav_chunks(path, realtime=False)]


@pytest.mark.parametrize("content", [b"", b"not a wav file"])
async def test_non_wav_file_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="x.wav"):
        [c async for c in wav_chunks(path, realtime=False)]


async def test_mic_open_failure_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class PortAudioError(Exception):
        pass

    def refuse(**_kw: object) -> None:
        raise PortAudioError("Invalid sample rate")

    fake = types.SimpleNamespace(PortAudioError=PortAudioError, RawInputStream=refuse)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    with pytest.raises(RuntimeError, match="sample rate"):
        [c async for c in mic_chunks()]


async def test_mic_retries_once_when_stream_start_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PortAudioError(Exception):
        pass

    starts: list[int] = []

    class FlakyStream:
        def __init__(self, callback: Any, **_kw: object) -> None:
            self.callback = callback

        def start(self) -> None:
            starts.append(1)
            if len(starts) == 1:
                raise PortAudioError("Error starting stream: Wait timed out")
            self.callback(b"\x01\x00", 1, None, None)

        def stop(self) -> None: ...

        def close(self) -> None: ...

    fake = types.SimpleNamespace(PortAudioError=PortAudioError, RawInputStream=FlakyStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    gen = cast(AsyncGenerator[bytes, None], mic_chunks())
    assert await anext(gen) == b"\x01\x00"
    await gen.aclose()
    assert len(starts) == 2


async def test_mic_start_failure_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class PortAudioError(Exception):
        pass

    class DeadStream:
        def __init__(self, **_kw: object) -> None: ...

        def start(self) -> None:
            raise PortAudioError("Error starting stream: Wait timed out")

        def close(self) -> None: ...

    fake = types.SimpleNamespace(PortAudioError=PortAudioError, RawInputStream=DeadStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    with pytest.raises(RuntimeError, match="cannot open microphone.*timed out"):
        [c async for c in mic_chunks()]


async def test_stdin_chunks_are_200ms_and_drop_odd_tail_byte() -> None:
    raw = io.BytesIO(b"\x01\x00" * 7200 + b"\x07")  # 0.3 s of PCM + a stray byte
    chunks = [c async for c in stdin_chunks(raw)]
    assert [len(c) for c in chunks] == [9600, 4800]


async def test_stdin_chunks_empty_input_yields_nothing() -> None:
    assert [c async for c in stdin_chunks(io.BytesIO(b""))] == []


async def test_recording_tees_audio_and_saves_wav(tmp_path: Path) -> None:
    rec = Recording()
    passed = [c async for c in rec.tee(_aiter(b"\x01\x00", b"\x02\x00"))]
    assert passed == [b"\x01\x00", b"\x02\x00"]
    path = rec.save(tmp_path / "recordings", "mic")
    assert path is not None and path.parent == tmp_path / "recordings"
    assert path.name.endswith("-mic.wav")
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 24000 and w.readframes(w.getnframes()) == b"\x01\x00\x02\x00"


def test_empty_recording_saves_nothing(tmp_path: Path) -> None:
    assert Recording().save(tmp_path, "mic") is None
    assert list(tmp_path.iterdir()) == []


async def _aiter(*parts: bytes) -> AsyncIterator[bytes]:
    for p in parts:
        yield p
