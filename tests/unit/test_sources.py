import wave
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.audio.sources import to_24k_mono_pcm16, wav_chunks


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
