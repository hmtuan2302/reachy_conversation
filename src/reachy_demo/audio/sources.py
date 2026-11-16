"""Audio sources for live transcription: WAV file or laptop mic, as 24 kHz mono PCM16 chunks."""

from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

RATE = 24_000


def save_wav(path: Path, pcm: bytes) -> None:
    """Write 24 kHz mono PCM16 as a WAV file."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


class Recording:
    """Keeps a copy of the audio sent for transcription, to save for debugging on exit."""

    def __init__(self) -> None:
        self.pcm = bytearray()

    async def tee(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for chunk in chunks:
            self.pcm += chunk
            yield chunk

    def save(self, directory: Path, label: str) -> Path | None:
        """Write `<directory>/<YYYYmmdd-HHMMSS>-<label>.wav`; None if nothing was recorded."""
        if not self.pcm:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.wav"
        save_wav(path, bytes(self.pcm))
        return path


def to_24k_mono_pcm16(samples: np.ndarray, rate: int) -> bytes:
    """int16 samples, shape (n,) or (n, channels), at `rate` -> 24 kHz mono PCM16 bytes."""
    mono = samples.astype(np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if rate != RATE and len(mono):
        # shortcut: linear resample, no anti-alias filter; upgrade if accuracy suffers on 44.1k+
        n_out = round(len(mono) * RATE / rate)
        mono = np.interp(np.arange(n_out) * rate / RATE, np.arange(len(mono)), mono)
    return np.clip(np.round(mono), -32768, 32767).astype("<i2").tobytes()


async def wav_chunks(
    path: Path, *, realtime: bool = True, chunk_ms: int = 200
) -> AsyncIterator[bytes]:
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                raise ValueError(f"{path}: need 16-bit PCM WAV, got {8 * w.getsampwidth()}-bit")
            rate, channels = w.getframerate(), w.getnchannels()
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    except (wave.Error, EOFError) as e:
        raise ValueError(f"{path}: not a readable WAV file ({e or 'empty'})") from e
    pcm = to_24k_mono_pcm16(frames.reshape(-1, channels), rate)
    step = RATE * 2 * chunk_ms // 1000
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]
        if realtime:
            await asyncio.sleep(chunk_ms / 1000)


async def stdin_chunks(stream: BinaryIO, *, chunk_ms: int = 200) -> AsyncIterator[bytes]:
    """Raw 24 kHz mono s16le PCM from a pipe (e.g. Windows ffmpeg.exe -f s16le -), until EOF."""
    step = RATE * 2 * chunk_ms // 1000
    while chunk := await asyncio.to_thread(stream.read, step):
        if usable := chunk[: len(chunk) // 2 * 2]:  # a half sample can only be the stream's tail
            yield usable


async def mic_chunks(*, chunk_ms: int = 200) -> AsyncIterator[bytes]:
    try:
        import sounddevice as sd
    except ImportError as e:  # sounddevice is the optional `mic` extra
        raise RuntimeError("install the mic extra: uv sync --extra mic") from e
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def on_audio(data: bytes, _frames: int, _time: object, _status: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, bytes(data))

    stream = None
    for attempt in (1, 2):
        try:
            stream = sd.RawInputStream(
                samplerate=RATE,
                channels=1,
                dtype="int16",
                blocksize=RATE * chunk_ms // 1000,
                callback=on_audio,
            )
            stream.start()
            break
        except sd.PortAudioError as e:
            if stream is not None:
                stream.close()
                stream = None
            if attempt == 2:
                raise RuntimeError(f"cannot open microphone: {e}") from e
            # WSLg's Pulse source can be asleep; the first start may time out (-9987).
            await asyncio.sleep(0.5)
    assert stream is not None
    try:
        while True:
            yield await queue.get()
    finally:
        stream.stop()
        stream.close()
