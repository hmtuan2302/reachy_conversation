"""Local VieNeu-TTS v3 Turbo (pnnbao-ump/VieNeu-TTS-v3-Turbo), streamed. `uv sync --extra vieneu`.

`Vieneu().infer_stream(text, voice=...)` yields float32 chunks at 48 kHz; we turn them into the
24 kHz mono PCM16 the player expects. Runs on CPU via ONNX Runtime; first run downloads ~0.5 GB.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from reachy_demo.audio.sources import RATE
from reachy_demo.voice.tts import TTSError

DEFAULT_VIENEU_VOICE = ""  # empty = the model's default preset voice
MODEL_DIR = Path("models/vieneu")
BYTES_PER_S = RATE * 2
# shortcut: length guessed from text, measured ~20 chars/s on Vietnamese answers (2026-11-17);
# a lower value buffers more. Use the model's phoneme count if answers still gap at the end.
CHARS_PER_S = 18.0
MARGIN_S = 0.3  # covers chunk-arrival jitter (chunks are ~0.32 s)


def _fetch_real_files(repo: str, files: list[str], subfolder: str | None) -> Path:
    """VieNeu's graph/codec fetch, into MODEL_DIR as real files instead of the HF cache.

    onnxruntime >= 1.24 (reachy-mini pins 1.27) refuses external weights (.data) whose path
    resolves outside the model's folder, which the HF cache's per-file symlinks always do.
    """
    from huggingface_hub import hf_hub_download

    local = MODEL_DIR / repo.replace("/", "--")
    last = None
    for fn in files:
        try:
            last = hf_hub_download(repo, fn, subfolder=subfolder or None, local_dir=local)
        except Exception:
            if not fn.endswith(".json"):  # same rule as VieNeu: optional meta may be missing
                raise
    return Path(str(last)).parent


@cache
def _model() -> Any:
    try:
        from vieneu import Vieneu
        from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
    except ImportError as e:
        raise TTSError("TTS_PROVIDER=vieneu needs the package: uv sync --extra vieneu") from e
    # shortcut: patches a private VieNeu helper (checked on vieneu 3.8.4); drop it once VieNeu
    # or onnxruntime handles HF cache symlinks.
    OnnxV3LiteEngine._fetch = staticmethod(_fetch_real_files)
    return Vieneu()  # loads once (downloads ~0.5 GB on the first run)


def to_pcm24k(chunk: Any) -> bytes:
    """48 kHz float32 [-1, 1] → 24 kHz PCM16 by averaging sample pairs (a cheap low-pass)."""
    x = np.asarray(chunk, dtype=np.float32).reshape(-1)
    # shortcut: drops one odd tail sample per chunk (~20 µs), use a real resampler if it clicks
    x = x[: len(x) // 2 * 2].reshape(-1, 2).mean(axis=1)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def buffer_needed(total_s: float, made_s: float, rate: float) -> float:
    """Seconds of audio to hold before playing, so the speaker never waits for the model.

    `rate` is audio seconds made per wall second. Below 1 the model falls behind playback by
    (1/rate - 1) s for each second still to make, so that much must be buffered up front.
    """
    return max(total_s - made_s, 0.0) * max(1 / rate - 1, 0.0) + MARGIN_S


async def vieneu_stream(
    text: str, *, voice: str = DEFAULT_VIENEU_VOICE, ref_audio: Path | None = None
) -> AsyncIterator[bytes]:
    """Yield 24 kHz mono PCM16 chunks of `text` spoken by the local model.

    With `ref_audio` (a 3-8 s WAV), it clones that voice instead of using `voice`.
    A thread keeps generating while earlier chunks play: pulling one chunk at a time left
    the model idle during playback, then the speaker starved waiting for the next chunk.
    The first chunks are held until enough audio is buffered to play without gaps.
    """
    kw: dict[str, Any] = {"voice": voice} if voice else {}
    if ref_audio is not None:
        kw = {"ref_audio": str(ref_audio)}  # the SDK caches the clip by content
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
    stop = threading.Event()

    def put(item: bytes | BaseException | None) -> None:
        with contextlib.suppress(RuntimeError):  # the loop already closed on exit
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def produce() -> None:
        try:
            for chunk in _model().infer_stream(text, **kw):
                if stop.is_set():
                    return
                if pcm := to_pcm24k(chunk):
                    put(pcm)
            put(None)
        except BaseException as e:  # handed to the consumer, which raises it
            put(e)

    threading.Thread(target=produce, daemon=True).start()
    # On this laptop the model is often slower than real time (RTF 1.1-1.6 with the agent
    # running), so playing each chunk at once starved the speaker between chunks: choppy speech.
    total_s = len(text) / CHARS_PER_S
    held: list[bytes] | None = []  # None once playback started
    made_s = 0.0
    start = time.monotonic()
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, TTSError):
                raise item
            if isinstance(item, BaseException):  # the SDK has no documented error type
                raise TTSError(f"VieNeu TTS failed: {item}") from item
            made_s += len(item) / BYTES_PER_S
            if held is None:
                yield item
                continue
            held.append(item)
            # speed since the request, startup included: cautious, since the speed varies
            rate = made_s / max(time.monotonic() - start, 1e-6)
            if made_s >= buffer_needed(total_s, made_s, rate):
                for h in held:
                    yield h
                held = None
        for h in held or []:  # finished before the buffer filled: play it all
            yield h
    finally:
        stop.set()  # interrupted or done: the thread stops at its next chunk
