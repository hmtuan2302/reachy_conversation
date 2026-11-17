"""Local VieNeu-TTS v3 Turbo (pnnbao-ump/VieNeu-TTS-v3-Turbo), streamed. `uv sync --extra vieneu`.

`Vieneu().infer_stream(text, voice=...)` yields float32 chunks at 48 kHz; we turn them into the
24 kHz mono PCM16 the player expects. Runs on CPU via ONNX Runtime; first run downloads ~0.5 GB.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from reachy_demo.voice.tts import TTSError

DEFAULT_VIENEU_VOICE = ""  # empty = the model's default preset voice
MODEL_DIR = Path("models/vieneu")


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


async def vieneu_stream(
    text: str, *, voice: str = DEFAULT_VIENEU_VOICE, ref_audio: Path | None = None
) -> AsyncIterator[bytes]:
    """Yield 24 kHz mono PCM16 chunks of `text` spoken by the local model.

    With `ref_audio` (a 3-8 s WAV), it clones that voice instead of using `voice`.
    A thread keeps generating while earlier chunks play: pulling one chunk at a time left
    the model idle during playback, then the speaker starved waiting for the next chunk.
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
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, TTSError):
                raise item
            if isinstance(item, BaseException):  # the SDK has no documented error type
                raise TTSError(f"VieNeu TTS failed: {item}") from item
            yield item
    finally:
        stop.set()  # interrupted or done: the thread stops at its next chunk
