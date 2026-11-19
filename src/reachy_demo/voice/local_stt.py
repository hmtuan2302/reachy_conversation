"""Local speech-to-text: Silero VAD, Smart Turn v3.2, then faster-whisper (STT_PROVIDER=local).

Same shape as the HF `transcribe_stream`: 24 kHz int16 PCM in, one transcript per finished turn.
`faster_whisper` is imported inside functions so the HF path works without the `stt` extra.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reachy_demo.voice.turns import FRAME, RATE16, Resampler24to16, TurnDetector, TurnSettings

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_DIR = Path("models/smart-turn")
DEFAULT_MODEL = "large-v3-turbo"
LANGUAGES = ("vi", "en")
NO_SPEECH_MAX = 0.6  # Whisper segments above this are likely hallucinated on noise

log = logging.getLogger(__name__)


class SttError(RuntimeError):
    """A local STT model failed to load."""


@dataclass
class LocalModels:
    vad: Callable[[np.ndarray], float]  # 512-sample frame -> speech probability
    turn: Callable[[np.ndarray], float]  # last 8 s of the turn -> probability it is complete
    transcribe: Callable[[np.ndarray], str]  # whole turn at 16 kHz -> text


def pick_language(all_probs: list[tuple[str, float]]) -> str:
    """The more likely of vi and en; en when Whisper lists neither."""
    probs = dict(all_probs)
    return max(LANGUAGES, key=lambda lang: probs.get(lang, -1.0 if lang == "vi" else 0.0))


def _session(path: str) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def _load_vad() -> Callable[[np.ndarray], float]:
    """Streaming Silero: keeps its state and 64-sample context between frames.

    faster-whisper's own SileroVADModel resets the state on every call, so it can't stream.
    """
    from faster_whisper.utils import get_assets_path

    session = _session(os.path.join(get_assets_path(), "silero_vad_v6.onnx"))
    h = np.zeros((1, 1, 128), np.float32)
    c = np.zeros((1, 1, 128), np.float32)
    context = np.zeros(64, np.float32)

    def vad(frame: np.ndarray) -> float:
        nonlocal h, c, context
        x = np.concatenate([context, frame.astype(np.float32)])[None]
        prob, h, c = session.run(None, {"input": x, "h": h, "c": c})
        context = frame[-64:].astype(np.float32)
        return float(prob.reshape(-1)[0])

    return vad


def _load_smart_turn() -> Callable[[np.ndarray], float]:
    from faster_whisper.feature_extractor import FeatureExtractor

    path = SMART_TURN_DIR / SMART_TURN_FILE
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        log.info("downloading %s to %s", SMART_TURN_FILE, SMART_TURN_DIR)
        hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_dir=SMART_TURN_DIR)
    session = _session(str(path))
    features = FeatureExtractor(chunk_length=8)

    def turn(audio: np.ndarray) -> float:
        return float(
            session.run(None, {"input_features": smart_turn_features(audio, features)})[0].reshape(
                -1
            )[0]
        )

    return turn


def smart_turn_features(audio: np.ndarray, extractor: Any) -> np.ndarray:
    """Whisper log-mel (1, 80, 800) of 8 s of audio, normalized like Smart Turn's training."""
    a = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    return np.asarray(extractor(a.astype(np.float32), padding=0), np.float32)[None]


def _preload_cuda_libs() -> None:
    """Load cuBLAS and cuDNN from the pip `nvidia-*` wheels, which are not on the library path.

    ctranslate2 dlopens them by name; once loaded globally, that lookup finds them.
    """
    import ctypes
    import importlib.util

    for pkg, names in (
        ("nvidia.cublas", ("libcublasLt.so.12", "libcublas.so.12")),
        ("nvidia.cudnn", ("libcudnn_graph.so.9", "libcudnn.so.9", "libcudnn_ops.so.9")),
    ):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue  # system CUDA install, or missing: WhisperModel reports it
        lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
        for name in names:
            if (lib / name).is_file():
                ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)


def _load_whisper(model: str, device: str) -> Callable[[np.ndarray], str]:
    import ctranslate2
    import faster_whisper

    if device == "gpu" or (device == "" and ctranslate2.get_cuda_device_count() > 0):
        dev, compute = "cuda", "int8_float16"
    else:
        dev, compute = "cpu", "int8"
    if dev == "cuda":
        _preload_cuda_libs()
    try:
        whisper = faster_whisper.WhisperModel(model, device=dev, compute_type=compute)
        # Segments are lazy: consume them so a missing CUDA library fails here, not mid-turn.
        segments, _ = whisper.transcribe(np.zeros(RATE16, np.float32), language="en", beam_size=1)
        list(segments)
    except (RuntimeError, OSError, ValueError) as e:
        raise SttError(f"Whisper failed to load on {dev}: {e}. Try STT_DEVICE=cpu") from e
    log.info("whisper %s loaded on %s (%s)", model, dev, compute)

    def transcribe(audio: np.ndarray) -> str:
        _, _, probs = whisper.detect_language(audio)
        lang = pick_language(probs)
        segments, _ = whisper.transcribe(
            audio,
            language=lang,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segments if s.no_speech_prob <= NO_SPEECH_MAX).strip()
        log.info("turn transcribed", extra={"lang": lang, "seconds": len(audio) / RATE16})
        return text

    return transcribe


def load_models(model: str = DEFAULT_MODEL, device: str = "") -> LocalModels:
    """Load and warm up all three models. `device`: "cpu", "gpu" or "" (GPU if found)."""
    return LocalModels(
        vad=_load_vad(), turn=_load_smart_turn(), transcribe=_load_whisper(model, device)
    )


def models_from_env(env: Mapping[str, str]) -> LocalModels:
    device = env.get("STT_DEVICE", "").strip().lower()
    if device not in ("", "cpu", "gpu"):
        raise SttError(f"STT_DEVICE must be cpu or gpu, got {device!r}")
    return load_models(env.get("STT_MODEL", "").strip() or DEFAULT_MODEL, device)


async def transcribe_local(
    pcm24k: AsyncIterator[bytes],
    *,
    models: LocalModels | None = None,
    settings: TurnSettings | None = None,
) -> AsyncIterator[str]:
    """Yield one transcript per finished turn. Raises SttError if the models fail to load."""
    if models is None:
        models = await asyncio.to_thread(models_from_env, os.environ)
    detector = TurnDetector(settings or TurnSettings.from_env(os.environ), models.turn)
    resampler = Resampler24to16()
    pending = np.zeros(0, np.float32)

    async def finish(audio: np.ndarray) -> str:
        log.info(
            "turn ended", extra={"seconds": len(audio) / RATE16, "smart_turn": detector.last_score}
        )
        return (await asyncio.to_thread(models.transcribe, audio)).strip()

    async for chunk in pcm24k:
        pending = np.concatenate([pending, resampler.feed(chunk)])
        n = len(pending) - len(pending) % FRAME
        for frame in pending[:n].reshape(-1, FRAME):
            if (audio := detector.feed(frame, models.vad(frame))) is not None and (
                text := await finish(audio)
            ):
                yield text
        pending = pending[n:]
    if (audio := detector.flush()) is not None and (text := await finish(audio)):
        yield text
