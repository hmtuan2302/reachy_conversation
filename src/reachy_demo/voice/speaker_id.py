"""Speaker verification: is this turn the owner's voice?

The mic tap feeds `SpeakerGate` the audio heard while the user was talking. When a turn's
transcript arrives, `take()` scores that audio against the owner's voiceprint and clears it.
The embedding model is injected (`Embed`), so this module needs no model library;
`load_sherpa_embed` builds the real one (sherpa-onnx, CAM++).
"""

from __future__ import annotations

import math
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

RATE = 24_000  # the app's mic rate
KEEP_S = 8.0  # speech kept since the last turn; older speech (untranscribed side talk) drops
MIN_S = 1.0  # shorter speech is too little to judge
DEFAULT_MODEL = Path("models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx")
VOICEPRINT = Path("data/voice/owner.npy")
# Measured 2026-11-17, enrolled on my_voice1.wav (CAM++): owner my_voice_2.wav 0.73, another
# voice 0.05, mic_check.wav 0.24. 0.5 is CAM++'s usual operating point; tune on the venue mic.
DEFAULT_HIGH = 0.5
DEFAULT_LOW = 0.3

Embed = Callable[[np.ndarray, int], np.ndarray]  # float32 samples in [-1, 1], rate -> embedding
Verdict = Literal["owner", "guest", "unsure"]


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def voiceprint(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean of the L2-normalised embeddings, normalised again."""
    return normalize(np.mean([normalize(np.asarray(e, dtype=np.float64)) for e in embeddings], 0))


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


class SpeakerGate:
    def __init__(
        self,
        embed: Embed,
        print_: np.ndarray,
        high: float,
        low: float,
        min_s: float = MIN_S,
        keep_s: float = KEEP_S,
    ) -> None:
        self._embed = embed
        self._print = normalize(np.asarray(print_, dtype=np.float64))
        self.high = high
        self.low = low
        self._min = int(min_s * RATE) * 2
        self._max = int(keep_s * RATE) * 2
        self._chunks: deque[bytes] = deque()
        self._bytes = 0

    def feed(self, chunk: bytes, speaking: bool) -> None:
        """24 kHz mono PCM16 chunk; only speech is kept, at most `keep_s` of it."""
        if not speaking:
            return
        self._chunks.append(chunk)
        self._bytes += len(chunk)
        while self._bytes > self._max:
            extra = self._bytes - self._max
            head = self._chunks[0]
            if len(head) <= extra:
                self._bytes -= len(self._chunks.popleft())
            else:
                self._chunks[0] = head[extra:]
                self._bytes -= extra

    def take(self) -> tuple[Verdict, float | None]:
        """Score the speech since the last call, then forget it. None score: too short."""
        pcm = b"".join(self._chunks)
        self._chunks.clear()
        self._bytes = 0
        if len(pcm) < self._min:
            return "unsure", None
        emb = normalize(np.asarray(self._embed(pcm16_to_float(pcm), RATE), dtype=np.float64))
        score = float(emb @ self._print)
        if score >= self.high:
            return "owner", score
        return ("guest" if score < self.low else "unsure"), score


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> mono float32 samples in [-1, 1], and its sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM WAV")
        rate, channels = w.getframerate(), w.getnchannels()
        x = pcm16_to_float(w.readframes(w.getnframes()))
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32), rate


def to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    """Resample to 16 kHz: zero-stuff by L, windowed-sinc low-pass, keep every M-th sample.

    Scores match sherpa-onnx's own resampler within 0.03 on the test clips (24 and 48 kHz).
    """
    if rate == 16_000:
        return x
    g = math.gcd(16_000, rate)
    up, down = 16_000 // g, rate // g
    m = max(up, down)
    n = np.arange(-16 * m, 16 * m + 1)
    h = np.sinc(n / m) * np.hamming(len(n)) / m
    y = np.zeros(len(x) * up, dtype=np.float32)
    y[::up] = x * up
    return np.convolve(y, h, "same")[::down].astype(np.float32)


def load_sherpa_embed(model: Path) -> Embed:
    """The CAM++ extractor as an `Embed`. ImportError without sherpa-onnx, FileNotFoundError
    without the model file."""
    if not model.is_file():
        raise FileNotFoundError(f"speaker model not found: {model}")
    import sherpa_onnx  # optional `speaker` extra

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1)
    )

    def embed(samples: np.ndarray, rate: int) -> np.ndarray:
        stream = extractor.create_stream()
        # resample here: sherpa's own resampler logs a line to stderr on every call
        x = to_16k(np.asarray(samples, dtype=np.float32), rate)
        stream.accept_waveform(16_000, np.ascontiguousarray(x))
        stream.input_finished()
        return np.array(extractor.compute(stream))

    return embed
