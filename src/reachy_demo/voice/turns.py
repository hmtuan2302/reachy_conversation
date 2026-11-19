"""When has the user finished speaking? Turns VAD frames into whole-turn audio (16 kHz).

A pause of `pause_ms` asks Smart Turn (`is_complete`) once. A score at or above `turn_threshold`
ends the turn; otherwise the turn ends after `max_silence_ms` of silence, or continues if the user
speaks again.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

RATE16 = 16_000
FRAME = 512  # Silero frame: 32 ms at 16 kHz
SMART_TURN_SAMPLES = 8 * RATE16


@dataclass(frozen=True)
class TurnSettings:
    vad_threshold: float = 0.5
    pause_ms: int = 200
    turn_threshold: float = 0.5
    max_silence_ms: int = 2000
    lead_ms: int = 300
    min_speech_ms: int = 250

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> TurnSettings:
        def get(name: str, default: float) -> float:
            return float(env.get(name, "").strip() or default)

        d = cls()
        return cls(
            vad_threshold=get("VAD_THRESHOLD", d.vad_threshold),
            pause_ms=int(get("STT_PAUSE_MS", d.pause_ms)),
            turn_threshold=get("STT_TURN_THRESHOLD", d.turn_threshold),
            max_silence_ms=int(get("STT_MAX_SILENCE_MS", d.max_silence_ms)),
        )


class Resampler24to16:
    """int16 PCM at 24 kHz in, float32 in [-1, 1] at 16 kHz out. Carries leftovers between calls.

    shortcut: linear 3:2 resample without low-pass filter; switch to a polyphase filter if Whisper
    accuracy suffers on sibilants.
    """

    def __init__(self) -> None:
        self._bytes = b""
        self._samples = np.zeros(0, np.float32)

    def feed(self, pcm24k: bytes) -> np.ndarray:
        data = self._bytes + pcm24k
        cut = len(data) - len(data) % 2
        self._bytes = data[cut:]
        x = np.concatenate([self._samples, np.frombuffer(data[:cut], np.int16) / 32768.0])
        n = len(x) - len(x) % 3
        self._samples = x[n:].astype(np.float32)
        t = x[:n].reshape(-1, 3)
        out = np.empty((len(t), 2), np.float32)
        out[:, 0] = t[:, 0]
        out[:, 1] = (t[:, 1] + t[:, 2]) / 2
        return out.reshape(-1)


class TurnDetector:
    def __init__(self, settings: TurnSettings, is_complete: Callable[[np.ndarray], float]) -> None:
        self._s = settings
        self._is_complete = is_complete
        self._lead: deque[np.ndarray] = deque(maxlen=-(-settings.lead_ms * RATE16 // 1000 // FRAME))
        self._turn: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._asked = False
        self.last_score: float | None = None  # Smart Turn's score for the last turn, for logging

    def feed(self, frame: np.ndarray, prob: float) -> np.ndarray | None:
        ms = len(frame) * 1000 / RATE16
        speaking = prob >= self._s.vad_threshold
        if not self._turn:
            if not speaking:
                self._lead.append(frame)
                return None
            self._turn = [*self._lead, frame]
            self._lead.clear()
            self._speech_ms, self._silence_ms, self._asked = ms, 0.0, False
            self.last_score = None
            return None
        self._turn.append(frame)
        if speaking:
            self._speech_ms += ms
            self._silence_ms, self._asked = 0.0, False
            return None
        self._silence_ms += ms
        if self._silence_ms < self._s.pause_ms:
            return None
        if self._speech_ms < self._s.min_speech_ms:
            self._reset()  # a cough or click, not a turn
            return None
        if not self._asked:
            self._asked = True
            self.last_score = self._is_complete(self._tail())
            if self.last_score >= self._s.turn_threshold:
                return self._end()
        return self._end() if self._silence_ms >= self._s.max_silence_ms else None

    def flush(self) -> np.ndarray | None:
        """The open turn when the audio source ends, if it holds enough speech."""
        if self._turn and self._speech_ms >= self._s.min_speech_ms:
            return self._end()
        self._reset()
        return None

    def _tail(self) -> np.ndarray:
        audio = np.concatenate(self._turn)[-SMART_TURN_SAMPLES:]
        return np.pad(audio, (SMART_TURN_SAMPLES - len(audio), 0))

    def _end(self) -> np.ndarray:
        audio = np.concatenate(self._turn)
        self._reset()
        return audio

    def _reset(self) -> None:
        self._turn = []
        self._speech_ms = self._silence_ms = 0.0
        self._asked = False
