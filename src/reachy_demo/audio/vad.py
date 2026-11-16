"""Local voice activity: is the user talking right now? Drives playback ducking."""

from __future__ import annotations

import numpy as np

RATE = 24_000


class EnergyVAD:
    """Loud for `onset_ms` starts speech; quiet for more than `hangover_ms` ends it.

    shortcut: energy only, so noise can duck too; switch to Silero (VAD_THRESHOLD) if ducking
    fires on noise.
    """

    def __init__(
        self, threshold_rms: float = 500, onset_ms: int = 150, hangover_ms: int = 450
    ) -> None:
        self._threshold = threshold_rms
        self._onset_ms = onset_ms
        self._hangover_ms = hangover_ms
        self._loud_ms = 0.0
        self._quiet_ms = 0.0
        self.speaking = False

    def feed(self, chunk: bytes) -> bool:
        """24 kHz mono PCM16 chunk in; the speaking state after it out."""
        x = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype="<i2").astype(np.float64)
        if not len(x):
            return self.speaking
        ms = 1000 * len(x) / RATE
        if float(np.sqrt(np.mean(x**2))) >= self._threshold:
            self._loud_ms += ms
            self._quiet_ms = 0.0
            if self._loud_ms >= self._onset_ms:
                self.speaking = True
        else:
            self._quiet_ms += ms
            self._loud_ms = 0.0
            if self._quiet_ms > self._hangover_ms:
                self.speaking = False
        return self.speaking
