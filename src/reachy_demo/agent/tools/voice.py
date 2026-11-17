"""Voice cloning tools: clone_voice and stop_cloning_voice (owner only, enforced here).

The mic keeps the last few seconds of the user's speech (`SpeechBuffer`). clone_voice saves it
as the reference WAV, or copies a fixed test WAV (CLONE_AUDIO) instead. While that file exists,
local VieNeu TTS speaks with that voice. The file outlives restarts, so cloning stays on until
stop_cloning_voice deletes it.
"""

from __future__ import annotations

import shutil
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reachy_demo.audio.sources import RATE, save_wav
from reachy_demo.memory.store import OWNER, Speaker

CLONE_WAV = Path("data/voice/clone.wav")
KEEP_S = 10.0  # VieNeu trims the reference to 8 s anyway
MIN_S = 3.0  # VieNeu wants a 3-8 s clip
PROMPT = (
    "You can speak in the user's own voice. Call clone_voice when they ask you to clone, copy "
    "or mimic their voice. If it says need_more_speech, ask them to talk for a few seconds, "
    "then try again. Call stop_cloning_voice when they ask you to stop using their voice."
)
NOT_OWNER = {"error": "not_owner: only the owner can change your voice"}


class SpeechBuffer:
    """The last KEEP_S seconds of 24 kHz PCM16 mic audio heard while the user was talking."""

    def __init__(self, keep_s: float = KEEP_S) -> None:
        self._chunks: deque[bytes] = deque()
        self._bytes = 0
        self._max = int(keep_s * RATE) * 2

    def feed(self, chunk: bytes, speaking: bool) -> None:
        if not speaking:
            return
        self._chunks.append(chunk)
        self._bytes += len(chunk)
        while self._bytes - len(self._chunks[0]) >= self._max:
            self._bytes -= len(self._chunks.popleft())

    def seconds(self) -> float:
        return self._bytes / 2 / RATE

    def pcm(self) -> bytes:
        return b"".join(self._chunks)


def _fn(name: str, description: str) -> dict[str, Any]:
    params = {"type": "object", "properties": {}, "required": []}
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


SPECS = [
    _fn("clone_voice", "Start speaking in the user's voice, from what they just said."),
    _fn("stop_cloning_voice", "Go back to your normal voice."),
]


class VoiceTools:
    def __init__(
        self,
        speech: SpeechBuffer,
        speaker: Callable[[], Speaker],
        path: Path = CLONE_WAV,
        sample: Path | None = None,
    ) -> None:
        """`sample`: a fixed WAV to clone instead of the mic (for testing)."""
        self._speech = speech
        self._sample = sample
        self._speaker = speaker
        self._path = path

    def specs(self) -> list[dict[str, Any]]:
        return SPECS

    def needs_response(self, name: str, arguments: str) -> bool:
        return True

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        if self._speaker() != OWNER:
            return NOT_OWNER
        if name == "stop_cloning_voice":
            self._path.unlink(missing_ok=True)
            return {"ok": True}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._sample:
            try:
                shutil.copyfile(self._sample, self._path)
            except OSError as e:
                return {"error": f"cannot read CLONE_AUDIO {self._sample}: {e}"}
            return {"ok": True, "source": "test recording (CLONE_AUDIO)"}
        heard = self._speech.seconds()
        if heard < MIN_S:
            return {"error": f"need_more_speech: heard {heard:.1f} s, need {MIN_S:.0f} s"}
        save_wav(self._path, self._speech.pcm())
        return {"ok": True, "seconds": round(heard, 1)}
