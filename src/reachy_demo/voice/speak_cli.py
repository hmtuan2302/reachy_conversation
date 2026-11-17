"""`reachy-demo speak TEXT`: text → speech (Gemini, streamed), played and saved as a WAV."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from reachy_demo.audio.player import DEFAULT_PLAYER, PlayerError, play_and_save, speech_path
from reachy_demo.voice.tts import DEFAULT_TTS_MODEL, DEFAULT_VOICE, TTSError, speak_stream
from reachy_demo.voice.vieneu_tts import vieneu_stream

SPEECH_DIR = Path("data/speech")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "speak", help="text → speech (Gemini or local VieNeu): play it and save a WAV"
    )
    p.add_argument("text", nargs="+", help="what to say")


def uses_gemini() -> bool:
    """TTS_PROVIDER: gemini (default, needs GEMINI_API_KEY) | vieneu (local model)."""
    return os.environ.get("TTS_PROVIDER", "").strip().lower() != "vieneu"


def speaker_from_env(
    api_key: str,
    gain: Callable[[], float] | None = None,
    clone: Path | None = None,
    on_play: Callable[[float], None] | None = None,
) -> Callable[[str], Coroutine[Any, Any, Path | None]]:
    """speak(text) -> saved WAV path, or None with SAVE_SPEECH=off. Raises TTSError / PlayerError.

    With `clone` and TTS_PROVIDER=vieneu, it speaks in the voice of that WAV while the file exists.
    """
    gemini = uses_gemini()
    model = os.environ.get("TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL
    voice = os.environ.get("TTS_VOICE", "").strip() or (DEFAULT_VOICE if gemini else "")
    player = os.environ.get("TTS_PLAYER", "").strip() or DEFAULT_PLAYER
    player_cmd = None if player.lower() == "none" else player
    save = os.environ.get("SAVE_SPEECH", "").strip().lower() not in ("off", "0", "false", "no")
    counter = itertools.count(1)

    async def speak(text: str) -> Path | None:
        cloned = clone is not None and clone.exists()  # noqa: ASYNC240 (one stat, microseconds)
        chunks = (
            speak_stream(text, api_key=api_key, model=model, voice=voice)
            if gemini
            else vieneu_stream(text, voice=voice, ref_audio=clone if cloned else None)
        )
        path = speech_path(SPEECH_DIR, next(counter)) if save else None
        return await play_and_save(
            chunks, path=path, player_cmd=player_cmd, gain=gain, on_play=on_play
        )

    return speak


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and uses_gemini():
        print("config error: GEMINI_API_KEY is required", file=sys.stderr)
        return 2
    # CLONE_AUDIO=<wav>: test cloning with that file's voice.
    clone = Path(fixed) if (fixed := os.environ.get("CLONE_AUDIO", "").strip()) else None
    if clone and not clone.exists():
        print(f"config error: CLONE_AUDIO file not found: {clone}", file=sys.stderr)
        return 2
    speak = speaker_from_env(key, clone=clone)
    try:
        path = asyncio.run(speak(" ".join(args.text)))
    except KeyboardInterrupt:
        return 0
    except (TTSError, PlayerError) as e:
        print(f"speech error: {e}", file=sys.stderr)
        return 1
    if path:
        print(f"saved {path}")
    return 0
