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

SPEECH_DIR = Path("data/speech")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("speak", help="text → speech (Gemini): play it and save a WAV")
    p.add_argument("text", nargs="+", help="what to say")


def speaker_from_env(
    api_key: str, gain: Callable[[], float] | None = None
) -> Callable[[str], Coroutine[Any, Any, Path]]:
    """speak(text) -> saved WAV path. Raises TTSError / PlayerError (audio saved for the latter)."""
    model = os.environ.get("TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL
    voice = os.environ.get("TTS_VOICE", "").strip() or DEFAULT_VOICE
    player = os.environ.get("TTS_PLAYER", "").strip() or DEFAULT_PLAYER
    player_cmd = None if player.lower() == "none" else player
    counter = itertools.count(1)

    async def speak(text: str) -> Path:
        chunks = speak_stream(text, api_key=api_key, model=model, voice=voice)
        path = speech_path(SPEECH_DIR, next(counter))
        return await play_and_save(chunks, path=path, player_cmd=player_cmd, gain=gain)

    return speak


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("config error: GEMINI_API_KEY is required", file=sys.stderr)
        return 2
    speak = speaker_from_env(key)
    try:
        path = asyncio.run(speak(" ".join(args.text)))
    except KeyboardInterrupt:
        return 0
    except (TTSError, PlayerError) as e:
        print(f"speech error: {e}", file=sys.stderr)
        return 1
    print(f"saved {path}")
    return 0
