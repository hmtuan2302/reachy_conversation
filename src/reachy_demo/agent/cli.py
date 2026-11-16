"""`reachy-demo agent`: speak a question, hear the answer (S2T → DeepSeek → Gemini TTS)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import TextIO

from dotenv import find_dotenv, load_dotenv

from reachy_demo.agent.agent import Agent
from reachy_demo.agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError, chat_stream
from reachy_demo.audio.player import PlayerError
from reachy_demo.voice.speak_cli import speaker_from_env
from reachy_demo.voice.transcribe import transcribe_stream
from reachy_demo.voice.transcribe_cli import (
    add_source_args,
    recorded,
    run_until_done,
    save_recording,
    source_from_args,
)
from reachy_demo.voice.tts import TTSError


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("agent", help="speak a question, hear the LLM answer")
    add_source_args(p)
    p.add_argument("--no-speak", action="store_true", help="text answers only (no Gemini TTS)")


Speak = Callable[[str], Awaitable[object]]


async def converse(
    turns: AsyncIterator[str], agent: Agent, out: TextIO, speak: Speak | None = None
) -> None:
    """Answer each completed user turn in order; turns arriving meanwhile wait in a queue.

    With `speak`, each completed answer is then spoken (the next turn waits for it).
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def listen() -> None:
        try:
            async for turn in turns:
                await queue.put(turn)
        finally:
            await queue.put(None)  # end of input (or failure): let pending answers finish

    listener = asyncio.create_task(listen())
    try:
        while (text := await queue.get()) is not None:
            out.write(f"you: {text}\nbot: ")
            out.flush()
            parts: list[str] = []
            try:
                async for piece in agent.answer(text):
                    parts.append(piece)
                    out.write(piece)
                    out.flush()
            except LLMError as e:  # fail small: report it and keep listening
                out.write(f" (error: {e})\n")
                out.flush()
                continue
            out.write("\n")
            out.flush()
            if speak and (answer := "".join(parts).strip()):
                # shortcut: speaks the whole answer once it is complete; stream per sentence
                # if long answers feel slow to start.
                try:
                    if saved := await speak(answer):
                        out.write(f"  (saved {saved})\n")
                except (TTSError, PlayerError) as e:
                    out.write(f"  (speech error: {e})\n")
                out.flush()
        await listener  # re-raise a source/S2T failure after the answers already queued
    finally:
        listener.cancel()


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    needed = ["HF_TOKEN", "DEEPSEEK_API_KEY"] + ([] if args.no_speak else ["GEMINI_API_KEY"])
    keys = {name: os.environ.get(name, "").strip() for name in needed}
    for name, value in keys.items():
        if not value:
            print(f"config error: {name} is required", file=sys.stderr)
            return 2
    try:
        source = source_from_args(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    chat = partial(
        chat_stream,
        api_key=keys["DEEPSEEK_API_KEY"],
        model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL,
        base_url=os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL,
    )
    source, rec = recorded(args, source)
    turns = transcribe_stream(source, hf_token=keys["HF_TOKEN"])
    speak = None if args.no_speak else speaker_from_env(keys["GEMINI_API_KEY"])
    code = run_until_done(converse(turns, Agent(chat), sys.stdout, speak))
    save_recording(args, rec)
    return code
