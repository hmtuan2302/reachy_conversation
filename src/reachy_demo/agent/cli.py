"""`reachy-demo agent`: speak a question, hear the answer (S2T → DeepSeek → Gemini TTS)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import TextIO

from dotenv import find_dotenv, load_dotenv

from reachy_demo.agent.agent import Agent
from reachy_demo.agent.intent import Intent, classify
from reachy_demo.agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError, chat_stream
from reachy_demo.audio.player import PlayerError
from reachy_demo.audio.vad import EnergyVAD
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


DUCK = 0.25  # playback volume while the user talks over Reachy
Speak = Callable[[str], Awaitable[object]]
Classify = Callable[[str], Awaitable[Intent]]


async def converse(
    turns: AsyncIterator[str],
    agent: Agent,
    out: TextIO,
    speak: Speak | None = None,
    classify: Classify | None = None,
) -> None:
    """Answer each completed user turn in order.

    With `speak`, each completed answer is then spoken. Without `classify`, turns arriving
    during an answer wait in a queue. With `classify`, each one is classified instead: a stop
    cancels the answer (its `rest`, if any, is the next turn), anything else is dropped.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    line_open = False  # "bot: " printed, answer line not ended yet

    def write(text: str) -> None:
        out.write(text)
        out.flush()

    async def listen() -> None:
        try:
            async for turn in turns:
                await queue.put(turn)
        finally:
            await queue.put(None)  # end of input (or failure): let pending answers finish

    async def answer(text: str) -> None:
        nonlocal line_open
        write(f"you: {text}\nbot: ")
        line_open = True
        parts: list[str] = []
        try:
            async for piece in agent.answer(text):
                parts.append(piece)
                write(piece)
        except LLMError as e:  # fail small: report it and keep listening
            write(f" (error: {e})\n")
            line_open = False
            return
        write("\n")
        line_open = False
        if speak and (spoken := "".join(parts).strip()):
            # shortcut: speaks the whole answer once it is complete; stream per sentence
            # if long answers feel slow to start.
            try:
                if saved := await speak(spoken):
                    write(f"  (saved {saved})\n")
            except (TTSError, PlayerError) as e:
                write(f"  (speech error: {e})\n")

    async def answer_with_barge_in(text: str, classify: Classify) -> tuple[str | None, bool]:
        """Run one answer; returns (next turn from a stop's `rest`, input ended)."""
        nonlocal line_open
        task = asyncio.create_task(answer(text))
        ignored: list[str] = []
        ended = False
        try:
            while not task.done() and not ended:
                getter = asyncio.create_task(queue.get())
                await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if not getter.done():
                    getter.cancel()
                    break
                if (heard := getter.result()) is None:
                    ended = True
                    break
                intent = await classify(heard)
                if not intent.stop:
                    ignored.append(heard)
                    continue
                if not task.done():  # else it ended during classify: just take the rest
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task  # nothing of the old answer runs after this
                    write(("\n" if line_open else "") + "  (stopped)\n")
                    line_open = False
                return intent.rest or None, ended
            await task
        finally:
            task.cancel()
        for heard in ignored:
            write(f"  (ignored: {heard})\n")
        return None, ended

    listener = asyncio.create_task(listen())
    try:
        text = await queue.get()
        while text is not None:
            if classify is None:
                await answer(text)
                text = await queue.get()
                continue
            follow_up, ended = await answer_with_barge_in(text, classify)
            text = follow_up or (None if ended else await queue.get())
        await listener  # re-raise a source/S2T failure after the answers already queued
    finally:
        listener.cancel()


async def _tap(chunks: AsyncIterator[bytes], vad: EnergyVAD) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        vad.feed(chunk)
        yield chunk


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
    vad = EnergyVAD(
        float(os.environ.get("VAD_RMS", "").strip() or 500),
        hangover_ms=int(os.environ.get("VAD_HANGOVER_MS", "").strip() or 450),
    )
    source, rec = recorded(args, _tap(source, vad))
    turns = transcribe_stream(source, hf_token=keys["HF_TOKEN"])
    speak = (
        None
        if args.no_speak
        else speaker_from_env(keys["GEMINI_API_KEY"], gain=lambda: DUCK if vad.speaking else 1.0)
    )
    # The stop check always uses the fast non-reasoning model: reasoning models would hit its
    # 2 s timeout every time, and stop would never work.
    stop = partial(classify, chat=partial(chat, model=DEFAULT_MODEL))
    code = run_until_done(converse(turns, Agent(chat), sys.stdout, speak, stop))
    save_recording(args, rec)
    return code
