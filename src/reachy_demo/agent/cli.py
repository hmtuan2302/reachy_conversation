"""`reachy-demo agent`: speak a question, get a DeepSeek answer (S2T → LLM; T2S comes later)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from functools import partial
from typing import TextIO

from dotenv import find_dotenv, load_dotenv

from reachy_demo.agent.agent import Agent
from reachy_demo.agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError, chat_stream
from reachy_demo.voice.transcribe import transcribe_stream
from reachy_demo.voice.transcribe_cli import add_source_args, run_until_done, source_from_args


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("agent", help="speak a question, get an LLM answer (text)")
    add_source_args(p)


async def converse(turns: AsyncIterator[str], agent: Agent, out: TextIO) -> None:
    """Answer each completed user turn in order; turns arriving meanwhile wait in a queue."""
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
            try:
                async for piece in agent.answer(text):
                    out.write(piece)
                    out.flush()
            except LLMError as e:  # fail small: report it and keep listening
                out.write(f" (error: {e})")
            out.write("\n")
            out.flush()
        await listener  # re-raise a source/S2T failure after the answers already queued
    finally:
        listener.cancel()


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    keys = {name: os.environ.get(name, "").strip() for name in ("HF_TOKEN", "DEEPSEEK_API_KEY")}
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
    turns = transcribe_stream(source, hf_token=keys["HF_TOKEN"])
    return run_until_done(converse(turns, Agent(chat), sys.stdout))
