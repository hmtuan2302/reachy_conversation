"""`reachy-demo agent`: speak a question, hear the answer (S2T → DeepSeek → Gemini TTS).

With `--robot`, the agent can also move Reachy's head (fixed safe angles, one move at a time).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import TextIO

from dotenv import find_dotenv, load_dotenv

from reachy_demo.agent.agent import SYSTEM_PROMPT, Agent
from reachy_demo.agent.intent import Intent, classify
from reachy_demo.agent.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMError,
    chat_events,
    chat_stream,
)
from reachy_demo.agent.tools import head
from reachy_demo.audio.player import PlayerError
from reachy_demo.audio.vad import EnergyVAD
from reachy_demo.robot.adapter import connect_head
from reachy_demo.robot.dances import load_dances
from reachy_demo.robot.motion import Motion
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
    add_source_args(p, typing=True)
    p.add_argument("--no-speak", action="store_true", help="text answers only (no Gemini TTS)")
    p.add_argument(
        "--robot", action="store_true", help="let the agent move the head (robot at ROBOT_HOST)"
    )


DUCK = 0.25  # playback volume while the user talks over Reachy
Speak = Callable[[str], Awaitable[object]]
Classify = Callable[[str], Awaitable[Intent]]


async def converse(
    turns: AsyncIterator[str],
    agent: Agent,
    out: TextIO,
    speak: Speak | None = None,
    classify: Classify | None = None,
    on_stop: Callable[[], None] | None = None,
    moving: Callable[[], bool] | None = None,
) -> None:
    """Answer each completed user turn in order.

    With `speak`, each completed answer is then spoken. Without `classify`, turns arriving
    during an answer wait in a queue. With `classify`, each one is classified instead: a stop
    cancels the answer (its `rest`, if any, is the next turn) and calls `on_stop` (e.g. to drop
    queued head moves); anything else is dropped. With `moving` and `classify`, a turn that
    arrives while no answer runs but the robot still moves is stop-checked first, too.
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
                if on_stop:
                    on_stop()
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
            if moving and moving():
                intent = await classify(text)
                if intent.stop:
                    if on_stop:
                        on_stop()
                    write(f"you: {text}\n  (stopped)\n")
                    text = intent.rest or await queue.get()
                    continue
            follow_up, ended = await answer_with_barge_in(text, classify)
            text = follow_up or (None if ended else await queue.get())
        await listener  # re-raise a source/S2T failure after the answers already queued
    finally:
        listener.cancel()


async def typed_turns(lines: TextIO) -> AsyncIterator[str]:
    """Each non-empty typed line is one turn; Ctrl-D ends the input.

    A daemon thread reads the lines, so Ctrl-C exits at once even while it waits for typing
    (a blocked asyncio.to_thread read would hold the exit until Enter).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def put(line: str | None) -> None:
        with contextlib.suppress(RuntimeError):  # the loop already closed on exit
            loop.call_soon_threadsafe(queue.put_nowait, line)

    def read() -> None:
        for line in lines:
            put(line)
        put(None)

    threading.Thread(target=read, daemon=True).start()
    while (line := await queue.get()) is not None:
        if text := line.strip():
            yield text


async def _tap(chunks: AsyncIterator[bytes], vad: EnergyVAD) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        vad.feed(chunk)
        yield chunk


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    needed = ([] if args.type else ["HF_TOKEN"]) + ["DEEPSEEK_API_KEY"]
    needed += [] if args.no_speak else ["GEMINI_API_KEY"]
    needed += ["ROBOT_HOST"] if args.robot else []
    keys = {name: os.environ.get(name, "").strip() for name in needed}
    for name, value in keys.items():
        if not value:
            print(f"config error: {name} is required", file=sys.stderr)
            return 2
    try:
        source = None if args.type else source_from_args(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    base_url = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
    llm = partial(chat_events, api_key=keys["DEEPSEEK_API_KEY"], base_url=base_url)
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL
    vad = EnergyVAD(
        float(os.environ.get("VAD_RMS", "").strip() or 500),
        hangover_ms=int(os.environ.get("VAD_HANGOVER_MS", "").strip() or 450),
    )
    rec = None
    if source is None:
        turns = typed_turns(sys.stdin)
    else:
        audio, rec = recorded(args, _tap(source, vad))
        turns = transcribe_stream(audio, hf_token=keys["HF_TOKEN"])
    speak = (
        None
        if args.no_speak
        else speaker_from_env(keys["GEMINI_API_KEY"], gain=lambda: DUCK if vad.speaking else 1.0)
    )
    # The stop check always uses the fast non-reasoning model: reasoning models would hit its
    # 2 s timeout every time, and stop would never work.
    stop = partial(
        classify,
        chat=partial(
            chat_stream,
            api_key=keys["DEEPSEEK_API_KEY"],
            model=DEFAULT_MODEL,
            base_url=base_url,
        ),
    )
    with contextlib.ExitStack() as stack:
        if args.robot:
            try:
                set_pose = stack.enter_context(connect_head(keys["ROBOT_HOST"]))
            except Exception as e:  # SDK raises its own connection errors
                print(f"robot error: cannot connect to {keys['ROBOT_HOST']}: {e}", file=sys.stderr)
                return 1
            motion = Motion(set_pose, on_error=lambda e: print(f"  (head error: {e})"))
            tools = head.HeadTools(motion, load_dances())
            agent = Agent(
                partial(llm, model=model, tools=tools.specs()),
                system_prompt=f"{SYSTEM_PROMPT} {head.PROMPT}",
                tools=tools,
            )
            main = converse(
                turns, agent, sys.stdout, speak, stop, on_stop=tools.stop, moving=lambda: tools.busy
            )
            code = run_until_done(_then_close(main, tools))
        else:
            agent = Agent(partial(llm, model=model))
            code = run_until_done(converse(turns, agent, sys.stdout, speak, stop))
    save_recording(args, rec)
    return code


async def _then_close(main: Awaitable[None], tools: head.HeadTools) -> None:
    """Run `main`, then let queued head moves finish before the robot goes neutral."""
    try:
        await main
    finally:
        await tools.close()
