"""`reachy-demo agent`: speak a question, hear the answer (S2T → DeepSeek → Gemini TTS).

With `--robot`, the agent can also move Reachy's head (fixed safe angles, one move at a time).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Literal, TextIO

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
from reachy_demo.agent.tools import Toolbox, ToolSet, camera, head
from reachy_demo.agent.tools import memory as memory_tools
from reachy_demo.agent.tools import sound as sound_tools
from reachy_demo.agent.tools import voice as voice_tools
from reachy_demo.audio.player import PlayerError
from reachy_demo.audio.sound_events import DEFAULT_LOUD_RMS, SoundEvents, with_sound
from reachy_demo.audio.vad import EnergyVAD
from reachy_demo.memory.briefing import DEFAULT_MEMORY_TOKENS
from reachy_demo.memory.consolidate import Consolidator
from reachy_demo.memory.embed import load_embedder
from reachy_demo.memory.session import MemorySession
from reachy_demo.memory.store import OWNER, MemoryStore
from reachy_demo.memory.vectors import VectorIndex
from reachy_demo.ops.tokens import est_tokens
from reachy_demo.perception.vision import ask_image_chat, ask_image_hf
from reachy_demo.robot.adapter import connect_robot
from reachy_demo.robot.dances import load_dances
from reachy_demo.robot.motion import Motion
from reachy_demo.voice.speak_cli import speaker_from_env, uses_gemini
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


async def _tap(
    chunks: AsyncIterator[bytes],
    vad: EnergyVAD,
    speech: voice_tools.SpeechBuffer | None = None,
    sound: SoundEvents | None = None,
) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        speaking = vad.feed(chunk)
        if speech:
            speech.feed(chunk, speaking)
        if sound:
            sound.feed(chunk, speaking, time.monotonic())
        yield chunk


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    vision = os.environ.get("VISION_PROVIDER", "").strip().lower() or "hf"
    if vision not in VISION_PROVIDERS:
        print(
            f"config error: VISION_PROVIDER must be one of {', '.join(VISION_PROVIDERS)}, "
            f"got {vision!r}",
            file=sys.stderr,
        )
        return 2
    context_max = _positive_int("CONTEXT_MAX_TOKENS", DEFAULT_CONTEXT_TOKENS)
    memory_max = _positive_int("MEMORY_MAX_TOKENS", DEFAULT_MEMORY_TOKENS)
    if context_max is None or memory_max is None:
        return 2
    if memory_max >= context_max:
        print(
            "config error: MEMORY_MAX_TOKENS must be below CONTEXT_MAX_TOKENS, "
            f"got {memory_max} and {context_max}",
            file=sys.stderr,
        )
        return 2
    # HF_TOKEN: speech-to-text, and the camera's vision model when it runs on the HF server.
    needs_hf = not args.type or (args.robot and vision == "hf")
    needed = (["HF_TOKEN"] if needs_hf else []) + ["DEEPSEEK_API_KEY"]
    needed += ["GEMINI_API_KEY"] if not args.no_speak and uses_gemini() else []
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
    # Voice cloning needs local TTS (Gemini can't clone). Typed input gives no sample, but the
    # tools stay so a clone from an earlier run can still be stopped.
    speech = None if args.no_speak or uses_gemini() else voice_tools.SpeechBuffer()
    sound = sound_from_env() if args.robot else None
    if sound is False:
        return 2
    if source is None:
        turns = typed_turns(sys.stdin)
    else:
        audio, rec = recorded(args, _tap(source, vad, speech, sound or None))
        turns = transcribe_stream(audio, hf_token=keys["HF_TOKEN"])
    speak = (
        None
        if args.no_speak
        else speaker_from_env(
            keys.get("GEMINI_API_KEY", ""),
            gain=lambda: DUCK if vad.speaking else 1.0,
            clone=voice_tools.CLONE_WAV,
            on_play=sound.mark_own_voice if sound else None,
        )
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
    memory = memory_from_env(keys["DEEPSEEK_API_KEY"], base_url, memory_max)
    push, push_line = push_status(memory)
    if memory:
        print(f"  (memory push: {push_line})")
    sets: list[ToolSet] = []
    prompt = SYSTEM_PROMPT
    if memory:
        session, tools_ = memory
        sets.append(tools_)
        prompt += f" {memory_tools.PROMPT}"
    if speech:
        # shortcut: everyone counts as the owner until speaker ID (M9) exists, same as memory
        sample = os.environ.get("CLONE_AUDIO", "").strip()
        sets.append(
            voice_tools.VoiceTools(
                speech, speaker=lambda: OWNER, sample=Path(sample) if sample else None
            )
        )
        prompt += f" {voice_tools.PROMPT}"

    def make_agent(sets: list[ToolSet], prompt: str) -> Agent | None:
        tools = Toolbox(*sets) if sets else None
        budget = context_max - (
            est_tokens(json.dumps(tools.specs(), ensure_ascii=False)) if tools else 0
        )
        if push and budget - PUSH_RESERVE_TOKENS <= memory_max:
            print(
                f"config error: related memories (MEMORY_PUSH) need ~{PUSH_RESERVE_TOKENS} more "
                f"tokens: raise CONTEXT_MAX_TOKENS={context_max} or set MEMORY_PUSH=off",
                file=sys.stderr,
            )
            return None
        if budget <= memory_max:
            print(
                f"config error: tool specs (~{context_max - budget} tokens) leave no room "
                f"in CONTEXT_MAX_TOKENS={context_max}",
                file=sys.stderr,
            )
            return None
        agent = Agent(
            partial(llm, model=model, tools=tools.specs()) if tools else partial(llm, model=model),
            system_prompt=prompt,
            tools=tools,
            context=memory[0].context if memory else None,
            on_turn=memory[0].submit if memory else None,
            max_tokens=budget,
            related=memory[0].related if push and memory else None,
            debug=show_context if _on("CONTEXT_DEBUG") else None,
        )
        if memory:
            agent.preload(memory[0].preload())
        return agent

    with contextlib.ExitStack() as stack:
        if args.robot:
            try:
                robot = stack.enter_context(connect_robot(keys["ROBOT_HOST"]))
            except Exception as e:  # SDK raises its own connection errors
                print(f"robot error: cannot connect to {keys['ROBOT_HOST']}: {e}", file=sys.stderr)
                return 1
            motion = Motion(robot.set_pose, on_error=lambda e: print(f"  (head error: {e})"))
            head_tools = head.HeadTools(motion, load_dances())
            prompt += f" {head.PROMPT}"
            sets.append(head_tools)
            assert sound  # made above for --robot
            sets.append(sound_tools.SoundTools(sound, motion))
            prompt += f" {sound_tools.PROMPT}"
            if robot.has_camera:
                ask, label = vision_asker(vision, keys, base_url)
                print(f"  (camera on, vision: {label})")
                sets.append(camera.CameraTools(robot.grab_jpeg, ask))
                prompt += f" {camera.PROMPT}"
            else:
                print("  (no camera: the robot can move but not see)")
            agent = make_agent(sets, prompt)
            if agent is None:
                return 2
            main = converse(
                turns,
                agent,
                sys.stdout,
                speak,
                stop,
                on_stop=head_tools.stop,
                moving=lambda: head_tools.busy,
            )
            main = with_sound(main, sound, robot.read_doa, lambda: motion.pose["yaw"])
            code = run_until_done(_with_memory(_then_close(main, head_tools), memory))
        else:
            agent = make_agent(sets, prompt)
            if agent is None:
                return 2
            main = converse(turns, agent, sys.stdout, speak, stop)
            code = run_until_done(_with_memory(main, memory))
    save_recording(args, rec)
    return code


VISION_PROVIDERS = ("hf", "deepseek")
DEFAULT_CONTEXT_TOKENS = 8000
PUSH_RESERVE_TOKENS = 200  # up to 3 related-memory lines of 240 chars, sent with the user turn


def sound_from_env() -> SoundEvents | Literal[False]:
    """The sound-direction store (SOUND_RMS), or False after printing a config error."""
    raw = os.environ.get("SOUND_RMS", "").strip()
    try:
        loud = float(raw) if raw else DEFAULT_LOUD_RMS
    except ValueError:
        loud = 0.0
    if loud <= 0:
        print(f"config error: SOUND_RMS must be a positive number, got {raw!r}", file=sys.stderr)
        return False
    return SoundEvents(loud)


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


def show_context(kind: str, text: str) -> None:
    """CONTEXT_DEBUG: print what the model is given, then reopen the "bot: " line."""
    body = "\n".join(f"  │ {line}" for line in text.splitlines())
    sys.stdout.write(f"\n  ┌ {kind}\n{body}\n  └\nbot: ")
    sys.stdout.flush()


def push_status(memory: tuple[MemorySession, object] | None) -> tuple[bool, str]:
    """Whether the per-turn related-memories push runs, and why not."""
    if os.environ.get("MEMORY_PUSH", "").strip().lower() in ("off", "0", "false", "no"):
        return False, "off: MEMORY_PUSH=off"
    if memory is None:
        return False, "off: memory is off"
    if not memory[0].has_vectors:
        return False, "off: no embedding model"
    return True, "on"


def _positive_int(name: str, default: int) -> int | None:
    """An env int > 0, or None after printing a config error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        print(f"config error: {name} must be a positive integer, got {raw!r}", file=sys.stderr)
        return None
    return value


def memory_from_env(
    api_key: str, base_url: str, memory_max_tokens: int = DEFAULT_MEMORY_TOKENS
) -> tuple[MemorySession, memory_tools.MemoryTools] | None:
    """Memory (MEMORY_DB, MEMORY_MODEL), or None when MEMORY=off or the DB can't be opened."""
    if os.environ.get("MEMORY", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    path = os.environ.get("MEMORY_DB", "").strip() or "data/memory.db"
    try:
        store = MemoryStore(path, time.time)
    except Exception as e:  # fail small: a broken DB turns memory off, not the agent
        print(f"  (memory off: cannot open {path}: {e})", file=sys.stderr)
        return None
    # Non-reasoning model by default: memory calls run in the background but must stay cheap.
    chat = partial(
        chat_stream,
        api_key=api_key,
        model=os.environ.get("MEMORY_MODEL", "").strip() or DEFAULT_MODEL,
        base_url=base_url,
    )
    print("  (memory search: loading the embedding model...)", flush=True)
    embedder, status = load_embedder(os.environ)
    index = VectorIndex.load(store, embedder.model) if embedder else None
    print(f"  (memory search: {status})")
    consolidator = Consolidator(store, chat, time.time, embedder=embedder, index=index)
    session = MemorySession(
        store,
        consolidator,
        chat,
        clock=time.time,
        memory_max_tokens=memory_max_tokens,
        embedder=embedder,
        index=index,
    )
    print(f"  (memory on: {path})")
    tools = memory_tools.MemoryTools(
        store, consolidator, speaker=lambda: OWNER, embedder=embedder, index=index
    )
    return session, tools


async def _with_memory(main: Awaitable[None], memory: tuple[MemorySession, object] | None) -> None:
    """Run the memory worker beside `main`; on exit, summarize the session and back up."""
    if memory is None:
        await main
        return
    session = memory[0]
    worker = asyncio.create_task(session.run())
    try:
        await main
    finally:
        await session.close(worker)


def vision_asker(provider: str, keys: dict[str, str], llm_base_url: str) -> tuple[camera.Ask, str]:
    """The camera's vision model picked by VISION_PROVIDER, and a label for the terminal."""
    if provider == "deepseek":
        # deepseek-chat sees images and doesn't reason first (fastest, 1.1 s in a test);
        # deepseek-v4-pro can't see images.
        model = os.environ.get("VISION_MODEL", "").strip() or "deepseek-chat"
        ask = partial(
            ask_image_chat,
            api_key=keys["DEEPSEEK_API_KEY"],
            model=model,
            base_url=os.environ.get("VISION_BASE_URL", "").strip() or llm_base_url,
        )
        return ask, f"deepseek {model}"
    return partial(ask_image_hf, hf_token=keys["HF_TOKEN"]), "hf realtime server"


async def _then_close(main: Awaitable[None], tools: head.HeadTools) -> None:
    """Run `main`, then let queued head moves finish before the robot goes neutral."""
    try:
        await main
    finally:
        await tools.close()
