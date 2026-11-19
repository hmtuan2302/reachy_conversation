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
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from functools import partial
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
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
from reachy_demo.agent.tools import Toolbox, ToolSet, camera, head, track
from reachy_demo.agent.tools import find as find_tools
from reachy_demo.agent.tools import memory as memory_tools
from reachy_demo.agent.tools import mimic as mimic_tools
from reachy_demo.agent.tools import owner as owner_tools
from reachy_demo.agent.tools import sound as sound_tools
from reachy_demo.agent.tools import voice as voice_tools
from reachy_demo.attention.follow import Follow
from reachy_demo.attention.mimic import Mimic
from reachy_demo.attention.search import Search
from reachy_demo.audio.player import PlayerError
from reachy_demo.audio.sound_events import DEFAULT_LOUD_RMS, SoundEvents, with_sound
from reachy_demo.audio.vad import EnergyVAD
from reachy_demo.memory.briefing import DEFAULT_MEMORY_TOKENS
from reachy_demo.memory.consolidate import Consolidator
from reachy_demo.memory.embed import load_embedder
from reachy_demo.memory.session import MemorySession
from reachy_demo.memory.store import OWNER, MemoryStore, Speaker
from reachy_demo.memory.vectors import VectorIndex
from reachy_demo.ops.logging import configure_logging
from reachy_demo.ops.tokens import est_tokens
from reachy_demo.perception.debug_view import DebugView
from reachy_demo.perception.head_pose import DEFAULT_MODEL as MIMIC_MODEL
from reachy_demo.perception.head_pose import HeadPoseEstimator
from reachy_demo.perception.objects import (
    CAMERA_LAG_S,
    BoxesDetector,
    ObjectDetector,
    ObjectSource,
    model_path,
)
from reachy_demo.perception.vision import ask_image_chat, ask_image_hf
from reachy_demo.robot.adapter import connect_robot, face_detector, scored_face_detector
from reachy_demo.robot.dances import load_dances
from reachy_demo.robot.motion import Motion
from reachy_demo.voice.local_stt import SttError, models_from_env, transcribe_local
from reachy_demo.voice.speak_cli import speaker_from_env, uses_gemini
from reachy_demo.voice.speaker_id import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    VOICEPRINT,
    SpeakerGate,
    load_sherpa_embed,
)
from reachy_demo.voice.speaker_id import DEFAULT_MODEL as SPEAKER_MODEL
from reachy_demo.voice.transcribe import transcribe_stream
from reachy_demo.voice.transcribe_cli import (
    RECORDINGS_DIR,
    add_source_args,
    local_notice,
    recorded,
    run_until_done,
    save_recording,
    source_from_args,
    stt_provider,
)
from reachy_demo.voice.tts import TTSError


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("agent", help="speak a question, hear the LLM answer")
    add_source_args(p, typing=True)
    p.add_argument("--no-speak", action="store_true", help="text answers only (no Gemini TTS)")
    p.add_argument(
        "--robot", action="store_true", help="let the agent move the head (robot at ROBOT_HOST)"
    )
    p.add_argument(
        "--debug-view",
        type=int,
        nargs="?",
        const=8090,
        metavar="PORT",
        help="with --robot: camera page with face boxes at http://127.0.0.1:PORT/ (default 8090)",
    )


FACE_MIN_SCORE = 0.85  # debug view: weaker face boxes are dropped (FACE_MIN_SCORE)
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
    on_answer: Callable[[str], None] | None = None,
    on_line: Callable[[str, str], None] | None = None,
    greeting: str | None = None,
) -> None:
    """Answer each completed user turn in order.

    With `speak`, each completed answer is then spoken. Without `classify`, turns arriving
    during an answer wait in a queue. With `classify`, each one is classified instead: a stop
    cancels the answer (its `rest`, if any, is the next turn) and calls `on_stop` (e.g. to drop
    queued head moves); anything else is dropped. With `moving` and `classify`, a turn that
    arrives while no answer runs but the robot still moves is stop-checked first, too.
    `on_answer(text)` runs as each answer starts (owner mode: whose turn the tools serve).
    `on_line(who, text)` gets each turn ("you") and each whole answer ("bot"), without the
    context or status lines (e.g. for the debug view's chat panel).
    `greeting` is said first, once listening runs, like an answer (barge-in applies) but
    without a user turn and without entering the agent's history.
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

    async def answer(text: str, fixed: str | None = None) -> None:
        """Answer `text`, or say `fixed` (the greeting) with no user turn and no LLM call."""
        nonlocal line_open
        if fixed is None:
            if on_answer:
                on_answer(text)
            write(f"you: {text}\n")
            if on_line:
                on_line("you", str(text))
        write("bot: ")
        line_open = True
        parts: list[str] = []
        pieces = _once(fixed) if fixed is not None else agent.answer(text)
        try:
            async for piece in pieces:
                parts.append(piece)
                write(piece)
        except LLMError as e:  # fail small: report it and keep listening
            write(f" (error: {e})\n")
            line_open = False
            if on_line:
                on_line("bot", f"{''.join(parts)} (error: {e})")
            return
        except asyncio.CancelledError:  # a spoken stop
            if on_line:
                on_line("bot", f"{''.join(parts)} (stopped)")
            raise
        write("\n")
        line_open = False
        if on_line:
            on_line("bot", "".join(parts))
        if speak and (spoken := "".join(parts).strip()):
            # shortcut: speaks the whole answer once it is complete; stream per sentence
            # if long answers feel slow to start.
            try:
                if saved := await speak(spoken):
                    write(f"  (saved {saved})\n")
            except (TTSError, PlayerError) as e:
                write(f"  (speech error: {e})\n")

    async def answer_with_barge_in(
        job: Coroutine[Any, Any, None], classify: Classify
    ) -> tuple[str | None, bool]:
        """Run one answer; returns (next turn from a stop's `rest`, input ended)."""
        nonlocal line_open
        task = asyncio.create_task(job)
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
                return _rest_of(heard, intent.rest) or None, ended
            await task
        finally:
            task.cancel()
        for heard in ignored:
            write(f"  (ignored: {heard})\n")
        return None, ended

    listener = asyncio.create_task(listen())
    try:
        text: str | None = None
        if greeting and classify:
            text, ended = await answer_with_barge_in(answer("", greeting), classify)
            if ended:
                await listener
                return
        elif greeting:
            await answer("", greeting)
        text = text or await queue.get()
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
                    text = _rest_of(text, intent.rest) or await queue.get()
                    continue
            follow_up, ended = await answer_with_barge_in(answer(text), classify)
            text = follow_up or (None if ended else await queue.get())
        await listener  # re-raise a source/S2T failure after the answers already queued
    finally:
        listener.cancel()


async def _once(text: str) -> AsyncIterator[str]:
    yield text


def robot_status(activities: dict[str, Callable[[], bool]]) -> str:
    """The live state of long-running moves, so the model never takes it from old chat or memory
    (every one of them ends when the app exits)."""
    doing = [name for name, active in activities.items() if active()]
    now = ", ".join(doing) if doing else "not following or mimicking anyone or anything"
    return (
        f"[robot] Right now you are {now}. This line is the truth: anything earlier in the chat "
        "or in memory about tracking, following or mimicking may be out of date."
    )


def _joined(*parts: Callable[[], str] | None) -> Callable[[], str] | None:
    """One context callable from several (memory block, robot state); None if all are None."""
    given = [p for p in parts if p]
    if not given:
        return None
    return lambda: "\n\n".join(text for p in given if (text := p()))


def _rest_of(turn: str, rest: str) -> str:
    """A stop's leftover request, still tagged with the voice that said it."""
    return owner_tools.Heard(rest, turn.verdict) if isinstance(turn, owner_tools.Heard) else rest


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
    gate: SpeakerGate | None = None,
) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        speaking = vad.feed(chunk)
        if speech:
            speech.feed(chunk, speaking)
        if gate:  # Reachy's own voice through the mic is not the user's
            gate.feed(chunk, speaking and not gate.own_voice(time.monotonic()))
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
    if (stt := stt_provider()) is None:
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
    needs_hf = (not args.type and stt == "hf") or (args.robot and vision == "hf")
    needed = (["HF_TOKEN"] if needs_hf else []) + ["DEEPSEEK_API_KEY"]
    needed += ["GEMINI_API_KEY"] if not args.no_speak and uses_gemini() else []
    needed += ["ROBOT_HOST"] if args.robot else []
    keys = {name: os.environ.get(name, "").strip() for name in needed}
    for name, value in keys.items():
        if not value:
            print(f"config error: {name} is required", file=sys.stderr)
            return 2
    log_path = configure_logging(
        Path(os.environ.get("LOG_DIR", "").strip() or "data/logs"),
        os.environ.get("LOG_LEVEL", "").strip().upper() or "INFO",
        secrets=keys.values(),
        console_level="WARNING",  # the terminal is the chat; details go to the file
    )
    print(f"  (log: {log_path})", file=sys.stderr)
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
    # Typed turns have no voice to check: they count as the owner.
    gate = None if source is None else speaker_gate_from_env()
    mode = owner_tools.OwnerMode(gate)
    if source is None:
        turns = typed_turns(sys.stdin)
    else:
        audio, rec = recorded(args, _tap(source, vad, speech, sound or None, gate))
        stt_models = None
        if stt == "local":
            local_notice()
            try:  # loaded now, so the greeting only comes once everything is ready
                stt_models = models_from_env(os.environ)
            except SttError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        turns = (
            transcribe_local(audio, models=stt_models)
            if stt == "local"
            else transcribe_stream(audio, hf_token=keys["HF_TOKEN"])
        )
    speak = (
        None
        if args.no_speak
        else speaker_from_env(
            keys.get("GEMINI_API_KEY", ""),
            gain=lambda: DUCK if vad.speaking else 1.0,
            clone=voice_tools.CLONE_WAV,
            on_play=partial(_mark_own_voice, [m for m in (sound, gate) if m]),
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

    async def is_stop(text: str) -> bool:
        return (await stop(text)).stop

    turns = owner_tools.gated_turns(turns, mode, sys.stdout, say=speak, is_stop=is_stop)
    memory = memory_from_env(keys["DEEPSEEK_API_KEY"], base_url, memory_max, mode.speaker)
    push, push_line = push_status(memory)
    if memory:
        print(f"  (memory push: {push_line})")
    sets: list[ToolSet] = []
    prompt = SYSTEM_PROMPT
    if source is not None:  # owner mode needs a voice; typed turns are always the owner
        sets.append(owner_tools.OwnerTools(mode))
        prompt += f" {owner_tools.PROMPT}"
    if memory:
        session, tools_ = memory
        sets.append(tools_)
        prompt += f" {memory_tools.PROMPT}"
    if speech:
        sample = os.environ.get("CLONE_AUDIO", "").strip()
        sets.append(
            voice_tools.VoiceTools(
                speech, speaker=mode.speaker, sample=Path(sample) if sample else None
            )
        )
        prompt += f" {voice_tools.PROMPT}"

    def make_agent(
        sets: list[ToolSet], prompt: str, status: Callable[[], str] | None = None
    ) -> Agent | None:
        """`status()`: live robot state appended to the memory block each round."""
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
            context=_joined(memory[0].context if memory else None, status),
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
            head_tools = head.HeadTools(motion, load_dances(), speaker=mode.speaker)
            prompt += f" {head.PROMPT}"
            sets.append(head_tools)
            assert sound  # made above for --robot
            sets.append(sound_tools.SoundTools(sound, motion, speaker=mode.speaker))
            prompt += f" {sound_tools.PROMPT}"
            lag_s = camera_lag_from_env()
            if lag_s is None:
                return 2
            # Faces go through the same pipeline as objects: YuNet on the laptop, then
            # nearest box + smoothing + CSRT on misses, steered from here with lag correction.
            faces = ObjectSource(
                robot.grab_frame,
                lambda: BoxesDetector("face", scored_face_detector(FACE_TRACK_SCORE)),
                lag_s=lag_s,
            )

            def face_switch(on: bool) -> bool:
                if not on:
                    return faces.switch(False)
                return robot.has_camera and faces.begin("face")

            follow = Follow(motion, faces.read, face_switch, sound)
            activities: dict[str, Callable[[], bool]] = {
                "following the user": lambda: follow.active
            }
            view: DebugView | None = None
            objects: tuple[Follow, ObjectSource] | None = None
            prompt += f" {track.PROMPT}"
            if robot.has_camera:
                ask, label = vision_asker(vision, keys, base_url)
                object_model = model_path(os.environ.get("OBJECT_MODEL", ""))
                seen = ObjectSource(
                    robot.grab_frame, partial(ObjectDetector, object_model), ask, lag_s=lag_s
                )
                objects = (
                    Follow(
                        motion,
                        seen.read,
                        seen.switch,
                        poll_s=OBJECT_POLL_S,
                        give_up_s=OBJECT_GIVE_UP_S,
                        first_wait_s=OBJECT_WAIT_S,
                        on_lost=lambda: print("  (lost the object, stopped tracking)"),
                        name="the object",
                    ),
                    seen,
                )
                prompt += f" {track.OBJECT_PROMPT}"
                activities["following an object"] = lambda: (
                    objects is not None and objects[0].active
                )
                look = Search(
                    motion,
                    robot.grab_frame,
                    partial(ObjectDetector, object_model),
                    ask,
                    lag_s=lag_s,
                )
                sets.append(find_tools.FindTools(look, speaker=mode.speaker))
                prompt += f" {find_tools.PROMPT}"
                face_model = Path(os.environ.get("MIMIC_MODEL", "").strip() or MIMIC_MODEL)
                copy = Mimic(
                    motion,
                    robot.grab_frame,
                    lambda: HeadPoseEstimator(face_detector(), face_model),
                )
                sets.append(mimic_tools.MimicTools(copy, speaker=mode.speaker))
                activities["mimicking the user"] = lambda: copy.active
                prompt += f" {mimic_tools.PROMPT}"
                print(f"  (camera on, vision: {label})")
                if args.debug_view is not None:
                    min_score = face_score_from_env()
                    if min_score is None:
                        return 2
                    view = DebugView(
                        robot.grab_frame,
                        partial(scored_face_detector, min_score),
                        lambda: None,  # the robot's face tracker is no longer used
                        record_dir=RECORDINGS_DIR,
                        read_object=lambda: faces.mark() or seen.mark() or look.mark(),
                    )
                    print(f"  (debug view: {stack.enter_context(view.serve(args.debug_view))})")
                sets.append(camera.CameraTools(robot.grab_jpeg, ask))
                prompt += f" {camera.PROMPT}"
            else:
                print("  (no camera: the robot can move but not see)")
            sets.append(track.TrackTools(follow, speaker=mode.speaker, objects=objects))
            agent = make_agent(sets, prompt, status=partial(robot_status, activities))
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
                on_answer=mode.begin,
                on_line=view.add_line if view else None,
                greeting=GREETING,
            )
            main = with_sound(main, sound, robot.read_doa, lambda: motion.pose["yaw"])
            code = run_until_done(_with_memory(_then_close(main, head_tools), memory))
        else:
            agent = make_agent(sets, prompt)
            if agent is None:
                return 2
            main = converse(
                turns, agent, sys.stdout, speak, stop, on_answer=mode.begin, greeting=GREETING
            )
            code = run_until_done(_with_memory(main, memory))
    save_recording(args, rec)
    return code


VISION_PROVIDERS = ("hf", "deepseek")
GREETING = "Hi, I'm Reachy Mini, nice to meet you."
OBJECT_POLL_S = 0.02  # reading a frame + tracking already takes ~30 ms
OBJECT_GIVE_UP_S = 10.0  # object unseen this long: stop tracking
FACE_TRACK_SCORE = 0.6  # YuNet confidence for tracking, the SDK's default (as on the robot)
OBJECT_WAIT_S = 60.0  # not in view at the start: wait this long for it to show up
DEFAULT_CONTEXT_TOKENS = 8000
PUSH_RESERVE_TOKENS = 200  # up to 3 related-memory lines of 240 chars, sent with the user turn


def _mark_own_voice(markers: list[SoundEvents | SpeakerGate], until: float) -> None:
    for m in markers:
        m.mark_own_voice(until)


def speaker_gate_from_env() -> SpeakerGate | None:
    """Owner mode's voice check (SPEAKER_MODEL, SPEAKER_HIGH, SPEAKER_LOW, the enrolled
    voiceprint), or None after printing why it is off. Never raises: no gate only means
    owner mode can't be turned on."""
    limits = {}
    for name, default in (("SPEAKER_HIGH", DEFAULT_HIGH), ("SPEAKER_LOW", DEFAULT_LOW)):
        raw = os.environ.get(name, "").strip()
        try:
            limits[name] = float(raw) if raw else default
        except ValueError:
            print(f"  (owner mode: off: {name} must be a number, got {raw!r})")
            return None
    high, low = limits["SPEAKER_HIGH"], limits["SPEAKER_LOW"]
    if not -1 <= low <= high <= 1:
        print(f"  (owner mode: off: need -1 <= SPEAKER_LOW <= SPEAKER_HIGH <= 1, got {low} {high})")
        return None
    if not VOICEPRINT.is_file():
        print(f"  (owner mode: off: no {VOICEPRINT}; run reachy-demo enroll --file YOUR.wav)")
        return None
    model = Path(os.environ.get("SPEAKER_MODEL", "").strip() or SPEAKER_MODEL)
    try:
        embed = load_sherpa_embed(model)
        voice = np.load(VOICEPRINT)
    except (ImportError, OSError, ValueError) as e:  # FileNotFoundError is an OSError
        print(f"  (owner mode: off: {e})")
        return None
    print(f"  (owner mode: available, high {high} low {low})")
    return SpeakerGate(embed, voice, high, low)


def face_score_from_env() -> float | None:
    """Debug view's minimum face confidence (FACE_MIN_SCORE), or None after a config error."""
    raw = os.environ.get("FACE_MIN_SCORE", "").strip()
    try:
        value = float(raw) if raw else FACE_MIN_SCORE
    except ValueError:
        value = -1.0
    if not 0.0 < value <= 1.0:
        print(f"config error: FACE_MIN_SCORE must be 0-1, got {raw!r}", file=sys.stderr)
        return None
    return value


def camera_lag_from_env() -> float | None:
    """Camera + motor delay for object tracking (CAMERA_LAG_S), or None after a config error."""
    raw = os.environ.get("CAMERA_LAG_S", "").strip()
    try:
        value = float(raw) if raw else CAMERA_LAG_S
    except ValueError:
        value = -1.0
    if not 0.0 <= value <= 2.0:
        print(f"config error: CAMERA_LAG_S must be 0-2 seconds, got {raw!r}", file=sys.stderr)
        return None
    return value


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
    api_key: str,
    base_url: str,
    memory_max_tokens: int = DEFAULT_MEMORY_TOKENS,
    speaker: Callable[[], Speaker] = lambda: OWNER,
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
        store, consolidator, speaker=speaker, embedder=embedder, index=index
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
