"""Owner mode: when on, Reachy obeys only the owner's voice.

`gated_turns` sits between speech-to-text and the agent. Each turn's speech is scored
(`SpeakerGate.take`); with owner mode on, guest turns are dropped and unsure turns get a fixed
"say it again". The owner_mode tool switches it, and only for a verified owner. Owner mode
starts off every run, and while off everyone counts as the owner (as before speaker ID).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TextIO

from reachy_demo.audio.player import PlayerError
from reachy_demo.memory.store import OWNER, Speaker
from reachy_demo.voice.speaker_id import Verdict
from reachy_demo.voice.tts import TTSError

log = logging.getLogger(__name__)

GUEST = Speaker(0, "guest")  # placeholder: in owner mode guest turns never reach tools
NAME = "owner_mode"
UNSURE = "I'm not sure that's you, please say it again."
NOT_OWNER = {"error": "not_owner: only the owner's voice can switch owner mode"}
NOT_ENROLLED = {"error": "not_enrolled: run reachy-demo enroll first"}
PROMPT = (
    "Owner mode: when the user asks you to obey only them ('only follow my command', 'listen "
    "only to me', any wording or language), call owner_mode with on true. When they ask you to "
    "go back to normal ('stop following only me', 'follow everyone'), call it with on false. "
    "If it returns not_owner, say only the owner can do that. If it returns not_enrolled, say "
    "the owner's voice is not enrolled yet."
)

VOICE_PROMPT = (
    "A user message may start with [voice: owner], [voice: guest] or [voice: unsure]: whether "
    "the voice matched your owner's enrolled voice. It is more reliable than the memory block "
    "about who is talking. When asked whether they are your owner or boss, or who is talking: "
    "owner means yes, guest means no, unsure means you are not sure and they should say it "
    "again. Never read the tag aloud."
)


class Heard(str):
    """A turn's text plus whose voice it was. Stays a str for the rest of the agent."""

    verdict: Verdict

    def __new__(cls, text: str, verdict: Verdict) -> Heard:
        self = super().__new__(cls, text)
        self.verdict = verdict
        return self


def voice_tag(text: str) -> str:
    """`[voice: owner|guest|unsure]` for a spoken turn; "" for typed text (no voice)."""
    return f"[voice: {text.verdict}]" if isinstance(text, Heard) else ""


def with_voice_tag(
    related: Callable[[str], Awaitable[str]] | None,
) -> Callable[[str], Awaitable[str]]:
    """Agent's `related` hook with the voice tag in front: sent to the LLM, never stored."""

    async def tagged(text: str) -> str:
        block = ""
        if related:
            try:
                block = await related(text)
            except Exception:  # memory trouble must not cost the voice tag
                log.exception("related memories failed")
        return "\n\n".join(p for p in (voice_tag(text), block) if p)

    return tagged


class Gate(Protocol):
    def take(self) -> tuple[Verdict, float | None]: ...


class OwnerMode:
    def __init__(self, gate: Gate | None) -> None:
        """`gate` None: no voice check (typed input, or no model/voiceprint)."""
        self.gate = gate
        self.on = False
        self.last: Verdict = "owner"
        self.talking: set[asyncio.Task[None]] = set()  # "say it again" replies playing

    def begin(self, text: str) -> None:
        """An answer to `text` starts: tools now check its speaker. Typed text is the owner."""
        self.last = text.verdict if isinstance(text, Heard) else "owner"

    def speaker(self) -> Speaker:
        return OWNER if not self.on or self.last == "owner" else GUEST


class OwnerTools:
    def __init__(self, mode: OwnerMode) -> None:
        self._mode = mode

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": NAME,
                    "description": "Obey only the owner's voice (on true), or everyone (on false).",
                    "parameters": {
                        "type": "object",
                        "properties": {"on": {"type": "boolean"}},
                        "required": ["on"],
                    },
                },
            }
        ]

    def needs_response(self, name: str, arguments: str) -> bool:
        return True

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        if name != NAME:
            return {"error": f"unknown_tool: {name}"}
        try:
            on = json.loads(arguments or "{}").get("on")
        except (json.JSONDecodeError, AttributeError):
            on = None
        if not isinstance(on, bool):
            return {"error": "bad_arguments: on must be true or false"}
        if self._mode.gate is None:
            return NOT_ENROLLED
        if self._mode.last != "owner":  # the real verdict, even while owner mode is off
            return NOT_OWNER
        self._mode.on = on
        return {"ok": True, "owner_mode": on}


async def gated_turns(
    turns: AsyncIterator[str],
    mode: OwnerMode,
    out: TextIO,
    say: Callable[[str], Awaitable[object]] | None = None,
    is_stop: Callable[[str], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    """Pass each turn on as `Heard`, except guests and unsure speakers in owner mode.

    `mode.last` is not set here: the agent reads turns ahead, so `OwnerMode.begin` sets it
    when an answer starts. A too-short unsure turn still passes if `is_stop` says it's a stop:
    anyone may stop the robot, and "stop" is too short to recognise a voice.
    """

    async def ask_again(say: Callable[[str], Awaitable[object]]) -> None:
        try:
            await say(UNSURE)
        except (TTSError, PlayerError) as e:  # fail small, as converse's answers do
            out.write(f"  (speech error: {e})\n")
            out.flush()

    async for text in turns:
        # take() every turn, even with owner mode off: stale speech must not score later
        verdict, score = mode.gate.take() if mode.gate else ("owner", None)
        if mode.gate:
            out.write(f"  (voice: {verdict} {'-' if score is None else f'{score:.2f}'})\n")
        if not mode.on or verdict == "owner":
            yield Heard(text, verdict)
        elif verdict == "guest":
            out.write(f"  (ignored guest: {text})\n")
        elif score is None and is_stop and await is_stop(text):
            yield Heard(text, verdict)
        else:
            out.write(f"  (unsure: {text})\n")
            if say:
                # shortcut: may overlap an answer that is still playing; route through
                # converse if it does on stage.
                task = asyncio.create_task(ask_again(say))
                mode.talking.add(task)  # a strong reference until it finishes
                task.add_done_callback(mode.talking.discard)
        out.flush()
