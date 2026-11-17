"""Question-answering agent: one completed user turn in, a streamed text answer out."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from reachy_demo.agent.llm import ToolCall
from reachy_demo.ops.tokens import est_tokens

ChatFn = Callable[[list[dict[str, str]]], AsyncIterator[str]]
# Like ChatFn, but the model may also ask for tool calls (tools are bound by the caller).
ToolChatFn = Callable[[list[dict[str, Any]]], AsyncIterator[str | ToolCall]]

SYSTEM_PROMPT = (
    "You are Reachy, a friendly robot assistant answering spoken questions. "
    "Always reply in the same language the user spoke (Vietnamese or English). "
    "Keep answers short: at most 3 sentences unless the user asks for detail. "
    "Write plain text only, with no markdown, lists, or emoji, "
    "because the answer will be spoken aloud. "
    "The question comes from speech recognition and may contain mistakes; "
    "infer the likely meaning. "
    "If you are not sure of a fact, say so instead of guessing."
)
log = logging.getLogger(__name__)

MAX_ROUNDS = 4  # LLM calls per user turn, tool rounds included
GAVE_UP = "Sorry, I could not finish that."


class Tools(Protocol):
    async def call(self, name: str, arguments: str) -> dict[str, Any]: ...

    def needs_response(self, name: str, arguments: str) -> bool:
        """False for a pure action: when it succeeds, the move is the answer (no text, no TTS)."""
        ...


class Agent:
    def __init__(
        self,
        chat: ToolChatFn,
        system_prompt: str = SYSTEM_PROMPT,
        max_turns: int = 10,
        tools: Tools | None = None,
        context: Callable[[], str] | None = None,
        on_turn: Callable[[list[dict[str, Any]]], None] | None = None,
        max_tokens: int | None = None,
        related: Callable[[str], Awaitable[str]] | None = None,
        debug: Callable[[str, str], None] | None = None,
    ) -> None:
        """`context()` is appended to the system prompt each round (memory briefing);
        `on_turn(turn)` gets each completed turn (user, assistant and tool messages);
        `max_tokens` caps each request (None = no cap): oldest whole history turns are left out;
        `related(text)` is awaited once per turn; its block is prepended to the user message
        that is *sent*, never to the one stored in history or given to `on_turn`;
        `debug(kind, text)` sees the context the model gets: the memory block when it changes,
        the related memories, and each tool result."""
        self._chat = chat
        self._context = context
        self._on_turn = on_turn
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._related = related
        self._debug = debug
        self._shown_context = ""
        self._tools = tools
        # Whole turns, so trimming never orphans a tool reply from its tool call.
        self._turns: list[list[dict[str, Any]]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return [m for turn in self._turns for m in turn]

    def preload(self, messages: list[dict[str, Any]]) -> None:
        """Seed history with earlier (user, assistant) pairs, e.g. from before a restart."""
        pairs = [messages[i : i + 2] for i in range(0, len(messages) - 1, 2)]
        self._turns = [*pairs, *self._turns][-self._max_turns :]

    def _fit(self, room: int) -> list[dict[str, Any]]:
        """Newest whole turns that fit in `room` tokens; stops at the first that doesn't."""
        kept: list[list[dict[str, Any]]] = []
        for turn in reversed(self._turns):
            room -= _tokens(turn)
            if room < 0:
                break
            kept.append(turn)
        return [m for turn in reversed(kept) for m in turn]

    async def answer(self, text: str) -> AsyncIterator[str]:
        """Stream the answer to `text`, running tool calls in between.

        Tool calls run one after another, in the order the model sent them: head moves
        must not overlap. History is updated only when the answer completes.
        """
        turn: list[dict[str, Any]] = [{"role": "user", "content": text}]
        block = ""
        if self._related:
            try:
                block = await self._related(text)
            except Exception:  # memory trouble must not cost the answer
                log.exception("related memories failed")
        sent_user = {"role": "user", "content": f"{block}\n\n{text}"} if block else turn[0]
        warned = False
        for round_no in range(MAX_ROUNDS):
            extra = self._context() if self._context else ""
            if self._debug and round_no == 0:
                if extra and extra != self._shown_context:
                    self._debug("memory block", extra)
                    self._shown_context = extra
                if block:
                    self._debug("related memories", block)
            system = f"{self._system_prompt}\n\n{extra}" if extra else self._system_prompt
            head = {"role": "system", "content": system}
            history = self.history
            if self._max_tokens is not None:
                room = self._max_tokens - _tokens([head, sent_user, *turn[1:]])
                if room < 0 and not warned:
                    log.warning(
                        "context budget exceeded by the current turn",
                        extra={"max_tokens": self._max_tokens, "over_by": -room},
                    )
                    warned = True
                history = self._fit(room)
                if self._turns and not history and room >= 0 and not warned:
                    log.warning(
                        "context budget leaves no room for history",
                        extra={"max_tokens": self._max_tokens, "turns_dropped": len(self._turns)},
                    )
                    warned = True
            messages = [head, *history, sent_user, *turn[1:]]
            parts: list[str] = []
            calls: list[ToolCall] = []
            async for piece in self._chat(messages):
                if isinstance(piece, ToolCall):
                    calls.append(piece)
                else:
                    parts.append(piece)
                    yield piece
            if not calls or self._tools is None:
                turn.append({"role": "assistant", "content": "".join(parts)})
                break
            turn.append(
                {
                    "role": "assistant",
                    "content": "".join(parts) or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in calls
                    ],
                }
            )
            results = []
            for c in calls:  # sequential on purpose
                result = await self._tools.call(c.name, c.arguments)
                results.append(result)
                if self._debug:
                    self._debug(f"tool {c.name}", json.dumps(result, ensure_ascii=False))
                turn.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
            silent = all(not self._tools.needs_response(c.name, c.arguments) for c in calls)
            if silent and all("error" not in r for r in results):
                break  # actions done: skip the follow-up LLM call; a failure still gets one
        else:
            yield GAVE_UP
            turn.append({"role": "assistant", "content": GAVE_UP})
        self._turns = [*self._turns, turn][-self._max_turns :]
        if self._on_turn:
            self._on_turn(turn)


def _tokens(messages: list[dict[str, Any]]) -> int:
    return sum(est_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
