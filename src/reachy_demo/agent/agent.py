"""Question-answering agent: one completed user turn in, a streamed text answer out."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from reachy_demo.agent.llm import ToolCall

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
    ) -> None:
        self._chat = chat
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._tools = tools
        # Whole turns, so trimming never orphans a tool reply from its tool call.
        self._turns: list[list[dict[str, Any]]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return [m for turn in self._turns for m in turn]

    async def answer(self, text: str) -> AsyncIterator[str]:
        """Stream the answer to `text`, running tool calls in between.

        Tool calls run one after another, in the order the model sent them: head moves
        must not overlap. History is updated only when the answer completes.
        """
        turn: list[dict[str, Any]] = [{"role": "user", "content": text}]
        for _ in range(MAX_ROUNDS):
            messages = [{"role": "system", "content": self._system_prompt}, *self.history, *turn]
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
                turn.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
            silent = all(not self._tools.needs_response(c.name, c.arguments) for c in calls)
            if silent and all("error" not in r for r in results):
                break  # actions done: skip the follow-up LLM call; a failure still gets one
        else:
            yield GAVE_UP
            turn.append({"role": "assistant", "content": GAVE_UP})
        self._turns = [*self._turns, turn][-self._max_turns :]
