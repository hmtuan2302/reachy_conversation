"""Question-answering agent: one completed user turn in, a streamed text answer out."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

ChatFn = Callable[[list[dict[str, str]]], AsyncIterator[str]]

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


class Agent:
    def __init__(
        self, chat: ChatFn, system_prompt: str = SYSTEM_PROMPT, max_turns: int = 10
    ) -> None:
        self._chat = chat
        self._system_prompt = system_prompt
        self._max_messages = 2 * max_turns
        self.history: list[dict[str, str]] = []

    async def answer(self, text: str) -> AsyncIterator[str]:
        """Stream the answer to `text`. History is updated only when the answer completes."""
        question = {"role": "user", "content": text}
        messages = [{"role": "system", "content": self._system_prompt}, *self.history, question]
        parts: list[str] = []
        async for piece in self._chat(messages):
            parts.append(piece)
            yield piece
        self.history += [question, {"role": "assistant", "content": "".join(parts)}]
        self.history = self.history[-self._max_messages :]
