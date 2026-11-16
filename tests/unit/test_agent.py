from collections.abc import AsyncIterator

import pytest

from reachy_demo.agent.agent import SYSTEM_PROMPT, Agent
from reachy_demo.agent.llm import LLMError


class FakeChat:
    def __init__(self, fail_on: int | None = None) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.fail_on = fail_on

    async def __call__(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append([dict(m) for m in messages])
        n = len(self.calls)
        yield f"a{n} "
        if n == self.fail_on:
            raise LLMError("network drop")
        yield "done"


async def _ask(agent: Agent, text: str) -> str:
    return "".join([t async for t in agent.answer(text)])


async def test_sends_system_prompt_history_and_question() -> None:
    chat = FakeChat()
    agent = Agent(chat)
    assert await _ask(agent, "q1") == "a1 done"
    await _ask(agent, "q2")
    assert chat.calls[1] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1 done"},
        {"role": "user", "content": "q2"},
    ]


async def test_history_keeps_last_10_pairs() -> None:
    agent = Agent(FakeChat())
    for i in range(12):
        await _ask(agent, f"q{i}")
    assert len(agent.history) == 20
    assert agent.history[0] == {"role": "user", "content": "q2"}


async def test_failed_answer_not_stored() -> None:
    agent = Agent(FakeChat(fail_on=2))
    await _ask(agent, "q1")
    with pytest.raises(LLMError):
        await _ask(agent, "q2")
    assert [m["content"] for m in agent.history] == ["q1", "a1 done"]


def test_prompt_asks_for_users_language_and_plain_short_text() -> None:
    p = SYSTEM_PROMPT.lower()
    assert "language" in p and "markdown" in p and "3 sentences" in p
