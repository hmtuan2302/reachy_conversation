from collections.abc import AsyncIterator
from typing import Any

import pytest

from reachy_demo.agent.agent import GAVE_UP, MAX_ROUNDS, SYSTEM_PROMPT, Agent
from reachy_demo.agent.llm import LLMError, ToolCall


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


class FakeTools:
    def __init__(self, silent: bool = False, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.silent = silent
        self.fail = fail

    def needs_response(self, name: str, arguments: str) -> bool:
        return not self.silent or '"reply": true' in arguments

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"error": "move_failed: x"} if self.fail else {"status": name}


def _tool_chat(rounds: list[list[str | ToolCall]]) -> Any:
    seen: list[list[dict[str, Any]]] = []

    async def chat(messages: list[dict[str, Any]]) -> AsyncIterator[str | ToolCall]:
        seen.append(list(messages))
        for piece in rounds[min(len(seen), len(rounds)) - 1]:
            yield piece

    chat.seen = seen  # type: ignore[attr-defined]
    return chat


async def test_tool_calls_run_in_order_then_answer() -> None:
    right = ToolCall("c1", "turn_head", '{"direction": "right"}')
    down = ToolCall("c2", "tilt_head", '{"direction": "down"}')
    chat = _tool_chat([[right, down], ["Done."]])
    tools = FakeTools()
    agent = Agent(chat, tools=tools)
    assert await _ask(agent, "look right and down") == "Done."
    assert tools.calls == [("turn_head", right.arguments), ("tilt_head", down.arguments)]
    second = chat.seen[1]
    assert second[2]["tool_calls"][1]["function"]["name"] == "tilt_head"
    assert second[3:] == [
        {"role": "tool", "tool_call_id": "c1", "content": '{"status": "turn_head"}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"status": "tilt_head"}'},
    ]
    assert agent.history[-1] == {"role": "assistant", "content": "Done."}


async def test_gives_up_after_max_rounds() -> None:
    loop = ToolCall("c", "turn_head", '{"direction": "left"}')
    tools = FakeTools()
    agent = Agent(_tool_chat([[loop]]), tools=tools)
    assert await _ask(agent, "spin forever") == GAVE_UP
    assert len(tools.calls) == MAX_ROUNDS


async def test_history_trims_whole_turns_with_tool_messages() -> None:
    call = ToolCall("c", "turn_head", '{"direction": "left"}')
    agent = Agent(_tool_chat([[call], ["ok"]]), tools=FakeTools(), max_turns=1)
    await _ask(agent, "q1")
    await _ask(agent, "q2")
    assert agent.history[0] == {"role": "user", "content": "q2"}


async def test_action_tools_end_the_turn_without_a_text_answer() -> None:
    right = ToolCall("c1", "turn_head", '{"direction": "right"}')
    down = ToolCall("c2", "tilt_head", '{"direction": "down"}')
    chat = _tool_chat([[right, down], ["Done."]])
    tools = FakeTools(silent=True)
    agent = Agent(chat, tools=tools)
    assert await _ask(agent, "look right and down") == ""
    assert len(chat.seen) == 1  # no follow-up LLM call
    assert [n for n, _ in tools.calls] == ["turn_head", "tilt_head"]
    assert agent.history[-1]["role"] == "tool"


async def test_failed_action_still_gets_a_spoken_answer() -> None:
    call = ToolCall("c1", "turn_head", '{"direction": "right"}')
    chat = _tool_chat([[call], ["I could not move."]])
    agent = Agent(chat, tools=FakeTools(silent=True, fail=True))
    assert await _ask(agent, "turn right") == "I could not move."


async def test_action_flagged_for_reply_gets_a_follow_up_answer() -> None:
    call = ToolCall("c1", "turn_head", '{"direction": "left", "reply": true}')
    chat = _tool_chat([[call], ["Why did the robot cross the road?"]])
    agent = Agent(chat, tools=FakeTools(silent=True))
    assert await _ask(agent, "turn left and tell me a joke") == "Why did the robot cross the road?"


async def test_context_is_appended_to_system_prompt_each_round() -> None:
    chat = FakeChat()
    calls = iter(["[memory] A", "[memory] B"])
    agent = Agent(chat, context=lambda: next(calls))
    await _ask(agent, "q1")
    await _ask(agent, "q2")
    assert chat.calls[0][0]["content"] == f"{SYSTEM_PROMPT}\n\n[memory] A"
    assert chat.calls[1][0]["content"] == f"{SYSTEM_PROMPT}\n\n[memory] B"


async def test_empty_context_leaves_prompt_unchanged() -> None:
    chat = FakeChat()
    await _ask(Agent(chat, context=lambda: ""), "q1")
    assert chat.calls[0][0]["content"] == SYSTEM_PROMPT


async def test_on_turn_gets_completed_turn_only() -> None:
    seen: list[list[dict[str, Any]]] = []
    agent = Agent(FakeChat(fail_on=2), on_turn=seen.append)
    await _ask(agent, "q1")
    with pytest.raises(LLMError):
        await _ask(agent, "q2")
    assert seen == [
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1 done"}]
    ]


async def test_preload_seeds_history_before_new_turns() -> None:
    chat = FakeChat()
    agent = Agent(chat, max_turns=2)
    agent.preload(
        [
            {"role": "user", "content": "old1"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "old2"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "old3"},
            {"role": "assistant", "content": "r3"},
        ]
    )
    await _ask(agent, "new")
    assert [m["content"] for m in chat.calls[0][1:]] == ["old2", "r2", "old3", "r3", "new"]
