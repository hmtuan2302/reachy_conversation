import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from reachy_demo.agent.agent import GAVE_UP, MAX_ROUNDS, SYSTEM_PROMPT, Agent
from reachy_demo.agent.llm import LLMError, ToolCall
from reachy_demo.ops.tokens import est_tokens


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


def _tok(*msgs: dict[str, Any]) -> int:
    return sum(est_tokens(json.dumps(m, ensure_ascii=False)) for m in msgs)


SYS = {"role": "system", "content": SYSTEM_PROMPT}


def _pair(q: str, n: int) -> list[dict[str, Any]]:
    return [{"role": "user", "content": q}, {"role": "assistant", "content": f"a{n} done"}]


async def test_token_budget_drops_oldest_whole_turns() -> None:
    chat = FakeChat()
    qs = [c * 400 for c in "xyz"]
    u3 = {"role": "user", "content": qs[2]}
    budget = _tok(SYS, *_pair(qs[1], 2), u3) + 5
    agent = Agent(chat, max_tokens=budget)
    for q in qs:
        await _ask(agent, q)
    assert chat.calls[-1] == [SYS, *_pair(qs[1], 2), u3]
    assert len(agent.history) == 6  # stored history is not trimmed


async def test_token_budget_keeps_tool_turn_whole() -> None:
    call = ToolCall("c1", "turn_head", '{"direction": "left"}')
    chat = _tool_chat([[call], ["ok"], ["fine"]])
    u2 = {"role": "user", "content": "q2"}
    agent = Agent(chat, tools=FakeTools(), max_tokens=10**6)
    await _ask(agent, "q1" * 200)
    tail = agent.history[-2:]  # tool reply + final answer, not the whole turn
    agent._max_tokens = _tok(SYS, u2, *tail) + 2
    await _ask(agent, "q2")
    assert chat.seen[-1] == [SYS, u2]


async def test_current_turn_sent_even_over_budget(caplog: pytest.LogCaptureFixture) -> None:
    chat = FakeChat()
    q = "x" * 2000
    with caplog.at_level(logging.WARNING):
        await _ask(Agent(chat, max_tokens=10), q)
    assert chat.calls[0] == [SYS, {"role": "user", "content": q}]
    warnings = [r for r in caplog.records if "context budget" in r.getMessage()]
    assert len(warnings) == 1


class BigTools(FakeTools):
    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        return {"status": "x" * 2000}


async def test_budget_rechecked_each_round() -> None:
    call = ToolCall("c1", "look", "{}")
    chat = _tool_chat([["hello"], [call], ["done"]])
    agent = Agent(chat, tools=BigTools())
    await _ask(agent, "hi")
    agent._max_tokens = _tok(SYS, *agent.history, {"role": "user", "content": "q2"}) + 50
    await _ask(agent, "q2")
    hi = {"role": "user", "content": "hi"}
    assert hi in chat.seen[1]  # round 1 fits the old turn
    assert hi not in chat.seen[2]  # round 2 carries the big tool result, so it is dropped


async def test_no_max_tokens_sends_full_history() -> None:
    chat = FakeChat()
    agent = Agent(chat)
    for i in range(5):
        await _ask(agent, str(i) * 2000)
    assert len(chat.calls[-1]) == 1 + 2 * 4 + 1


async def test_vietnamese_counts_characters_not_escapes() -> None:
    chat = FakeChat()
    q1, q2 = "Xin chào, bạn tên là gì? " * 20, "Hôm nay trời đẹp quá."
    u2 = {"role": "user", "content": q2}
    agent = Agent(chat, max_tokens=_tok(SYS, *_pair(q1, 1), u2) + 2)
    await _ask(agent, q1)
    await _ask(agent, q2)
    assert chat.calls[-1] == [SYS, *_pair(q1, 1), u2]


async def test_warns_when_history_dropped_entirely(caplog: pytest.LogCaptureFixture) -> None:
    chat = FakeChat()
    u2 = {"role": "user", "content": "q2"}
    agent = Agent(chat, max_tokens=10**6)
    await _ask(agent, "x" * 2000)
    agent._max_tokens = _tok(SYS, u2) + 20
    with caplog.at_level(logging.WARNING):
        await _ask(agent, "q2")
    assert chat.calls[-1] == [SYS, u2]
    warnings = [r for r in caplog.records if "context budget" in r.getMessage()]
    assert len(warnings) == 1


# -- related memories (push) -----------------------------------------------------------------


def _related(block: str, fail: bool = False) -> Any:
    calls: list[str] = []

    async def related(text: str) -> str:
        calls.append(text)
        if fail:
            raise RuntimeError("db gone")
        return block

    related.calls = calls  # type: ignore[attr-defined]
    return related


async def test_related_block_sent_but_not_stored() -> None:
    chat = FakeChat()
    turns: list[list[dict[str, Any]]] = []
    agent = Agent(chat, related=_related("[related memories]\n- fact: x"), on_turn=turns.append)
    await _ask(agent, "hello")
    assert chat.calls[0][-1]["content"] == "[related memories]\n- fact: x\n\nhello"
    assert agent.history[0] == {"role": "user", "content": "hello"}
    assert turns[0][0] == {"role": "user", "content": "hello"}


async def test_related_awaited_once_per_turn_across_tool_rounds() -> None:
    call = ToolCall("c", "turn_head", '{"direction": "left"}')
    chat = _tool_chat([[call], ["ok"]])
    rel = _related("[related memories]\n- fact: x")
    await _ask(Agent(chat, tools=FakeTools(), related=rel), "q")
    assert rel.calls == ["q"]
    assert all(r[1]["content"].startswith("[related memories]") for r in chat.seen)


async def test_related_hook_error_still_answers() -> None:
    chat = FakeChat()
    assert await _ask(Agent(chat, related=_related("", fail=True)), "hello") == "a1 done"
    assert chat.calls[0][-1]["content"] == "hello"


async def test_related_block_counts_in_budget() -> None:
    chat = FakeChat()
    block = "[related memories]\n- fact: " + "z" * 400
    agent = Agent(chat, max_tokens=10**6, related=_related(block))
    await _ask(agent, "first")
    sent = {"role": "user", "content": f"{block}\n\nsecond"}
    agent._max_tokens = _tok(SYS, sent) + 5  # fits the turn with its block, not the old turn
    await _ask(agent, "second")
    assert chat.calls[-1] == [SYS, sent]


async def test_debug_shows_memory_block_once_related_and_tool_results() -> None:
    call = ToolCall("c", "turn_head", '{"direction": "left"}')
    chat = _tool_chat([[call], ["ok"], ["fine"]])
    shown: list[tuple[str, str]] = []
    agent = Agent(
        chat,
        tools=FakeTools(),
        context=lambda: "[memory] A",
        related=_related("[related memories]\n- fact: x"),
        debug=lambda kind, text: shown.append((kind, text)),
    )
    await _ask(agent, "q1")
    await _ask(agent, "q2")
    assert [k for k, _ in shown] == [
        "memory block", "related memories", "tool turn_head", "related memories",
    ]  # fmt: skip
    assert shown[2][1] == '{"status": "turn_head"}'


async def test_no_debug_by_default() -> None:
    chat = FakeChat()
    await _ask(Agent(chat, context=lambda: "[memory] A"), "q")  # nothing to call, no error
    assert chat.calls
