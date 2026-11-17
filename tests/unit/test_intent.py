import asyncio
from collections.abc import AsyncIterator

import pytest

from reachy_demo.agent.intent import Intent, classify
from reachy_demo.agent.llm import LLMError


def _chat(reply: str, seen: list[list[dict[str, str]]] | None = None):  # type: ignore[no-untyped-def]
    async def chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if seen is not None:
            seen.append(messages)
        for i in range(0, len(reply), 5):  # streamed in pieces
            yield reply[i : i + 5]

    return chat


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"stop": true, "rest": ""}', Intent(True, "")),
        ('{"stop": true, "rest": "what is the weather"}', Intent(True, "what is the weather")),
        ('{"stop": false, "rest": ""}', Intent(False, "")),
        ('{"stop": true, "rest": null}', Intent(True, "")),
        ('{"stop": true}', Intent(True, "")),
        ('```json\n{"stop": true, "rest": ""}\n```', Intent(True, "")),
        ("sure!", Intent(False, "")),
        ('{"stop": "yes", "rest": ""}', Intent(False, "")),
        ("[1, 2]", Intent(False, "")),
    ],
)
async def test_classify_parses(reply: str, expected: Intent) -> None:
    assert await classify("stop talking", _chat(reply)) == expected


async def test_chat_error_is_not_stop() -> None:
    async def chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        raise LLMError("DeepSeek returned 401")
        yield ""

    assert await classify("stop talking", chat) == Intent(False, "")


async def test_timeout_is_not_stop() -> None:
    async def chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        await asyncio.sleep(1)
        yield '{"stop": true, "rest": ""}'

    assert await classify("stop talking", chat, timeout_s=0.05) == Intent(False, "")


async def test_prompt_carries_the_text() -> None:
    seen: list[list[dict[str, str]]] = []
    await classify("dừng lại, kể chuyện khác đi", _chat('{"stop": true, "rest": ""}', seen))
    assert seen[0][0]["role"] == "system"
    assert seen[0][-1] == {"role": "user", "content": "dừng lại, kể chuyện khác đi"}


@pytest.mark.parametrize(
    "text",
    [
        "Stop.",
        "Stop, stop!",
        "stop",
        "Okay, stop.",
        "Thôi.",
        "Thoi--",
        "Thôi đi.",
        "Dừng lại.",
        "...dừng lại đi.",
        "Đủ rồi.",
        "Wait!",
        "Im đi",
    ],
)
async def test_bare_stop_skips_the_llm(text: str) -> None:
    seen: list[list[dict[str, str]]] = []
    assert await classify(text, _chat('{"stop": false}', seen)) == Intent(True, "")
    assert seen == []


@pytest.mark.parametrize(
    "text",
    ["ok", "stop, what time is it?", "don't stop", "wait a second, tell me more", "I'm fine"],
)
async def test_anything_more_goes_to_the_llm(text: str) -> None:
    seen: list[list[dict[str, str]]] = []
    await classify(text, _chat('{"stop": false}', seen))
    assert len(seen) == 1
