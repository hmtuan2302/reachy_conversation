import json

import httpx
import pytest

from reachy_demo.agent.llm import LLMError, chat_stream

MSGS = [{"role": "user", "content": "Thủ đô?"}]


def _sse(*events: str) -> bytes:
    return "".join(events).encode()


def _delta(content: str | None = None, reasoning: str | None = None) -> str:
    delta: dict[str, str] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return "data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n"


async def _collect(transport: httpx.MockTransport) -> list[str]:
    return [t async for t in chat_stream(MSGS, api_key="k", transport=transport)]


async def test_streams_answer_deltas_only() -> None:
    seen: list[httpx.Request] = []

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        body = _sse(
            ": keep-alive\n\n",
            _delta(reasoning="thinking..."),
            _delta("Hà "),
            _delta(""),
            _delta("Nội"),
            "data: [DONE]\n\n",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    assert await _collect(httpx.MockTransport(handle)) == ["Hà ", "Nội"]
    req = seen[0]
    assert str(req.url) == "https://api.deepseek.com/chat/completions"
    assert req.headers["Authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body == {"model": "deepseek-chat", "messages": MSGS, "stream": True}


async def test_http_error_raises_llm_error() -> None:
    def handle(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Authentication Fails"}})

    with pytest.raises(LLMError, match="401.*Authentication Fails"):
        await _collect(httpx.MockTransport(handle))


async def test_network_error_raises_llm_error() -> None:
    def handle(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=req)

    with pytest.raises(LLMError, match="no route"):
        await _collect(httpx.MockTransport(handle))
