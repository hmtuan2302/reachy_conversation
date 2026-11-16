import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from websockets.asyncio.server import ServerConnection, serve

from reachy_demo.perception.vision import VisionError
from reachy_demo.perception.vision import ask_image_hf as ask_image

Handler = Callable[[ServerConnection], Awaitable[None]]
JPEG = b"\xff\xd8fake-jpeg\xff\xd9"


def _allocator(connect_url: str, status: int = 200) -> httpx.MockTransport:
    def handle(req: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"connect_url": connect_url})

    return httpx.MockTransport(handle)


async def _ask(handler: Handler, status: int = 200, timeout_s: float = 5.0) -> str:
    async with serve(handler, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/v1/realtime"
        return await ask_image(
            JPEG,
            "what is this?",
            hf_token="hf_test",
            transport=_allocator(url, status),
            timeout_s=timeout_s,
        )


def _server(log: list[Any], events: list[dict[str, Any]]) -> Handler:
    async def handler(ws: ServerConnection) -> None:
        log.append(ws.request.headers["Authorization"] if ws.request else None)
        async for raw in ws:
            msg = json.loads(raw)
            log.append(msg)
            if msg["type"] == "response.create":
                for e in events:
                    await ws.send(json.dumps(e))

    return handler


async def test_sends_image_and_question_and_joins_the_answer() -> None:
    log: list[Any] = []
    events = [
        {"type": "response.output_audio.delta", "delta": "AAAA"},  # audio is ignored
        {"type": "response.output_audio_transcript.delta", "delta": "A white "},
        {"type": "response.output_text.delta", "delta": "mug."},
        {"type": "response.done"},
    ]
    assert await _ask(_server(log, events)) == "A white mug."
    assert log[0] == "Bearer hf_test"
    session, item, create = log[1], log[2], log[3]
    assert session["type"] == "session.update"
    assert session["session"]["output_modalities"] == ["text"]
    content = item["item"]["content"]
    assert content[0] == {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode(),
    }
    assert content[1] == {"type": "input_text", "text": "what is this?"}
    assert create == {"type": "response.create"}


async def test_server_error_raises() -> None:
    events = [{"type": "error", "error": {"message": "image too large"}}]
    with pytest.raises(VisionError, match="image too large"):
        await _ask(_server([], events))


async def test_empty_answer_raises() -> None:
    with pytest.raises(VisionError, match="empty"):
        await _ask(_server([], [{"type": "response.done"}]))


async def test_allocator_rejection_raises() -> None:
    with pytest.raises(VisionError, match="401"):
        await _ask(_server([], []), status=401)


async def test_timeout_raises() -> None:
    async def silent(ws: ServerConnection) -> None:
        async for _ in ws:  # reads, never answers; ends when the client gives up
            pass

    with pytest.raises(VisionError, match="took over"):
        await _ask(silent, timeout_s=0.3)


async def test_closed_connection_raises() -> None:
    async def closes(ws: ServerConnection) -> None:
        await ws.recv()
        await ws.close()

    with pytest.raises(VisionError):
        await _ask(closes)


async def test_chat_provider_sends_image_and_returns_answer() -> None:
    from reachy_demo.perception.vision import ask_image_chat

    seen: list[httpx.Request] = []

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": " A mug. "}}]})

    answer = await ask_image_chat(
        JPEG,
        "what is this?",
        api_key="sk-x",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/",
        transport=httpx.MockTransport(handle),
    )
    assert answer == "A mug."
    req = seen[0]
    assert str(req.url) == "https://api.deepseek.com/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-x"
    body = json.loads(req.content)
    assert body["model"] == "deepseek-chat"
    content = body["messages"][1]["content"]
    assert content[0]["image_url"]["url"] == (
        "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    )
    assert content[1] == {"type": "text", "text": "what is this?"}


@pytest.mark.parametrize(
    ("status", "payload", "match"),
    [
        (402, {"error": "no balance"}, "402"),
        (200, {"choices": [{"message": {"content": ""}}]}, "empty answer"),
        (200, {"oops": 1}, "unexpected reply"),
    ],
)
async def test_chat_provider_failures(status: int, payload: Any, match: str) -> None:
    from reachy_demo.perception.vision import ask_image_chat

    transport = httpx.MockTransport(lambda req: httpx.Response(status, json=payload))
    with pytest.raises(VisionError, match=match):
        await ask_image_chat(
            JPEG, "q", api_key="k", model="m", base_url="https://x", transport=transport
        )
