import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from reachy_demo.voice.translate import translate_stream

Handler = Callable[[ServerConnection], Awaitable[None]]


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for p in parts:
        yield p


async def _run(handler: Handler, target: str = "vi") -> list[str]:
    async with serve(handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        return [
            d
            async for d in translate_stream(
                _chunks(b"\x01\x00", b"\x02\x00"), api_key="sk-test", target=target, url=url
            )
        ]


def _echo_server(log: list[dict[str, Any]]) -> Handler:
    async def handler(ws: ServerConnection) -> None:
        log.append({"auth": ws.request.headers["Authorization"]} if ws.request else {})
        async for raw in ws:
            msg = json.loads(raw)
            log.append(msg)
            if msg["type"] == "session.close":
                for d in ("Xin ", "chào"):
                    await ws.send(
                        json.dumps({"type": "session.output_transcript.delta", "delta": d})
                    )
                await ws.send(json.dumps({"type": "session.closed"}))

    return handler


@pytest.mark.asyncio
async def test_yields_output_transcript_deltas_until_closed() -> None:
    log: list[dict[str, Any]] = []
    assert await _run(_echo_server(log)) == ["Xin ", "chào"]
    appended = b"".join(
        base64.b64decode(m["audio"])
        for m in log
        if m.get("type") == "session.input_audio_buffer.append"
    )
    assert appended == b"\x01\x00\x02\x00"


@pytest.mark.asyncio
async def test_sends_auth_header_and_target_language_first() -> None:
    log: list[dict[str, Any]] = []
    await _run(_echo_server(log), target="en")
    assert log[0]["auth"] == "Bearer sk-test"
    assert log[1] == {
        "type": "session.update",
        "session": {"audio": {"output": {"language": "en"}}},
    }


@pytest.mark.asyncio
async def test_ws_close_mid_stream_raises_connection_error() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        await ws.send(json.dumps({"type": "session.output_transcript.delta", "delta": "Hi"}))
        await ws.close(code=1011)

    with pytest.raises(ConnectionError):
        await _run(handler)
