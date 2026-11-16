import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

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


async def _bad_source() -> AsyncIterator[bytes]:
    yield b"\x00\x00"
    raise ValueError("need 16-bit PCM WAV")


async def test_source_error_surfaces_without_hang() -> None:
    async def silent(ws: ServerConnection) -> None:
        async for _ in ws:
            pass

    async with serve(silent, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}"
        with pytest.raises(ValueError, match="16-bit"):
            async with asyncio.timeout(2):
                async for _ in translate_stream(_bad_source(), api_key="k", target="en", url=url):
                    pass


async def test_handshake_rejected_raises_connection_error() -> None:
    def reject(conn: ServerConnection, _req: Request) -> Response:
        return conn.respond(HTTPStatus.UNAUTHORIZED, "bad key\n")

    async def never(ws: ServerConnection) -> None:
        pass

    async with serve(never, "127.0.0.1", 0, process_request=reject) as server:
        url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}"
        with pytest.raises(ConnectionError, match="401"):
            async for _ in translate_stream(_chunks(b""), api_key="k", target="en", url=url):
                pass


async def test_server_error_message_reaches_connection_error() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        err = {"type": "error", "error": {"message": "invalid model"}}
        await ws.send(json.dumps(err))
        await ws.close()

    with pytest.raises(ConnectionError, match="invalid model"):
        await _run(handler)
