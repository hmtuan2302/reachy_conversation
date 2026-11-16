import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import pytest
from websockets.asyncio.server import ServerConnection, serve

from reachy_demo.voice.transcribe import transcribe_stream

Handler = Callable[[ServerConnection], Awaitable[None]]
DONE = "conversation.item.input_audio_transcription.completed"


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for p in parts:
        yield p


def _allocator(
    connect_url: str, log: list[httpx.Request], status: int = 200
) -> httpx.MockTransport:
    def handle(req: httpx.Request) -> httpx.Response:
        log.append(req)
        return httpx.Response(status, json={"connect_url": connect_url, "state": "granted"})

    return httpx.MockTransport(handle)


async def _run(
    handler: Handler,
    source: AsyncIterator[bytes] | None = None,
    alloc_log: list[httpx.Request] | None = None,
    status: int = 200,
) -> list[str]:
    async with serve(handler, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/v1/realtime"
        transport = _allocator(url, alloc_log if alloc_log is not None else [], status)
        async with asyncio.timeout(5):
            return [
                t
                async for t in transcribe_stream(
                    source or _chunks(b"\x01\x00"),
                    hf_token="hf_test",
                    transport=transport,
                    idle_s=0.2,
                )
            ]


def _transcribing_server(log: list[Any]) -> Handler:
    async def handler(ws: ServerConnection) -> None:
        log.append(ws.request.headers["Authorization"] if ws.request else None)
        sent = False
        async for raw in ws:
            msg = json.loads(raw)
            log.append(msg)
            if msg["type"] == "input_audio_buffer.append" and not sent:
                sent = True
                for text in ("Xin chào", "  ", "Hello"):
                    await ws.send(json.dumps({"type": DONE, "transcript": text}))

    return handler


async def test_yields_completed_transcripts_in_spoken_language() -> None:
    log: list[Any] = []
    assert await _run(_transcribing_server(log)) == ["Xin chào", "Hello"]


async def test_authenticates_allocator_and_socket_with_hf_token() -> None:
    log: list[Any] = []
    alloc: list[httpx.Request] = []
    await _run(_transcribing_server(log), alloc_log=alloc)
    assert alloc[0].headers["X-Reachy-Mini-Authorization"] == "Bearer hf_test"
    assert log[0] == "Bearer hf_test"


async def test_session_is_transcription_only_with_auto_language() -> None:
    log: list[Any] = []
    await _run(_transcribing_server(log))
    audio_in = log[1]["session"]["audio"]["input"]
    assert log[1]["type"] == "session.update"
    assert audio_in["transcription"]["language"] == "auto"
    assert audio_in["turn_detection"] == {"type": "server_vad", "create_response": False}
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}


async def test_audio_is_framed_by_leading_and_trailing_silence() -> None:
    log: list[Any] = []
    await _run(_transcribing_server(log))
    sent = b"".join(
        base64.b64decode(m["audio"]) for m in log[1:] if m["type"] == "input_audio_buffer.append"
    )
    # 0.5 s lead-in: the server VAD misses speech that starts at sample 0. 2 s tail closes the turn.
    assert sent == b"\x00" * 24000 + b"\x01\x00" + b"\x00" * 96000


async def test_allocator_rejection_raises_connection_error() -> None:
    with pytest.raises(ConnectionError, match="401"):
        await _run(_transcribing_server([]), status=401)


async def _bad_source() -> AsyncIterator[bytes]:
    yield b"\x00\x00"
    raise ValueError("need 16-bit PCM WAV")


async def test_source_error_surfaces_without_hang() -> None:
    async def silent(ws: ServerConnection) -> None:
        async for _ in ws:
            pass

    with pytest.raises(ValueError, match="16-bit"):
        await _run(silent, source=_bad_source())


async def test_server_error_then_close_raises_with_message() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        await ws.send(json.dumps({"type": "error", "error": {"message": "session expired"}}))
        await ws.close()

    async def endless() -> AsyncIterator[bytes]:
        while True:
            yield b"\x00\x00"
            await asyncio.sleep(0.01)

    with pytest.raises(ConnectionError, match="session expired"):
        await _run(handler, source=endless())
