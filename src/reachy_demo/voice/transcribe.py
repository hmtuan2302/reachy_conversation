"""Live speech-to-text (in the spoken language) on the Reachy Hugging Face realtime server.

Protocol (verified by spike 2026-11-16; mirrors reachy_mini_conversation_app):
- POST SESSION_URL, header X-Reachy-Mini-Authorization: Bearer <HF_TOKEN>, body {} -> {connect_url}
- WS connect_url, header Authorization: Bearer <HF_TOKEN>; OpenAI-compatible realtime events
- client  session.update (24 kHz PCM in, transcription language "auto",
          server_vad with create_response=false), input_audio_buffer.append {audio: base64}
- server  conversation.item.input_audio_transcription.completed {transcript}, error {error.message}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

SESSION_URL = "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session"
RATE = 24_000
# The server VAD misses speech that starts at sample 0 (a 1.7 s "hello" clip gave no text),
# so lead with 0.5 s of silence. The 2 s tail lets it close the last utterance.
LEAD_SILENCE = b"\x00\x00" * (RATE // 2)
TAIL_SILENCE = b"\x00\x00" * RATE * 2

log = logging.getLogger(__name__)

SESSION = {
    "type": "realtime",
    "instructions": "Transcribe only.",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": RATE},
            "transcription": {"model": "gpt-4o-transcribe", "language": "auto"},
            "turn_detection": {"type": "server_vad", "create_response": False},
        }
    },
}


async def allocate_session(
    session_url: str, token: str, transport: httpx.AsyncBaseTransport | None
) -> str:
    headers = {"User-Agent": "reachy-demo", "X-Reachy-Mini-Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=15.0, transport=transport) as http:
        resp = await http.post(session_url, headers=headers, json={})
    if resp.status_code != 200:
        raise ConnectionError(f"session allocator returned {resp.status_code}: {resp.text[:200]}")
    url = resp.json().get("connect_url")
    if not isinstance(url, str) or not url:
        raise ConnectionError("session allocator gave no connect_url")
    return url


async def _send_audio(ws: ClientConnection, chunk: bytes) -> None:
    audio = base64.b64encode(chunk).decode()
    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))


async def _pump(ws: ClientConnection, pcm24k: AsyncIterator[bytes]) -> None:
    try:
        await _send_audio(ws, LEAD_SILENCE)
        async for chunk in pcm24k:
            await _send_audio(ws, chunk)
    except Exception:
        await ws.close()  # unblock the reader; the caller re-raises this task's exception
        raise
    await _send_audio(ws, TAIL_SILENCE)


async def transcribe_stream(
    pcm24k: AsyncIterator[bytes],
    *,
    hf_token: str,
    session_url: str = SESSION_URL,
    transport: httpx.AsyncBaseTransport | None = None,
    idle_s: float = 5.0,
) -> AsyncIterator[str]:
    """Yield one transcript per utterance. Ends `idle_s` after the source runs out.

    Audio-source errors are re-raised as-is; network/server failures become ConnectionError.
    """
    last_error = ""
    source_error: BaseException | None = None
    try:
        url = await allocate_session(session_url, hf_token, transport)
        headers = {"Authorization": f"Bearer {hf_token}"}
        async with connect(url, additional_headers=headers, max_size=None) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": SESSION}))
            pump = asyncio.create_task(_pump(ws, pcm24k))
            recv: asyncio.Task[Any] | None = None
            try:
                while True:
                    recv = recv or asyncio.create_task(ws.recv())
                    waiting: set[asyncio.Task[Any]] = {recv} if pump.done() else {recv, pump}
                    done, _ = await asyncio.wait(
                        waiting,
                        timeout=idle_s if pump.done() else None,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        return  # source finished and the server has gone quiet
                    if (source_error := await _source_failure(pump, done)) is not None:
                        break
                    if recv not in done:
                        continue
                    try:
                        raw = recv.result()
                    except ConnectionClosed:
                        # The pump closes the socket when the source fails: report that cause.
                        if (source_error := await _source_failure(pump, {pump})) is not None:
                            break
                        raise
                    event = json.loads(raw)
                    recv = None
                    kind = event.get("type")
                    if kind == "conversation.item.input_audio_transcription.completed":
                        if text := str(event.get("transcript") or "").strip():
                            yield text
                    elif kind == "error":
                        last_error = str(event.get("error", {}).get("message", event))
                        log.warning("transcribe error: %s", last_error)
            finally:
                pump.cancel()
                if recv:
                    recv.cancel()
    except ConnectionError:
        raise
    except (ConnectionClosed, InvalidHandshake, OSError, httpx.HTTPError) as e:
        raise ConnectionError(f"transcription connection failed: {e}. {last_error}".strip()) from e
    # Raised outside the network try so a local file OSError isn't reported as a network failure.
    if source_error is not None:
        raise source_error


async def _source_failure(
    pump: asyncio.Task[None], done: set[asyncio.Task[Any]]
) -> BaseException | None:
    """The pump's exception if it failed (waits briefly for it to settle when asked about it)."""
    if pump not in done:
        return None
    await asyncio.wait({pump}, timeout=1)
    exc = pump.exception() if pump.done() and not pump.cancelled() else None
    return None if isinstance(exc, ConnectionClosed) else exc  # socket loss is not a source error
