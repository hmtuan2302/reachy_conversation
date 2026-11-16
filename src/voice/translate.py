"""Live speech translation over the OpenAI Realtime translation WebSocket.

Protocol (verified 2026-11-16, developers.openai.com/api/docs/guides/realtime-translation):
- connect  wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate
- client   session.update {session.audio.output.language}, session.input_audio_buffer.append
           {audio: base64 24 kHz PCM16 mono, ideally 200 ms chunks}, session.close (flush + end)
- server   session.output_transcript.delta {delta}, error {error.message}, session.closed
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

TRANSLATE_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

log = logging.getLogger(__name__)


async def _pump(ws: ClientConnection, pcm24k: AsyncIterator[bytes]) -> None:
    async for chunk in pcm24k:
        audio = base64.b64encode(chunk).decode()
        await ws.send(json.dumps({"type": "session.input_audio_buffer.append", "audio": audio}))
    await ws.send(json.dumps({"type": "session.close"}))


async def translate_stream(
    pcm24k: AsyncIterator[bytes], *, api_key: str, target: str, url: str = TRANSLATE_URL
) -> AsyncIterator[str]:
    """Yield translated text fragments (in `target` language) as audio streams in."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with connect(url, additional_headers=headers) as ws:
            update = {"audio": {"output": {"language": target}}}
            await ws.send(json.dumps({"type": "session.update", "session": update}))
            pump = asyncio.create_task(_pump(ws, pcm24k))
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind == "session.output_transcript.delta":
                        yield event["delta"]
                    elif kind == "error":
                        # Most translation errors are recoverable; the session stays open.
                        log.warning("translate error", extra={"error": event.get("error")})
                    elif kind == "session.closed":
                        return
                if pump.done() and pump.exception():
                    raise ConnectionError(str(pump.exception()))
                raise ConnectionError("translation socket closed before session.closed")
            finally:
                pump.cancel()
    except ConnectionClosed as e:
        raise ConnectionError(f"translation socket closed: {e}") from e
