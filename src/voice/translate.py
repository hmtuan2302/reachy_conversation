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
from websockets.exceptions import ConnectionClosed, InvalidHandshake

TRANSLATE_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

log = logging.getLogger(__name__)


async def _pump(ws: ClientConnection, pcm24k: AsyncIterator[bytes]) -> None:
    try:
        async for chunk in pcm24k:
            audio = base64.b64encode(chunk).decode()
            await ws.send(json.dumps({"type": "session.input_audio_buffer.append", "audio": audio}))
    except Exception:
        await ws.close()  # unblock the reader; the caller re-raises this task's exception
        raise
    await ws.send(json.dumps({"type": "session.close"}))


async def translate_stream(
    pcm24k: AsyncIterator[bytes], *, api_key: str, target: str, url: str = TRANSLATE_URL
) -> AsyncIterator[str]:
    """Yield translated text fragments (in `target` language) as audio streams in.

    Audio-source errors are re-raised as-is; network/server failures become ConnectionError.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error = ""
    source_error: BaseException | None = None
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
                        last_error = str(event.get("error", {}).get("message", event))
                        log.warning("translate error: %s", last_error)
                    elif kind == "session.closed":
                        return
                await asyncio.wait([pump], timeout=1)
                if pump.done() and not pump.cancelled():
                    source_error = pump.exception()
                if source_error is None:
                    raise ConnectionError(f"translation socket closed. {last_error}".strip())
            finally:
                pump.cancel()
    except ConnectionError:
        raise
    except (ConnectionClosed, InvalidHandshake, OSError) as e:
        raise ConnectionError(f"translation connection failed: {e}. {last_error}".strip()) from e
    # Raised outside the network try so a local file OSError isn't reported as a network failure.
    raise source_error
