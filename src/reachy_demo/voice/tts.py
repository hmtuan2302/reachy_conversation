"""Gemini text-to-speech, streamed (verified 2026-11-16, ai.google.dev speech-generation docs).

POST .../v1beta/models/{model}:streamGenerateContent?alt=sse, header x-goog-api-key.
Each SSE event: candidates[0].content.parts[].inlineData.data = base64 raw PCM
(audio/l16; rate=24000; channels=1). First audio ~1 s; generation is faster than real time.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import AsyncIterator

import httpx

API = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "Kore"


class TTSError(Exception):
    """Text-to-speech failed (HTTP status, auth, network, or a malformed stream)."""


async def speak_stream(
    text: str,
    *,
    api_key: str,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[bytes]:
    """Yield 24 kHz mono PCM16 chunks of `text` spoken, as Gemini generates them."""
    url = f"{API}/models/{model}:streamGenerateContent?alt=sse"
    body = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"voice": voice}},
        },
    }
    try:
        async with (
            httpx.AsyncClient(timeout=60.0, transport=transport) as http,
            http.stream("POST", url, json=body, headers={"x-goog-api-key": api_key}) as resp,
        ):
            if resp.status_code != 200:
                detail = (await resp.aread()).decode(errors="replace")[:300]
                raise TTSError(f"Gemini TTS returned {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                for cand in json.loads(line[len("data: ") :]).get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        if data := (part.get("inlineData") or {}).get("data"):
                            yield base64.b64decode(data)
    except httpx.HTTPError as e:
        raise TTSError(f"Gemini TTS request failed: {e}") from e
    except (json.JSONDecodeError, binascii.Error) as e:
        raise TTSError(f"Gemini TTS sent a malformed stream: {e}") from e
