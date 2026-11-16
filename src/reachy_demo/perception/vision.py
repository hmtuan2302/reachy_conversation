"""Ask a vision-language model about one camera frame. Two providers (VISION_PROVIDER):

- `hf` (`ask_image_hf`): the Reachy Hugging Face realtime server, the same server as
  speech-to-text (voice/transcribe.py) and as reachy_mini_conversation_app's `camera` tool. The
  frame goes into a realtime session as an `input_image` item with the question, and the answer
  is read as text. Spike 2026-11-16: test PNG described correctly, answer after 1.3 s, +0.9 s to
  allocate the session.
- `deepseek` (`ask_image_chat`): any OpenAI-compatible /chat/completions with `image_url` content.
  Tested 2026-11-16 on DeepSeek: `deepseek-chat` (served as deepseek-flash, no reasoning) 1.1 s
  and `deepseek-flash` 1.8 s answer correctly; `deepseek-v4-pro` can't see images.

HF protocol: session.update {output_modalities: ["text"]}; conversation.item.create {message, user,
[input_image data URL, input_text]}; response.create. The server still streams the answer as
`response.output_audio_transcript.delta` (and audio, ignored); `response.output_text.delta` is
read too. `response.done` ends it, `error` fails it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from reachy_demo.voice.transcribe import SESSION_URL, allocate_session

log = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are the eyes of a small robot. You get one photo from its head camera and a question. "
    "Answer the question about the photo in one or two short, factual sentences. "
    "Name objects plainly (for example: a white ceramic mug). "
    "If the photo is too dark or blurry, or you cannot see what is asked, say so."
)
TEXT_DELTAS = {"response.output_audio_transcript.delta", "response.output_text.delta"}


class VisionError(Exception):
    """The vision question failed (allocator, connection, server error, timeout, empty answer)."""


async def ask_image_chat(
    jpeg: bytes,
    question: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 15.0,
) -> str:
    """Return the answer from an OpenAI-compatible chat model that accepts images."""
    image = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"
    body = {
        "model": model,
        "max_tokens": 2000,  # reasoning models think first; a small cap leaves the answer empty
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": question},
                ],
            },
        ],
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as http:
            resp = await http.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.TimeoutException as e:
        raise VisionError(f"vision answer took over {timeout_s:.0f} s") from e
    except httpx.HTTPError as e:
        raise VisionError(f"vision connection failed: {e}") from e
    if resp.status_code != 200:
        raise VisionError(f"vision model returned {resp.status_code}: {resp.text[:200]}")
    try:
        answer = str(resp.json()["choices"][0]["message"].get("content") or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise VisionError(f"vision model sent an unexpected reply: {e}") from e
    if not answer:
        raise VisionError(f"vision model {model} gave an empty answer (does it accept images?)")
    log.info("vision answer (%s): %s", model, answer)
    return answer


async def ask_image_hf(
    jpeg: bytes,
    question: str,
    *,
    hf_token: str,
    session_url: str = SESSION_URL,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 15.0,
) -> str:
    """Return the model's answer to `question` about the JPEG `jpeg`."""
    image = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"
    try:
        async with asyncio.timeout(timeout_s):
            # shortcut: one new session per question (+0.9 s); keep one open if it feels slow.
            url = await allocate_session(session_url, hf_token, transport)
            headers = {"Authorization": f"Bearer {hf_token}"}
            async with connect(url, additional_headers=headers, max_size=None) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "instructions": INSTRUCTIONS,
                                "output_modalities": ["text"],
                            },
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_image", "image_url": image},
                                    {"type": "input_text", "text": question},
                                ],
                            },
                        }
                    )
                )
                await ws.send(json.dumps({"type": "response.create"}))
                parts: list[str] = []
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind in TEXT_DELTAS:
                        parts.append(str(event.get("delta") or ""))
                    elif kind == "error":
                        message = event.get("error", {}).get("message", event)
                        raise VisionError(f"vision server error: {message}")
                    elif kind == "response.done":
                        break
    except TimeoutError as e:
        raise VisionError(f"vision answer took over {timeout_s:.0f} s") from e
    except (ConnectionError, ConnectionClosed, InvalidHandshake, OSError, httpx.HTTPError) as e:
        raise VisionError(f"vision connection failed: {e}") from e
    answer = "".join(parts).strip()
    if not answer:
        raise VisionError("vision server gave an empty answer")
    log.info("vision answer: %s", answer)
    return answer
