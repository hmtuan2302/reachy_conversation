"""DeepSeek chat client (OpenAI-compatible /chat/completions, streamed over SSE).

Verified 2026-11-16: `deepseek-chat` answers directly (fastest first token); `deepseek-flash` and
`deepseek-v4-pro` stream `reasoning_content` first. Only `content` is yielded, never the reasoning.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class LLMError(Exception):
    """The LLM call failed (HTTP status, auth, network, or a malformed stream)."""


async def chat_stream(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[str]:
    """Yield answer text pieces as they stream in."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {"model": model, "messages": messages, "stream": True}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with (
            httpx.AsyncClient(timeout=30.0, transport=transport) as http,
            http.stream("POST", url, json=body, headers=headers) as resp,
        ):
            if resp.status_code != 200:
                detail = (await resp.aread()).decode(errors="replace")[:300]
                raise LLMError(f"DeepSeek returned {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue  # blank separators and ": keep-alive" comments
                data = line[len("data: ") :]
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or [{}]
                if text := (choices[0].get("delta") or {}).get("content"):
                    yield text
    except httpx.HTTPError as e:
        raise LLMError(f"DeepSeek request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"DeepSeek sent a malformed stream: {e}") from e
