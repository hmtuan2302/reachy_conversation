"""DeepSeek chat client (OpenAI-compatible /chat/completions, streamed over SSE).

Verified 2026-11-16: `deepseek-chat` answers directly (fastest first token); `deepseek-flash` and
`deepseek-v4-pro` stream `reasoning_content` first. Only `content` is yielded, never the reasoning.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class LLMError(Exception):
    """The LLM call failed (HTTP status, auth, network, or a malformed stream)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, as the model wrote it


async def chat_stream(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[str]:
    """Yield answer text pieces as they stream in."""
    async for piece in chat_events(
        messages, api_key=api_key, model=model, base_url=base_url, transport=transport
    ):
        if isinstance(piece, str):
            yield piece


async def chat_events(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    tools: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[str | ToolCall]:
    """Yield text pieces as they stream in, then any tool calls once the stream ends."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if tools:
        body["tools"] = tools
    calls: dict[int, dict[str, str]] = {}  # tool calls stream as fragments, keyed by index
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
                    break
                choices = json.loads(data).get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                if text := delta.get("content"):
                    yield text
                for frag in delta.get("tool_calls") or []:
                    call = calls.setdefault(frag["index"], {"id": "", "name": "", "arguments": ""})
                    fn = frag.get("function") or {}
                    call["id"] = frag.get("id") or call["id"]
                    call["name"] += fn.get("name") or ""
                    call["arguments"] += fn.get("arguments") or ""
    except httpx.HTTPError as e:
        raise LLMError(f"DeepSeek request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"DeepSeek sent a malformed stream: {e}") from e
    except KeyError as e:
        raise LLMError(f"DeepSeek sent a tool call without {e}") from e
    for _, call in sorted(calls.items()):
        yield ToolCall(call["id"], call["name"], call["arguments"])
