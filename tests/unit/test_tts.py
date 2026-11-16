import base64
import json

import httpx
import pytest

from reachy_demo.voice.tts import TTSError, speak_stream


def _event(*pcm: bytes, text_part: bool = False) -> str:
    parts: list[dict[str, object]] = [
        {"inlineData": {"mimeType": "audio/l16; rate=24000", "data": base64.b64encode(p).decode()}}
        for p in pcm
    ]
    if text_part:
        parts.insert(0, {"text": "ignored"})
    return "data: " + json.dumps({"candidates": [{"content": {"parts": parts}}]}) + "\r\n\r\n"


async def _collect(transport: httpx.MockTransport) -> list[bytes]:
    return [c async for c in speak_stream("Xin chào", api_key="g", transport=transport)]


async def test_streams_pcm_chunks_in_order() -> None:
    seen: list[httpx.Request] = []

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        body = _event(b"\x01\x00", b"\x02\x00") + _event(text_part=True) + _event(b"\x03\x00")
        return httpx.Response(200, content=body.encode())

    assert await _collect(httpx.MockTransport(handle)) == [b"\x01\x00", b"\x02\x00", b"\x03\x00"]
    req = seen[0]
    assert str(req.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash-tts:streamGenerateContent?alt=sse"
    )
    assert req.headers["x-goog-api-key"] == "g"
    body = json.loads(req.content)
    assert body["contents"][0]["parts"][0]["text"] == "Xin chào"
    assert body["generationConfig"] == {
        "responseModalities": ["AUDIO"],
        "speechConfig": {"voiceConfig": {"voice": "Kore"}},
    }


async def test_http_error_raises_tts_error() -> None:
    def handle(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "API key not valid"}})

    with pytest.raises(TTSError, match="403.*API key not valid"):
        await _collect(httpx.MockTransport(handle))


async def test_network_error_raises_tts_error() -> None:
    def handle(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure", request=req)

    with pytest.raises(TTSError, match="dns failure"):
        await _collect(httpx.MockTransport(handle))
