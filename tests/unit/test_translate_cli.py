import argparse
import io
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from reachy_demo.voice import translate_cli


async def _gen(*parts: str) -> AsyncIterator[str]:
    for p in parts:
        yield p


def _args(file: Path | None = None, mic: bool = False, to: str = "en") -> argparse.Namespace:
    return argparse.Namespace(file=file, mic=mic, to=to)


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 480)
    return path


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translate_cli, "load_dotenv", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


async def test_render_writes_fragments_then_newline() -> None:
    out = io.StringIO()
    await translate_cli.render(_gen("Xin ", "chào"), out)
    assert out.getvalue() == "Xin chào\n"


def test_run_without_api_key_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    assert translate_cli.run(_args(file=_wav(tmp_path))) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_run_missing_file_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert translate_cli.run(_args(file=tmp_path / "nope.wav")) == 2
    assert "nope.wav" in capsys.readouterr().err


def test_run_stream_error_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def boom(*_a: Any, **_k: Any) -> AsyncIterator[str]:
        raise ConnectionError("socket closed")
        yield ""

    monkeypatch.setattr(translate_cli, "translate_stream", boom)
    assert translate_cli.run(_args(file=_wav(tmp_path))) == 1


def test_run_file_prints_translation_in_target_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    async def fake(pcm: AsyncIterator[bytes], **kw: Any) -> AsyncIterator[str]:
        seen.update(kw, audio=b"".join([c async for c in pcm]))
        yield "Xin chào"

    monkeypatch.setattr(translate_cli, "translate_stream", fake)
    assert translate_cli.run(_args(file=_wav(tmp_path), to="vi")) == 0
    assert capsys.readouterr().out == "Xin chào\n"
    assert seen["target"] == "vi" and seen["api_key"] == "sk-test" and len(seen["audio"]) == 960
