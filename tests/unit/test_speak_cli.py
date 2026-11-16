import argparse
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from reachy_demo.voice import speak_cli


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(speak_cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("TTS_PLAYER", "none")
    monkeypatch.setenv("TTS_VOICE", "Puck")
    monkeypatch.chdir(tmp_path)


def test_speak_saves_wav_and_prints_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    async def fake(text: str, **kw: Any) -> AsyncIterator[bytes]:
        seen.update(kw, text=text)
        yield b"\x01\x00\x02\x00"

    monkeypatch.setattr(speak_cli, "speak_stream", fake)
    assert speak_cli.run(argparse.Namespace(text=["xin", "chào"])) == 0
    saved = Path(capsys.readouterr().out.strip().removeprefix("saved "))
    assert saved.parent == Path("data/speech")
    with wave.open(str(tmp_path / saved), "rb") as w:
        assert w.readframes(w.getnframes()) == b"\x01\x00\x02\x00"
    assert seen["text"] == "xin chào" and seen["voice"] == "Puck" and seen["api_key"] == "g"


def test_speak_without_key_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    assert speak_cli.run(argparse.Namespace(text=["hi"])) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
