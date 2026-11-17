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


def test_vieneu_provider_needs_no_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    async def fake(text: str, **kw: Any) -> AsyncIterator[bytes]:
        seen.update(kw, text=text)
        yield b"\x01\x00"

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("TTS_PROVIDER", "vieneu")
    monkeypatch.setenv("TTS_VOICE", "Mai Anh")
    monkeypatch.setattr(speak_cli, "vieneu_stream", fake)
    assert speak_cli.run(argparse.Namespace(text=["xin", "chào"])) == 0
    assert seen == {"text": "xin chào", "voice": "Mai Anh", "ref_audio": None}


async def test_speaker_clones_only_while_file_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Any] = []

    async def fake(text: str, **kw: Any) -> AsyncIterator[bytes]:
        seen.append(kw["ref_audio"])
        yield b"\x01\x00"

    monkeypatch.setenv("TTS_PROVIDER", "vieneu")
    monkeypatch.setattr(speak_cli, "vieneu_stream", fake)
    clone = tmp_path / "clone.wav"
    speak = speak_cli.speaker_from_env("", clone=clone)
    await speak("a")
    clone.write_bytes(b"x")
    await speak("b")
    assert seen == [None, clone]


def test_save_speech_off_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    async def fake(text: str, **kw: Any) -> AsyncIterator[bytes]:
        yield b"\x01\x00"

    monkeypatch.setenv("SAVE_SPEECH", "off")
    monkeypatch.setattr(speak_cli, "speak_stream", fake)
    assert speak_cli.run(argparse.Namespace(text=["hi"])) == 0
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "data").exists()


def test_speak_clone_audio_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    seen: list[Any] = []

    async def fake(text: str, **kw: Any) -> AsyncIterator[bytes]:
        seen.append(kw["ref_audio"])
        yield b"\x01\x00"

    fixed = tmp_path / "fixed.wav"
    monkeypatch.setenv("TTS_PROVIDER", "vieneu")
    monkeypatch.setenv("CLONE_AUDIO", str(fixed))
    monkeypatch.setattr(speak_cli, "vieneu_stream", fake)
    assert speak_cli.run(argparse.Namespace(text=["hi"])) == 2
    assert "CLONE_AUDIO file not found" in capsys.readouterr().err
    fixed.write_bytes(b"x")
    assert speak_cli.run(argparse.Namespace(text=["hi"])) == 0
    assert seen == [fixed]
