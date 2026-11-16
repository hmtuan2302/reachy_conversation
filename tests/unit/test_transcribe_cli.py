import argparse
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from reachy_demo.voice import transcribe_cli


def _args(file: Path | None = None, mic: bool = False) -> argparse.Namespace:
    return argparse.Namespace(file=file, mic=mic)


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 480)
    return path


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe_cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_test")


def test_run_prints_one_line_per_utterance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    async def fake(pcm: AsyncIterator[bytes], **kw: Any) -> AsyncIterator[str]:
        seen.update(kw, audio=b"".join([c async for c in pcm]))
        yield "Xin chào"
        yield "Hello"

    monkeypatch.setattr(transcribe_cli, "transcribe_stream", fake)
    assert transcribe_cli.run(_args(file=_wav(tmp_path))) == 0
    assert capsys.readouterr().out == "Xin chào\nHello\n"
    assert seen["hf_token"] == "hf_test" and len(seen["audio"]) == 960


def test_run_without_hf_token_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN")
    assert transcribe_cli.run(_args(file=_wav(tmp_path))) == 2
    assert "HF_TOKEN" in capsys.readouterr().err


def test_run_missing_file_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert transcribe_cli.run(_args(file=tmp_path / "nope.wav")) == 2
    assert "nope.wav" in capsys.readouterr().err


def test_run_bad_wav_returns_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake(pcm: AsyncIterator[bytes], **_kw: Any) -> AsyncIterator[str]:
        async for _ in pcm:
            pass
        yield ""

    monkeypatch.setattr(transcribe_cli, "transcribe_stream", fake)
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not a wav")
    assert transcribe_cli.run(_args(file=bad)) == 2
    assert "not a readable WAV" in capsys.readouterr().err


def test_run_connection_error_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def boom(*_a: Any, **_k: Any) -> AsyncIterator[str]:
        raise ConnectionError("allocator returned 401")
        yield ""

    monkeypatch.setattr(transcribe_cli, "transcribe_stream", boom)
    assert transcribe_cli.run(_args(file=_wav(tmp_path))) == 1
