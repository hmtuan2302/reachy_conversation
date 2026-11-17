from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.agent import cli
from reachy_demo.audio.vad import EnergyVAD


def test_speaker_gate_from_env_missing_voiceprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.speaker_gate_from_env() is None
    assert "owner mode: off" in capsys.readouterr().out


def test_speaker_gate_from_env_bad_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPEAKER_HIGH", "abc")
    assert cli.speaker_gate_from_env() is None
    assert "SPEAKER_HIGH" in capsys.readouterr().out


def test_speaker_gate_from_env_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "voice").mkdir(parents=True)
    np.save(tmp_path / "data" / "voice" / "owner.npy", np.array([1.0, 0.0]))
    monkeypatch.setattr(cli, "load_sherpa_embed", lambda model: lambda x, rate: np.ones(2))
    gate = cli.speaker_gate_from_env()
    assert gate is not None and gate.high == 0.5 and gate.low == 0.3
    assert "owner mode: available" in capsys.readouterr().out


async def test_tap_feeds_gate() -> None:
    fed: list[tuple[bytes, bool]] = []

    class Gate:
        def feed(self, chunk: bytes, speaking: bool) -> None:
            fed.append((chunk, speaking))

    async def chunks() -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 100

    out = [c async for c in cli._tap(chunks(), EnergyVAD(), gate=Gate())]  # type: ignore[arg-type]
    assert out == [b"\x00\x00" * 100] and fed == [(b"\x00\x00" * 100, False)]


async def test_tap_skips_own_voice() -> None:
    fed: list[bool] = []

    class Gate:
        def feed(self, chunk: bytes, speaking: bool) -> None:
            fed.append(speaking)

        def own_voice(self, now: float) -> bool:
            return True

    async def chunks() -> AsyncIterator[bytes]:
        yield b"\xff\x7f" * 4800  # loud: VAD would call it speech

    vad = EnergyVAD(onset_ms=0)
    _ = [c async for c in cli._tap(chunks(), vad, gate=Gate())]  # type: ignore[arg-type]
    assert vad.speaking and fed == [False]


async def test_converse_tells_mode_which_turn_is_answered() -> None:
    import io

    from reachy_demo.agent.intent import Intent
    from reachy_demo.agent.tools.owner import Heard

    begun: list[tuple[str, str]] = []

    class FakeAgent:
        async def answer(self, text: str) -> AsyncIterator[str]:
            yield "ok"

    async def turns() -> AsyncIterator[str]:
        yield Heard("stop, dance", "owner")

    async def classify(text: str) -> Intent:
        return Intent(True, "dance") if text.startswith("stop") else Intent(False, "")

    def on_answer(text: str) -> None:
        begun.append((text, getattr(text, "verdict", "-")))

    await cli.converse(
        turns(),
        FakeAgent(),  # type: ignore[arg-type]
        io.StringIO(),
        classify=classify,
        moving=lambda: True,
        on_answer=on_answer,
    )
    assert begun == [("dance", "owner")]  # the stop's rest keeps the speaker's verdict


def test_normal_chat_has_no_voice_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from typing import Any

    class Gate:
        def feed(self, chunk: bytes, speaking: bool) -> None: ...
        def own_voice(self, now: float) -> bool:
            return False

        def mark_own_voice(self, until: float) -> None: ...
        def take(self) -> tuple[str, float | None]:
            return "unsure", None

    sent: list[list[dict[str, Any]]] = []

    async def fake_transcribe(_pcm: Any, **kw: Any) -> AsyncIterator[str]:
        yield "Hello."

    async def fake_chat(messages: list[dict[str, Any]], **kw: Any) -> AsyncIterator[str]:
        sent.append(messages)
        yield "Hi!"

    async def no_audio() -> AsyncIterator[bytes]:
        return
        yield b""

    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("MEMORY", "off")
    monkeypatch.setattr(cli, "speaker_gate_from_env", Gate)
    monkeypatch.setattr(cli, "transcribe_stream", fake_transcribe)
    monkeypatch.setattr(cli, "chat_events", fake_chat)
    monkeypatch.setattr(cli, "source_from_args", lambda _a: no_audio())
    args = argparse.Namespace(
        mic=False, stdin=False, file=None, type=False, no_speak=True, robot=False
    )
    assert cli.run(args) == 0
    assert sent[0][-1] == {"role": "user", "content": "Hello."}
    assert "[voice:" not in sent[0][0]["content"]
