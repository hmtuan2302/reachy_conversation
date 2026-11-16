import argparse
import asyncio
import io
from collections.abc import AsyncIterator
from typing import Any

import pytest

from reachy_demo.agent import cli
from reachy_demo.agent.agent import Agent
from reachy_demo.agent.llm import LLMError
from reachy_demo.voice.tts import TTSError


async def _turns(*texts: str) -> AsyncIterator[str]:
    for t in texts:
        yield t


def _agent(fail: set[str] | None = None, delay: float = 0.0) -> Agent:
    async def chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        q = messages[-1]["content"]
        await asyncio.sleep(delay)
        yield q.lower() + "1"
        if fail and q in fail:
            raise LLMError("DeepSeek returned 401")

    return Agent(chat)


async def test_converse_prints_turns_and_answers_in_order() -> None:
    out = io.StringIO()
    await cli.converse(_turns("A", "B"), _agent(), out)
    assert out.getvalue() == "you: A\nbot: a1\nyou: B\nbot: b1\n"


async def test_llm_error_prints_and_keeps_listening() -> None:
    out = io.StringIO()
    await cli.converse(_turns("A", "B"), _agent(fail={"A"}), out)
    assert out.getvalue() == ("you: A\nbot: a1 (error: DeepSeek returned 401)\nyou: B\nbot: b1\n")


async def test_turns_arriving_while_answering_are_queued() -> None:
    out = io.StringIO()
    await cli.converse(_turns("A", "B", "C"), _agent(delay=0.05), out)
    assert out.getvalue() == "you: A\nbot: a1\nyou: B\nbot: b1\nyou: C\nbot: c1\n"


async def test_source_error_propagates_after_pending_answers() -> None:
    async def broken() -> AsyncIterator[str]:
        yield "A"
        raise ConnectionError("socket closed")

    out = io.StringIO()
    with pytest.raises(ConnectionError):
        await cli.converse(broken(), _agent(), out)
    assert out.getvalue() == "you: A\nbot: a1\n"


def _args(no_speak: bool = True) -> argparse.Namespace:
    return argparse.Namespace(mic=False, stdin=False, file=None, no_speak=no_speak)


@pytest.mark.parametrize("missing", ["HF_TOKEN", "DEEPSEEK_API_KEY"])
def test_run_without_key_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.delenv(missing)
    assert cli.run(_args()) == 2
    assert missing in capsys.readouterr().err


def test_run_wires_transcripts_into_deepseek(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    seen: dict[str, Any] = {}

    async def fake_transcribe(_pcm: Any, **kw: Any) -> AsyncIterator[str]:
        seen["hf"] = kw["hf_token"]
        yield "Hello."

    async def fake_chat(messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        seen.update(kw)
        yield "Hi!"

    monkeypatch.setattr(cli, "transcribe_stream", fake_transcribe)
    monkeypatch.setattr(cli, "chat_stream", fake_chat)
    monkeypatch.setattr(cli, "source_from_args", lambda _a: _turns())
    assert cli.run(_args()) == 0
    assert capsys.readouterr().out == "you: Hello.\nbot: Hi!\n"
    assert seen["hf"] == "hf_x" and seen["api_key"] == "sk-x"
    assert seen["model"] == "deepseek-v4-pro"


async def test_speaks_each_completed_answer_but_not_failed_ones() -> None:
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    out = io.StringIO()
    await cli.converse(_turns("A", "B"), _agent(fail={"A"}), out, speak=speak)
    assert spoken == ["b1"]


async def test_speech_error_is_printed_and_listening_continues() -> None:
    async def speak(text: str) -> None:
        raise TTSError("Gemini TTS returned 429")

    out = io.StringIO()
    await cli.converse(_turns("A", "B"), _agent(), out, speak=speak)
    assert out.getvalue() == (
        "you: A\nbot: a1\n  (speech error: Gemini TTS returned 429)\n"
        "you: B\nbot: b1\n  (speech error: Gemini TTS returned 429)\n"
    )


def test_run_needs_gemini_key_only_when_speaking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cli.run(_args(no_speak=False)) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
