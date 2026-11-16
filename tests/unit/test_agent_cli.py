import argparse
import asyncio
import io
from collections.abc import AsyncIterator
from typing import Any

import pytest

from reachy_demo.agent import cli
from reachy_demo.agent.agent import Agent
from reachy_demo.agent.intent import Intent
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
    return argparse.Namespace(
        mic=False, stdin=False, file=None, type=False, no_speak=no_speak, robot=False
    )


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
    monkeypatch.setattr(cli, "chat_events", fake_chat)
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


def _classify(table: dict[str, Intent], calls: list[str] | None = None):  # type: ignore[no-untyped-def]
    async def classify(text: str) -> Intent:
        if calls is not None:
            calls.append(text)
        return table.get(text, Intent(False, ""))

    return classify


async def _timed(*items: tuple[float, str]) -> AsyncIterator[str]:
    for delay, text in items:
        await asyncio.sleep(delay)
        yield text


async def test_stop_mid_answer_cancels_and_answers_rest() -> None:
    out = io.StringIO()
    cls = _classify({"stop, B": Intent(True, "B")})
    await cli.converse(_turns("A", "stop, B"), _agent(delay=0.2), out, classify=cls)
    assert out.getvalue() == "you: A\nbot: \n  (stopped)\nyou: B\nbot: b1\n"


async def test_stop_alone_answers_nothing() -> None:
    out = io.StringIO()
    cls = _classify({"stop": Intent(True, "")})
    await cli.converse(_turns("A", "stop"), _agent(delay=0.2), out, classify=cls)
    assert out.getvalue() == "you: A\nbot: \n  (stopped)\n"


async def test_non_stop_mid_answer_is_dropped() -> None:
    out = io.StringIO()
    await cli.converse(_turns("A", "hmm"), _agent(delay=0.1), out, classify=_classify({}))
    assert out.getvalue() == "you: A\nbot: a1\n  (ignored: hmm)\n"


async def test_ignored_then_stop() -> None:
    out = io.StringIO()
    cls = _classify({"stop": Intent(True, "")})
    await cli.converse(_turns("A", "hmm", "stop"), _agent(delay=0.2), out, classify=cls)
    assert out.getvalue() == "you: A\nbot: \n  (stopped)\n"


async def test_idle_turn_is_not_classified() -> None:
    out, calls = io.StringIO(), list[str]()
    turns = _timed((0, "A"), (0.1, "B"))
    await cli.converse(turns, _agent(), out, classify=_classify({}, calls))
    assert calls == []
    assert out.getvalue() == "you: A\nbot: a1\nyou: B\nbot: b1\n"


async def test_stop_during_speech_cancels_speak() -> None:
    state: list[str] = []

    async def speak(text: str) -> None:
        state.append("started")
        await asyncio.sleep(1)
        state.append("finished")

    out = io.StringIO()
    cls = _classify({"stop": Intent(True, "")})
    turns = _timed((0, "A"), (0.1, "stop"))
    await cli.converse(turns, _agent(), out, speak=speak, classify=cls)
    await asyncio.sleep(0.05)  # anything still running from the old answer would act now
    assert state == ["started"]
    assert out.getvalue() == "you: A\nbot: a1\n  (stopped)\n"


async def test_input_end_waits_for_running_answer() -> None:
    out = io.StringIO()
    await cli.converse(_turns("A"), _agent(delay=0.1), out, classify=_classify({}))
    assert out.getvalue() == "you: A\nbot: a1\n"


async def test_cancelled_answer_not_in_history() -> None:
    agent = _agent(delay=0.2)
    cls = _classify({"stop, B": Intent(True, "B")})
    await cli.converse(_turns("A", "stop, B"), agent, io.StringIO(), classify=cls)
    assert agent.history == [
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "b1"},
    ]


def test_stop_check_uses_fast_model_not_llm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")  # streams reasoning first: too slow
    models: list[str] = []

    async def fake_chat(messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        models.append(kw["model"])
        yield '{"stop": true, "rest": ""}'

    async def fake_converse(*a: Any) -> None:
        await a[4]("stop")

    monkeypatch.setattr(cli, "chat_stream", fake_chat)
    monkeypatch.setattr(cli, "converse", fake_converse)
    monkeypatch.setattr(cli, "source_from_args", lambda _a: _turns())
    assert cli.run(_args()) == 0
    assert models == ["deepseek-chat"]


def test_robot_flag_needs_robot_host(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.delenv("ROBOT_HOST", raising=False)
    args = _args()
    args.robot = True
    assert cli.run(args) == 2
    assert "ROBOT_HOST" in capsys.readouterr().err


async def test_stop_mid_answer_calls_on_stop_once() -> None:
    stops: list[str] = []
    cls = _classify({"stop": Intent(True, ""), "uh": Intent(False, "")})
    await cli.converse(
        _turns("A", "uh", "stop"),
        _agent(delay=0.2),
        io.StringIO(),
        classify=cls,
        on_stop=lambda: stops.append("x"),
    )
    assert stops == ["x"]  # side talk ("uh") does not stop the head


async def test_stop_while_idle_but_moving_stops_without_asking_the_agent() -> None:
    stops: list[str] = []
    asked: list[str] = []

    async def chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        asked.append(messages[-1]["content"])
        yield "ok"

    out = io.StringIO()
    await cli.converse(
        _turns("thôi", "hello"),
        Agent(chat),
        out,
        classify=_classify({"thôi": Intent(True, "")}),
        on_stop=lambda: stops.append("x"),
        moving=lambda: not stops,  # moving until stopped
    )
    assert stops == ["x"]
    assert asked == ["hello"]  # the stop never reached the agent
    assert out.getvalue() == "you: thôi\n  (stopped)\nyou: hello\nbot: ok\n"


async def test_non_stop_while_moving_goes_to_the_agent() -> None:
    out = io.StringIO()
    await cli.converse(_turns("A"), _agent(), out, classify=_classify({}), moving=lambda: True)
    assert out.getvalue() == "you: A\nbot: a1\n"


async def test_typed_turns_skip_blank_lines_and_end_at_eof() -> None:
    got = [
        t async for t in cli.typed_turns(io.StringIO("turn left\n\n  \ndance and tell me a joke\n"))
    ]
    assert got == ["turn left", "dance and tell me a joke"]


def test_type_mode_needs_no_hf_token_and_sends_typed_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setattr("sys.stdin", io.StringIO("Hello.\n"))

    async def fake_chat(messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        yield "Hi!"

    def no_audio(_a: Any) -> Any:
        raise AssertionError("--type must not open an audio source")

    monkeypatch.setattr(cli, "chat_events", fake_chat)
    monkeypatch.setattr(cli, "source_from_args", no_audio)
    args = _args()
    args.type = True
    assert cli.run(args) == 0
    assert capsys.readouterr().out == "you: Hello.\nbot: Hi!\n"


def test_type_flag_is_agent_only() -> None:
    import argparse as ap

    from reachy_demo.voice import transcribe_cli

    parser = ap.ArgumentParser()
    transcribe_cli.add_source_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--type"])


def test_unknown_vision_provider_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    assert cli.run(_args()) == 2
    assert "VISION_PROVIDER" in capsys.readouterr().err


def test_vision_asker_picks_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = {"HF_TOKEN": "hf_x", "DEEPSEEK_API_KEY": "sk-x"}
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("VISION_BASE_URL", raising=False)
    ask, label = cli.vision_asker("deepseek", keys, "https://api.deepseek.com")
    assert label == "deepseek deepseek-chat"
    assert ask.keywords == {  # type: ignore[attr-defined]
        "api_key": "sk-x",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    }
    monkeypatch.setenv("VISION_MODEL", "deepseek-flash")
    assert cli.vision_asker("deepseek", keys, "u")[1] == "deepseek deepseek-flash"
    ask, label = cli.vision_asker("hf", keys, "u")
    assert label == "hf realtime server"
    assert ask.keywords == {"hf_token": "hf_x"}  # type: ignore[attr-defined]


def test_type_robot_with_deepseek_vision_needs_no_hf_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda *_a: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.delenv("ROBOT_HOST", raising=False)
    monkeypatch.setenv("VISION_PROVIDER", "deepseek")
    args = _args()
    args.type, args.robot = True, True
    assert cli.run(args) == 2
    err = capsys.readouterr().err
    assert "ROBOT_HOST" in err and "HF_TOKEN" not in err
