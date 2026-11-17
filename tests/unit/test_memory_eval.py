import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from reachy_demo.agent.llm import ToolCall
from reachy_demo.agent.memory_eval import evaluate, norm, score


def test_norm_and_score() -> None:
    assert norm("Tuấn Đà Lạt") == "tuan da lat"
    assert score("Em gái bạn tên là Mai.", {"expect": ["mai"]})
    assert not score("I don't know.", {"expect": ["Mai"]})
    assert score("Mình không nhớ bạn đã kể chuyện đó.", {"expect_none": True})
    assert not score("Your dog is Bin.", {"expect_none": True})


def _agent_chat(specs: list[dict[str, Any]] | None) -> Any:
    """Calls recall on a question, answers from its result; otherwise says 'ok'."""

    async def chat(messages: list[dict[str, Any]]) -> AsyncIterator[str | ToolCall]:
        last = messages[-1]
        if last["role"] == "tool":
            yield last["content"]
        elif specs and last["content"].endswith("?"):
            yield ToolCall("1", "recall", json.dumps({"query": last["content"]}))
        else:
            yield "ok"

    return chat


async def _memory_chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    if messages[0]["content"].startswith("Summarize"):
        yield json.dumps({"summary": "Talked about the dog.", "topics": [], "importance": 0.5})
    elif "Speaker said" in messages[-1]["content"]:
        said = messages[-1]["content"]
        facts = [{"subject": "Tuan", "text": "Tuan has a dog named Bin."}] if "Bin" in said else []
        yield json.dumps({"facts": facts, "follow_ups": [], "name": None})
    else:
        yield json.dumps({"action": "ADD"})


STEPS: list[dict[str, Any]] = [
    {"t": "2026-11-01T20:00:00+07:00", "say": "My dog is called Bin"},
    {"restart": True},
    {"t": "2026-11-09T20:00:00+07:00", "ask": "dog name?", "expect": ["Bin"], "category": "cross"},
    {"t": "2026-11-09T20:01:00+07:00", "ask": "favourite colour?", "expect_none": True,
     "category": "abstain"},
]  # fmt: skip


async def test_memory_run_scores_across_restart(tmp_path: Path) -> None:
    report = await evaluate(STEPS, _agent_chat, _memory_chat, tmp_path / "e.db")
    assert dict(report.results) == {"cross": [True], "abstain": [False]}  # fake never abstains
    assert report.turns == 3 and report.memory_calls >= 2 and report.briefing_ms
    assert "overall        1/2  50%" in report.lines()


async def test_baseline_keeps_history_and_has_no_memory_calls(tmp_path: Path) -> None:
    report = await evaluate(STEPS, _agent_chat, None, None)
    assert report.memory_calls == 0 and report.briefing_ms == []
    assert report.results["cross"] == [False]  # the fake agent can only answer through recall


def _db(tmp_path: Path) -> Path:
    from reachy_demo.memory.store import MemoryStore

    db = tmp_path / "m.db"
    MemoryStore(db).backup(tmp_path / "m.db.bak")
    (tmp_path / "m.db-wal").write_bytes(b"x")
    (tmp_path / "other.db").write_bytes(b"keep me")
    return db


def test_reset_asks_then_deletes_db_but_keeps_backups(tmp_path: Path) -> None:
    from reachy_demo.agent.memory_eval import reset

    db = _db(tmp_path)
    assert reset(db, everything=False, yes=False, ask=lambda _q: "y") == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.db.bak", "other.db"]


def test_reset_all_deletes_backups_too(tmp_path: Path) -> None:
    from reachy_demo.agent.memory_eval import reset

    db = _db(tmp_path)
    assert reset(db, everything=True, yes=True) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["other.db"]


def test_reset_cancelled_keeps_everything(tmp_path: Path) -> None:
    from reachy_demo.agent.memory_eval import reset

    db = _db(tmp_path)
    assert reset(db, everything=True, yes=False, ask=lambda _q: "") == 1
    assert db.exists() and (tmp_path / "m.db.bak").exists()


def test_reset_missing_db_is_fine(tmp_path: Path) -> None:
    from reachy_demo.agent.memory_eval import reset

    assert reset(tmp_path / "none.db", everything=False, yes=True) == 0


def _embed_args(db: Path) -> Any:
    import argparse

    return argparse.Namespace(memory_command="embed", db=str(db))


def test_memory_embed_command_counts(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    from reachy_demo.agent import memory_eval
    from reachy_demo.memory.store import OWNER_ID, MemoryStore
    from tests.unit.fake_embedder import FakeEmbedder

    db = tmp_path / "m.db"
    s = MemoryStore(db, lambda: 0.0)
    for i in range(3):
        s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text=f"fact {i}")
    s.close()
    monkeypatch.setattr(memory_eval, "load_dotenv", lambda *_a: False)
    monkeypatch.setattr(memory_eval, "load_embedder", lambda _env: (FakeEmbedder(), "hybrid"))
    assert memory_eval.run(_embed_args(db)) == 0
    assert "memory embed: 3 facts, 0 episodes (fake@256)" in capsys.readouterr().out


def test_memory_embed_without_embedder_returns_2(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from reachy_demo.agent import memory_eval

    monkeypatch.setattr(memory_eval, "load_dotenv", lambda *_a: False)
    monkeypatch.setattr(
        memory_eval, "load_embedder", lambda _env: (None, "keywords only (EMBED=off)")
    )
    assert memory_eval.run(_embed_args(tmp_path / "m.db")) == 2
    assert "EMBED=off" in capsys.readouterr().err


def _eval_args(tmp_path: Path, baseline: bool = False) -> Any:
    import argparse

    f = tmp_path / "s.jsonl"
    f.write_text('{"say": "hi"}\n')
    return argparse.Namespace(memory_command="eval", file=str(f), baseline=baseline)


def _no_network(monkeypatch: Any) -> list[Any]:
    from reachy_demo.agent import memory_eval

    seen: list[Any] = []

    async def fake_evaluate(*a: Any, **kw: Any) -> Any:
        seen.append(kw.get("embedder"))
        return memory_eval.Report()

    monkeypatch.setattr(memory_eval, "load_dotenv", lambda *_a: False)
    monkeypatch.setattr(memory_eval, "evaluate", fake_evaluate)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    return seen


def test_eval_needs_embedder_unless_embed_off(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from reachy_demo.agent import memory_eval

    _no_network(monkeypatch)
    monkeypatch.delenv("EMBED", raising=False)
    monkeypatch.setattr(
        memory_eval, "load_embedder",
        lambda _env: (None, "keywords only (install: uv sync --extra embed)"),
    )  # fmt: skip
    assert memory_eval.run(_eval_args(tmp_path)) == 2
    assert "uv sync --extra embed" in capsys.readouterr().err
    monkeypatch.setenv("EMBED", "off")
    monkeypatch.setattr(
        memory_eval, "load_embedder", lambda _env: (None, "keywords only (EMBED=off)")
    )
    assert memory_eval.run(_eval_args(tmp_path)) == 0


def test_baseline_runs_without_embedder(tmp_path: Path, monkeypatch: Any) -> None:
    from reachy_demo.agent import memory_eval

    seen = _no_network(monkeypatch)

    def never(_env: Any) -> Any:
        raise AssertionError("baseline must not load the embedder")

    monkeypatch.setattr(memory_eval, "load_embedder", never)
    assert memory_eval.run(_eval_args(tmp_path, baseline=True)) == 0
    assert seen == [None]


def test_report_prints_embed_timing() -> None:
    from reachy_demo.agent.memory_eval import Report

    lines = Report(embed_ms=[10.0, 20.0, 30.0]).lines()
    assert any(line.startswith("query embed ms") and "p50 20.0" in line for line in lines)


async def test_memory_run_with_embedder_times_queries(tmp_path: Path) -> None:
    from tests.unit.fake_embedder import FakeEmbedder

    report = await evaluate(
        STEPS, _agent_chat, _memory_chat, tmp_path / "e.db", embedder=FakeEmbedder()
    )
    assert report.results["cross"] == [True] and report.embed_ms


def test_score_expect_absent() -> None:
    step = {"expect_absent": ["Mai", "sister"]}
    assert score("Sounds fun, enjoy the match!", step)
    assert not score("Your sister Mai would like that", step)
    assert score("The main idea is in this email", step)  # whole words only
    both = {"expect": ["Arsenal"], "expect_absent": ["Copilot"]}
    assert score("Arsenal again?", both) and not score("Arsenal and Copilot", both)


async def test_memory_run_with_embedder_times_push(tmp_path: Path) -> None:
    from tests.unit.fake_embedder import FakeEmbedder

    report = await evaluate(
        STEPS, _agent_chat, _memory_chat, tmp_path / "e.db", embedder=FakeEmbedder()
    )
    assert report.push_ms
    assert any(line.startswith("push ms") for line in report.lines())
