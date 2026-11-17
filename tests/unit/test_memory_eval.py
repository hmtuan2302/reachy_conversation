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
