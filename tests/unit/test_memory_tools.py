import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from reachy_demo.agent.tools import memory as mem_tools
from reachy_demo.agent.tools.memory import MemoryTools
from reachy_demo.memory.consolidate import Consolidator
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore, Speaker

NOW = datetime(2026, 11, 17, 19, 0, tzinfo=timezone(timedelta(hours=7)))
GUEST = Speaker(2, "guest")


class Chat:
    def __init__(self, reply: dict[str, Any], delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay

    async def __call__(self, _m: list[dict[str, str]]) -> AsyncIterator[str]:
        await asyncio.sleep(self.delay)
        yield json.dumps(self.reply)


EXTRACT = {
    "facts": [
        {
            "subject": "Tuan",
            "text": "Tuan's favourite drink is iced coffee.",
            "category": "preference",
            "importance": 0.6,
            "sensitive": False,
        }
    ],
    "follow_ups": [],
    "name": None,
    "language": None,
}


def _tools(speaker: Speaker = OWNER, chat: Chat | None = None) -> tuple[MemoryTools, MemoryStore]:
    store = MemoryStore(":memory:", lambda: NOW.timestamp())
    store._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    cons = Consolidator(store, chat or Chat(EXTRACT), lambda: NOW.timestamp())
    return MemoryTools(store, cons, speaker=lambda: speaker, now=lambda: NOW), store


def _add(store: MemoryStore, text: str) -> int:
    return store.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text=text)


async def _call(t: MemoryTools, name: str, **args: Any) -> dict[str, Any]:
    return await t.call(name, json.dumps(args))


def test_specs_names() -> None:
    t, _ = _tools()
    assert [s["function"]["name"] for s in t.specs()] == ["recall", "remember", "forget"]
    assert all(t.needs_response(n, "{}") for n in ("recall", "remember", "forget"))


async def test_recall_through_tool() -> None:
    t, store = _tools()
    _add(store, "Tuan has a sister named Mai.")
    r = await _call(t, "recall", query="sister")
    assert r["found"] is True and "Mai" in r["results"][0]


@pytest.mark.parametrize("args", ["not json", "[1]", json.dumps({"query": 3})])
async def test_bad_arguments(args: str) -> None:
    t, _ = _tools()
    assert "error" in await t.call("recall", args)


async def test_remember_stores_then_reports_already_known() -> None:
    t, store = _tools()
    assert await _call(t, "remember", fact="my favourite drink is iced coffee") == {
        "ok": True,
        "stored": ["Tuan's favourite drink is iced coffee."],
    }
    assert len(store.search_facts("coffee", OWNER)) == 1


async def test_slow_remember_returns_and_finishes_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mem_tools, "REMEMBER_WAIT_S", 0.01)
    t, store = _tools(chat=Chat(EXTRACT, delay=0.05))
    assert (await _call(t, "remember", fact="iced coffee"))["note"] == "saving in the background"
    await asyncio.sleep(0.1)
    assert len(store.search_facts("coffee", OWNER)) == 1


@pytest.mark.parametrize(
    ("name", "args"), [("remember", {"fact": "x"}), ("forget", {"query": "x"})]
)
async def test_guest_cannot_write_or_delete(name: str, args: dict[str, Any]) -> None:
    t, _ = _tools(GUEST)
    assert await _call(t, name, **args) == mem_tools.NOT_OWNER


async def test_forget_single_match_deletes_it_and_its_turn() -> None:
    t, store = _tools()
    eid = store.log_event("turn", {"user": "my PIN is 1234"}, person_id=OWNER_ID, session_id="s")
    store.add_fact(
        subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan's PIN is 1234.", evidence=[eid]
    )
    assert await _call(t, "forget", query="PIN") == {"ok": True, "forgot": ["Tuan's PIN is 1234."]}
    assert store.search_facts("PIN", OWNER) == []
    assert store.last_turns(5) == []


async def test_forget_several_asks_then_deletes_by_id() -> None:
    t, store = _tools()
    a = _add(store, "Tuan likes green tea.")
    _add(store, "Tuan's sister likes black tea.")
    r = await _call(t, "forget", query="tea")
    assert len(r["matches"]) == 2
    assert await _call(t, "forget", ids=[a]) == {"ok": True, "forgot_count": 1}
    assert [f.text for f, _ in store.search_facts("tea", OWNER)] == [
        "Tuan's sister likes black tea."
    ]


async def test_forget_nothing_found() -> None:
    t, _ = _tools()
    assert await _call(t, "forget", query="dog") == {"found": False}


async def test_forget_everything_needs_confirmation() -> None:
    t, store = _tools()
    _add(store, "Tuan likes green tea.")
    assert (await _call(t, "forget", everything=True))["confirm_needed"] is True
    assert len(store.search_facts("tea", OWNER)) == 1
    assert (await _call(t, "forget", everything=True, confirmed=True))["ok"] is True
    assert store.search_facts("tea", OWNER) == []


async def test_recall_tool_uses_vectors_and_falls_back_when_embedder_fails() -> None:
    from reachy_demo.memory.vectors import VectorIndex
    from tests.unit.fake_embedder import FakeEmbedder

    emb = FakeEmbedder({"claude": "aitool", "tool": "aitool"})
    t, store = _tools()
    fid = _add(store, "Tuan codes with Claude.")
    idx = VectorIndex(emb.model)
    idx.put("fact", fid, emb.embed(["Tuan codes with Claude."], "doc")[0])
    hybrid = MemoryTools(store, t._consolidator, speaker=lambda: OWNER, now=lambda: NOW,
                         embedder=emb, index=idx)  # fmt: skip
    assert (await _call(hybrid, "recall", query="tool"))["found"] is True
    broken = MemoryTools(store, t._consolidator, speaker=lambda: OWNER, now=lambda: NOW,
                         embedder=FakeEmbedder(fail=True), index=idx)  # fmt: skip
    r = await _call(broken, "recall", query="Claude")
    assert r["found"] is True and "error" not in r  # keyword path still answers


def test_recall_spec_asks_for_standalone_queries() -> None:
    from reachy_demo.agent.tools.memory import SPECS

    assert "standalone query" in SPECS[0]["function"]["description"]
