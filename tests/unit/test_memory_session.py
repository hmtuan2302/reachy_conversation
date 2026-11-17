import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from reachy_demo.memory.consolidate import Consolidator
from reachy_demo.memory.episodes import summarize_session
from reachy_demo.memory.session import MemorySession
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore

T0 = 1_763_380_000.0


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> float:
        return self.t


class RoutingChat:
    """Answers memory prompts by kind: extract → no facts, summarize → a summary."""

    def __init__(self, summary: str = "Talked about the trip to Da Lat.") -> None:
        self.summary = summary
        self.kinds: list[str] = []

    async def __call__(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        system = messages[0]["content"]
        if system.startswith("Summarize"):
            self.kinds.append("summarize")
            yield json.dumps(
                {
                    "summary": self.summary,
                    "topics": ["trip"],
                    "importance": 0.6,
                    "follow_ups": [{"text": "Ask how Da Lat was.", "due": None}],
                }
            )
        else:
            self.kinds.append("extract")
            yield json.dumps({"facts": [], "follow_ups": [], "name": None})


def _turn(user: str, answer: str = "ok") -> list[dict[str, Any]]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def _session(**kw: Any) -> tuple[MemorySession, MemoryStore, Clock, RoutingChat]:
    clock = Clock()
    store = MemoryStore(":memory:", clock)
    chat = RoutingChat()
    s = MemorySession(store, Consolidator(store, chat, clock), chat, clock=clock, **kw)
    return s, store, clock, chat


async def _run_jobs(s: MemorySession) -> None:
    worker = asyncio.create_task(s.run())
    await asyncio.wait_for(s.drain(), 2)
    worker.cancel()


async def test_briefing_built_once_per_session_and_rebuilt_after_gap() -> None:
    s, store, clock, _ = _session()
    first = s.context()
    sid = s.session_id
    store.set_name(OWNER_ID, "Tuan")
    clock.t += 300
    assert s.context() == first and s.session_id == sid  # cached inside the session
    clock.t += 601
    assert "Tuan" in s.context() and s.session_id != sid


async def test_gap_ends_session_and_summarizes_it() -> None:
    s, store, clock, chat = _session()
    s.context()
    s.submit(_turn("we are planning a trip to Da Lat next month with my sister"))
    old = s.session_id
    clock.t += 601
    s.context()
    await _run_jobs(s)
    eps = store.recent_episodes(OWNER_ID, 5)
    assert [(e.session_id, e.summary) for e in eps] == [(old, "Talked about the trip to Da Lat.")]
    assert chat.kinds == ["extract", "summarize"]  # turn consolidated before the summary
    assert [f.text for f in store.all_open_follow_ups(OWNER_ID)] == ["Ask how Da Lat was."]


async def test_idle_tick_ends_session_without_a_new_turn() -> None:
    s, _, clock, _ = _session()
    s.context()
    clock.t += 601
    s.tick()
    assert s.session_id is None


async def test_small_talk_is_logged_but_not_sent_to_the_llm() -> None:
    s, store, _, chat = _session()
    s.submit(_turn("hello"))
    s.submit(_turn("turn left"))
    await _run_jobs(s)
    assert chat.kinds == []
    assert len(store.last_turns(5)) == 2
    assert store.unconsolidated_turns(0, 10) == []


async def test_submit_records_tool_names() -> None:
    s, store, _, _ = _session()
    turn: list[dict[str, Any]] = [
        {"role": "user", "content": "turn left"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "turn_head", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
    ]
    s.submit(turn)
    assert store.last_turns(1)[0][1] == {"user": "turn left", "answer": "", "tools": ["turn_head"]}


async def test_full_queue_drops_oldest() -> None:
    s, _, _, _ = _session(maxsize=2)
    for i in range(4):
        s.submit(_turn(f"I have a sister number {i} who likes tea"))
    assert s.dropped == 2
    assert [j[2] for j in list(s._queue._queue)] == [  # type: ignore[attr-defined]
        "I have a sister number 2 who likes tea",
        "I have a sister number 3 who likes tea",
    ]


async def test_crash_recovery_summarizes_old_session_and_requeues_turns() -> None:
    clock = Clock()
    store = MemoryStore(":memory:", clock)
    store.log_event(
        "turn",
        {"user": "we are planning a trip to Da Lat next month together", "answer": "nice"},
        person_id=OWNER_ID,
        session_id="dead",
    )
    chat = RoutingChat()
    s = MemorySession(store, Consolidator(store, chat, clock), chat, clock=clock)
    await _run_jobs(s)
    assert chat.kinds == ["summarize", "extract"]
    assert [e.session_id for e in store.recent_episodes(OWNER_ID, 5)] == ["dead"]
    assert store.unconsolidated_turns(0, 10) == []


async def test_close_summarizes_open_session_and_backs_up(tmp_path: Any) -> None:
    clock = Clock()
    store = MemoryStore(tmp_path / "m.db", clock)
    chat = RoutingChat()
    s = MemorySession(store, Consolidator(store, chat, clock), chat, clock=clock)
    worker = asyncio.create_task(s.run())
    s.submit(_turn("we are planning a trip to Da Lat next month together"))
    await s.close(worker)
    assert len(store.recent_episodes(OWNER_ID, 5)) == 1
    assert (tmp_path / "m.db.bak").exists()
    assert worker.cancelled()


async def test_preload_only_after_a_short_gap() -> None:
    s, _, clock, _ = _session()
    s.submit(_turn("q1", "a1"))
    clock.t += 3600
    assert s.preload() == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    clock.t += 7200
    assert s.preload() == []


async def test_short_greeting_session_needs_no_llm_and_stays_out_of_briefing() -> None:
    clock = Clock()
    store = MemoryStore(":memory:", clock)
    store.log_event("turn", {"user": "hi there"}, person_id=OWNER_ID, session_id="g")
    chat = RoutingChat()
    assert await summarize_session(store, chat, "g", OWNER_ID, clock) is not None
    assert chat.kinds == []
    assert store.recent_episodes(OWNER_ID, 5) == []
    assert store.sessions_without_episode(None) == []  # not retried every startup
    assert await summarize_session(store, chat, "g", OWNER_ID, clock) is None  # idempotent


async def test_guest_turns_are_consolidated_under_the_guest() -> None:
    from reachy_demo.memory.store import Speaker

    s, store, _, chat = _session(speaker=lambda: Speaker(OWNER.person_id + 1, "guest"))
    store._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    s.submit(_turn("I have a sister named Mai who likes tea"))
    await _run_jobs(s)
    assert "extract" in chat.kinds  # facts land under the guest (tested in test_consolidate)


async def test_short_yes_closes_the_follow_up_its_reply_settles() -> None:
    s, store, _, chat = _session()
    store.set_name(OWNER_ID, "Tuan")
    fu = store.add_follow_up(
        OWNER_ID, "Ask whether Tuan wants the robot to switch back to its own voice.", None
    )
    other = store.add_follow_up(OWNER_ID, "Ask how the math exam went.", None)
    s.submit(_turn("yes", "All set, I'm back to my own voice. Anything else, Tuan?"))
    assert [f.id for f in store.all_open_follow_ups(OWNER_ID)] == [other]
    assert fu != other and chat.kinds == []  # "yes" is still not sent to extraction


def test_session_passes_memory_max_tokens(monkeypatch: Any) -> None:
    seen: dict[str, int] = {}

    def fake(*_a: Any, cap_tokens: int) -> Any:
        from reachy_demo.memory.briefing import Briefing

        seen["cap"] = cap_tokens
        return Briefing("[memory]", frozenset(), frozenset())

    monkeypatch.setattr("reachy_demo.memory.session.build_briefing_parts", fake)
    s, *_ = _session(memory_max_tokens=123)
    assert s.context() == "[memory]"
    assert seen["cap"] == 123


async def test_episode_summary_gets_vector() -> None:
    from reachy_demo.memory.vectors import VectorIndex
    from tests.unit.fake_embedder import FakeEmbedder

    emb = FakeEmbedder()
    idx = VectorIndex(emb.model)
    s, store, clock, _ = _session(embedder=emb, index=idx)
    s.context()
    s.submit(_turn("We are planning a trip to Da Lat next month"))
    clock.t += 3 * 3600
    s.tick()
    await _run_jobs(s)
    [(eid, _)] = store.load_vecs("episode", emb.model)
    assert idx.top(emb.embed(["trip to Da Lat"], "query")[0], "episode", 1, {eid})


async def test_backfill_embeds_missing_then_returns_zero() -> None:
    from reachy_demo.memory.vectors import VectorIndex
    from tests.unit.fake_embedder import FakeEmbedder

    emb = FakeEmbedder()
    s, store, _, _ = _session(embedder=emb, index=VectorIndex(emb.model))
    for i in range(20):
        store.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text=f"fact {i}")
    assert await s.backfill(15) == 15
    assert await s.backfill(15) == 5
    assert await s.backfill(15) == 0
    assert len(store.load_vecs("fact", emb.model)) == 20


def _push_session() -> tuple[MemorySession, MemoryStore, Any]:
    from reachy_demo.memory.vectors import VectorIndex
    from tests.unit.fake_embedder import FakeEmbedder

    emb = FakeEmbedder({"sister": "mai", "she": "mai", "study": "mai"})
    idx = VectorIndex(emb.model)
    s, store, _, _ = _session(embedder=emb, index=idx)
    for i in range(12):  # enough facts that the briefing doesn't show them all
        store.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text=f"note {i}",
                       importance=0.9)  # fmt: skip
    mai = store.add_fact(subject_person_id=OWNER_ID, subject_text="Mai",
                         text="Mai studies medicine", importance=0.0)  # fmt: skip
    idx.put("fact", mai, emb.embed(["Mai studies medicine"], "doc")[0])
    return s, store, emb


async def test_session_related_small_talk_reuses_last_block() -> None:
    s, _, emb = _push_session()
    first = await s.related("my sister Mai came to visit me")
    calls = emb.calls
    assert "Mai studies medicine" in first
    assert await s.related("ok") == first
    assert emb.calls == calls  # small talk: no embedding


async def test_session_related_mean_carries_previous_turn() -> None:
    s, _, _ = _push_session()
    await s.related("my sister Mai came to visit me")
    assert "Mai studies medicine" in await s.related("what does she study these days")


async def test_session_related_embedder_failure_returns_last_block() -> None:
    s, _, emb = _push_session()
    first = await s.related("my sister Mai came to visit me")
    emb.fail = True
    assert await s.related("tell me about my sister again") == first


async def test_session_related_off_without_embedder() -> None:
    s, *_ = _session()
    assert await s.related("my sister Mai came to visit me") == ""


async def test_idle_backfill_waits_for_quiet_and_retries_after_failure() -> None:
    from reachy_demo.memory.session import BACKFILL_QUIET_S
    from reachy_demo.memory.vectors import VectorIndex
    from tests.unit.fake_embedder import FakeEmbedder

    emb = FakeEmbedder(fail=True)
    s, store, clock, _ = _session(embedder=emb, index=VectorIndex(emb.model))
    store.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="fact a")
    s.context()  # a turn just happened
    await s.idle()
    assert emb.calls == 0  # not while the user is talking
    clock.t += BACKFILL_QUIET_S + 1
    await s.idle()
    assert emb.calls == 1 and store.load_vecs("fact", emb.model) == []  # failed
    emb.fail = False
    await s.idle()
    assert len(store.load_vecs("fact", emb.model)) == 1  # retried on the next idle tick


async def test_related_cache_is_dropped_when_the_speaker_changes() -> None:
    from reachy_demo.memory.store import Speaker

    s, store, _ = _push_session()
    current = [OWNER]
    s._speaker = lambda: current[0]
    assert "Mai studies medicine" in await s.related("my sister Mai came to visit me")
    store._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    current[0] = Speaker(2, "guest")
    assert await s.related("ok") == ""  # never the owner's cached block
