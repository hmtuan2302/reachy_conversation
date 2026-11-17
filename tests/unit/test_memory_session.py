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
