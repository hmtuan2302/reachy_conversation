"""Conversation episodes: one LLM summary per session, plus follow-ups (spec §7.7)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from reachy_demo.memory.consolidate import parse_date, worth_keeping
from reachy_demo.memory.prompts import SUMMARIZE, ChatFn, ask_json
from reachy_demo.memory.store import Clock, MemoryStore
from reachy_demo.memory.vectors import Embedder, VectorIndex, store_vec

log = logging.getLogger(__name__)

MAX_TURNS = 40
MAX_CHARS = 24000  # ~6k tokens


async def summarize_session(
    store: MemoryStore,
    chat: ChatFn,
    session_id: str,
    person_id: int,
    clock: Clock,
    *,
    timeout_s: float = 10.0,
    embedder: Embedder | None = None,
    index: VectorIndex | None = None,
) -> int | None:
    """Write the session's episode once (idempotent). Returns its id, or None if nothing written."""
    turns = store.session_turns(session_id)[-MAX_TURNS:]
    if not turns:
        return None
    start, end = turns[0][0], turns[-1][0]
    if not any(worth_keeping(str(p.get("user", ""))) for _, p in turns):
        # Small talk only: stored without an LLM call, importance 0 keeps it out of the briefing.
        return _insert(store, session_id, start, end, person_id, "Only small talk.", [], 0.0)

    transcript = "\n".join(
        f"user: {p.get('user', '')}\nrobot: {p.get('answer', '')}" for _, p in turns
    )[-MAX_CHARS:]
    person = store.person(person_id)
    system = SUMMARIZE.format(
        speaker=(person.name if person and person.name else "its owner"),
        today=datetime.fromtimestamp(clock()).strftime("%A %Y-%m-%d"),
    )
    data = await ask_json(chat, system, transcript, timeout_s=timeout_s)
    summary = data.get("summary") if data else None
    if not isinstance(summary, str) or not summary.strip():
        log.warning("episode summary failed", extra={"session_id": session_id})
        return None  # retried at next startup (the session still has no episode)
    assert data is not None
    topics = [t for t in data.get("topics") or [] if isinstance(t, str)][:5]
    try:
        importance = float(data.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5
    episode_id = _insert(
        store, session_id, start, end, person_id, summary.strip(), topics, importance
    )
    if episode_id is not None:
        await store_vec(store, index, embedder, "episode", episode_id, summary.strip())
        for fu in data.get("follow_ups") or []:
            if isinstance(fu, dict) and isinstance(fu.get("text"), str) and fu["text"].strip():
                store.add_follow_up(
                    person_id, fu["text"].strip(), parse_date(fu.get("due")), episode_id
                )
    return episode_id


def _insert(
    store: MemoryStore,
    session_id: str,
    start: float,
    end: float,
    person_id: int,
    summary: str,
    topics: list[str],
    importance: float,
) -> int | None:
    try:
        return store.add_episode(
            session_id=session_id,
            start=start,
            end=end,
            person_ids=[person_id],
            summary=summary,
            topics=topics,
            importance=importance,
        )
    except sqlite3.IntegrityError:  # already summarized (unique session_id)
        return None
