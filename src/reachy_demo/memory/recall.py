"""`recall`: search facts and past conversations, scoped to the speaker, abstaining when empty.

Spec: docs/superpowers/specs/2026-11-17-memory-design.md §7.5.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from reachy_demo.memory.briefing import ago
from reachy_demo.memory.store import Episode, Fact, MemoryStore, Speaker, content_words
from reachy_demo.memory.timeparse import parse

RRF_K = 60


def recall(
    store: MemoryStore,
    speaker: Speaker,
    query: str,
    *,
    now: datetime,
    person: str | None = None,
    kind: str = "any",
    when: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    limit = min(max(limit, 1), 10)
    note = None
    start = end = None
    if when:
        window = parse(when, now)
        if window:
            start, end = window
        else:
            note = f"could not understand the time {when!r}; searched all time"

    person_id = None
    subject_text = None
    if person and person.strip().lower() in ("me", "i", "tôi", "mình"):
        person_id = speaker.person_id
    elif person:
        found = store.find_person(person)
        if found:
            person_id = found.id
        else:
            subject_text = person.strip()

    facts: list[tuple[Fact, float]] = []
    episodes: list[tuple[Episode, float]] = []
    if kind in ("fact", "any"):
        facts = store.search_facts(
            query,
            speaker,
            person_id=person_id,
            subject_text=subject_text,
            start=start,
            end=end,
            limit=limit * 2,
        )
        facts = _dedupe(facts)
    if kind in ("episode", "any"):
        episodes = store.search_episodes(query, speaker, start=start, end=end, limit=limit * 2)

    # Reciprocal rank fusion of text relevance (BM25 order) and recency.
    items: list[tuple[float, Fact | Episode]] = []
    groups: list[list[tuple[Fact | Episode, float]]] = [list(facts), list(episodes)]
    for group in groups:
        newest = sorted(range(len(group)), key=lambda i: _time(group[i][0]), reverse=True)
        time_rank = {i: r for r, i in enumerate(newest)}
        for i, (item, _bm25) in enumerate(group):  # group is already in BM25 order
            items.append((1 / (RRF_K + i) + 1 / (RRF_K + time_rank[i]), item))
    items.sort(key=lambda x: x[0], reverse=True)
    chosen = [item for _, item in items[:limit]]

    now_ts = now.timestamp()
    result: dict[str, Any] = {"found": bool(chosen)}
    if chosen:
        result["results"] = [_line(item, now_ts, now) for item in chosen]
        store.touch_facts([i.id for i in chosen if isinstance(i, Fact)])
    if note:
        result["note"] = note
    return result


def _dedupe(facts: list[tuple[Fact, float]]) -> list[tuple[Fact, float]]:
    """Drop a fact whose words mostly repeat a better-ranked fact about the same subject."""
    kept: list[tuple[Fact, float]] = []
    for f, rank in facts:
        words = content_words(f.text)
        if not any(
            k.subject_text == f.subject_text
            and words
            and len(words & content_words(k.text)) / len(words) > 0.8
            for k, _ in kept
        ):
            kept.append((f, rank))
    return kept


def _time(item: Fact | Episode) -> float:
    return item.created if isinstance(item, Fact) else item.start


def _line(item: Fact | Episode, now_ts: float, now: datetime) -> str:
    if isinstance(item, Fact):
        line = f"fact: {item.text} [{item.source}, {ago(now_ts - item.created)}]"
        if item.expired:
            since = datetime.fromtimestamp(item.expired, now.tzinfo).strftime("%d %b %Y")
            line += f" (no longer true since {since})"
        return line
    label = datetime.fromtimestamp(item.start, now.tzinfo).strftime("%a %d %b %Y %H:%M")
    return f"conversation {label}: {item.summary}"
