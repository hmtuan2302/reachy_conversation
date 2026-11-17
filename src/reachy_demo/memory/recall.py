"""`recall`: search facts and past conversations, scoped to the speaker, abstaining when empty.

Spec: docs/superpowers/specs/2026-11-17-memory-design.md §7.5.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from reachy_demo.memory.briefing import ago
from reachy_demo.memory.store import Episode, Fact, MemoryStore, Speaker, content_words
from reachy_demo.memory.timeparse import parse

if TYPE_CHECKING:
    import numpy as np

    from reachy_demo.memory.vectors import VectorIndex

RRF_K = 60
VEC_MIN_SIM = 0.3  # below this a vector hit is noise (tune on the eval)
VEC_TOP_N = 20


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
    index: VectorIndex | None = None,
    query_vec: np.ndarray | None = None,
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

    scored: list[tuple[float, Fact | Episode]] = []
    if kind in ("fact", "any"):
        scored += hybrid_facts(
            store,
            speaker,
            query,
            query_vec,
            index,
            person_id=person_id,
            subject_text=subject_text,
            start=start,
            end=end,
            limit=limit * 2,
        )
    if kind in ("episode", "any"):
        scored += hybrid_episodes(
            store, speaker, query, query_vec, index, start=start, end=end, limit=limit * 2
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [item for _, item in scored[:limit]]

    now_ts = now.timestamp()
    result: dict[str, Any] = {"found": bool(chosen)}
    if chosen:
        result["results"] = [render_line(item, now_ts, now) for item in chosen]
        store.touch_facts([i.id for i in chosen if isinstance(i, Fact)])
    if note:
        result["note"] = note
    return result


def hybrid_facts(
    store: MemoryStore,
    speaker: Speaker,
    query: str,
    query_vec: np.ndarray | None,
    index: VectorIndex | None,
    *,
    person_id: int | None = None,
    subject_text: str | None = None,
    start: float | None = None,
    end: float | None = None,
    limit: int,
) -> list[tuple[float, Fact]]:
    """Keyword + vector + recency, fused by RRF; best first. Vectors only among allowed ids."""
    words = [f for f, _ in store.search_facts(
        query, speaker, person_id=person_id, subject_text=subject_text,
        start=start, end=end, limit=limit,
    )]  # fmt: skip
    near: list[Fact] = []
    if index is not None and query_vec is not None:
        allowed = store.allowed_ids(
            "fact", speaker, person_id=person_id, subject_text=subject_text, start=start, end=end
        )
        hits = index.top(query_vec, "fact", VEC_TOP_N, allowed, min_sim=VEC_MIN_SIM)
        near = store.facts_by_ids([i for i, _ in hits])
    fused = _fuse(words, near)
    kept = {f.id for f, _ in _dedupe([(f, 0.0) for _, f in fused])}
    return [(score, f) for score, f in fused if f.id in kept][:limit]


def hybrid_episodes(
    store: MemoryStore,
    speaker: Speaker,
    query: str,
    query_vec: np.ndarray | None,
    index: VectorIndex | None,
    *,
    start: float | None = None,
    end: float | None = None,
    limit: int,
) -> list[tuple[float, Episode]]:
    words = [e for e, _ in store.search_episodes(query, speaker, start=start, end=end, limit=limit)]
    near: list[Episode] = []
    if index is not None and query_vec is not None:
        allowed = store.allowed_ids("episode", speaker, start=start, end=end)
        hits = index.top(query_vec, "episode", VEC_TOP_N, allowed, min_sim=VEC_MIN_SIM)
        near = store.episodes_by_ids([i for i, _ in hits])
    return _fuse(words, near)[:limit]


def _fuse[T: (Fact, Episode)](words: list[T], near: list[T]) -> list[tuple[float, T]]:
    """Reciprocal rank fusion of keyword order, vector order and recency over their union."""
    items: dict[int, T] = {}
    score: dict[int, float] = {}
    for ranked in (words, near):
        for r, item in enumerate(ranked):
            items.setdefault(item.id, item)
            score[item.id] = score.get(item.id, 0.0) + 1 / (RRF_K + r)
    newest = sorted(items.values(), key=_time, reverse=True)
    for r, item in enumerate(newest):
        score[item.id] += 1 / (RRF_K + r)
    return sorted(((score[i], items[i]) for i in items), key=lambda x: x[0], reverse=True)


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


def render_line(item: Fact | Episode, now_ts: float, now: datetime) -> str:
    if isinstance(item, Fact):
        line = f"fact: {item.text} [{item.source}, {ago(now_ts - item.created)}]"
        if item.expired:
            since = datetime.fromtimestamp(item.expired, now.tzinfo).strftime("%d %b %Y")
            line += f" (no longer true since {since})"
        return line
    label = datetime.fromtimestamp(item.start, now.tzinfo).strftime("%a %d %b %Y %H:%M")
    return f"conversation {label}: {item.summary}"
