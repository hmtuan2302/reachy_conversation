"""Per-turn push: a few memories related to what the user just said (smart-context §7.4)."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from reachy_demo.memory.recall import render_line
from reachy_demo.memory.store import Episode, Fact, MemoryStore, Speaker
from reachy_demo.memory.vectors import VectorIndex

PUSH_MIN_SIM = 0.55  # quiet over noisy; tune on the eval (proactive vs noise)
PUSH_TOP = 3
PUSH_LINE_CHARS = 240
HEADER = "[related memories — use only if it fits the conversation]"


def related_block(
    store: MemoryStore,
    index: VectorIndex,
    speaker: Speaker,
    query_vec: np.ndarray,
    *,
    exclude_facts: frozenset[int],
    exclude_episodes: frozenset[int],
    now: datetime,
) -> str:
    """Up to PUSH_TOP current, non-sensitive memories of this speaker; "" when none is close.

    Pushing is not using: no touch_facts, so the push never feeds the briefing ranking.
    """
    pid = speaker.person_id
    facts = store.allowed_ids("fact", speaker, person_id=pid, include_sensitive=False)
    episodes = store.allowed_ids("episode", speaker, person_id=pid)
    hits = [
        ("fact", i, sim)
        for i, sim in index.top(query_vec, "fact", PUSH_TOP, facts - exclude_facts, PUSH_MIN_SIM)
    ] + [
        ("episode", i, sim)
        for i, sim in index.top(
            query_vec, "episode", PUSH_TOP, episodes - exclude_episodes, PUSH_MIN_SIM
        )
    ]
    hits.sort(key=lambda h: h[2], reverse=True)
    hits = hits[:PUSH_TOP]
    if not hits:
        return ""
    by_fact = {f.id: f for f in store.facts_by_ids([i for k, i, _ in hits if k == "fact"])}
    by_ep = {e.id: e for e in store.episodes_by_ids([i for k, i, _ in hits if k == "episode"])}
    items: list[Fact | Episode] = []
    for kind, i, _ in hits:
        item = by_fact.get(i) if kind == "fact" else by_ep.get(i)
        if item is not None:
            items.append(item)
    lines = [_cut("- " + render_line(item, now.timestamp(), now)) for item in items]
    return "\n".join([HEADER, *lines]) if lines else ""


def _cut(s: str) -> str:
    return s if len(s) <= PUSH_LINE_CHARS else s[: PUSH_LINE_CHARS - 1].rstrip() + "…"
