"""The [memory] block put in the system prompt, built once per session. SQL only, no LLM.

Blocks and order: docs/superpowers/specs/2026-11-17-memory-design.md §7.3.
"""

from __future__ import annotations

import math
from datetime import datetime, tzinfo

from reachy_demo.memory.store import Fact, MemoryStore, Speaker

CAP_CHARS = 2400  # ~600 tokens at ~4 chars per token
MIN_FACTS = 5
EPISODE_CHARS = 240  # ~60 tokens per conversation summary
RULE = (
    "Use recall for anything else. Never invent memories: "
    "if recall finds nothing, say you don't remember."
)


def ago(seconds: float) -> str:
    def n(v: float, unit: str) -> str:
        k = int(v)
        return f"{k} {unit}{'' if k == 1 else 's'} ago"

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return n(seconds / 60, "minute")
    if seconds < 86400:
        return n(seconds / 3600, "hour")
    if seconds < 2 * 86400:
        return "yesterday"
    if seconds < 14 * 86400:
        return n(seconds / 86400, "day")
    if seconds < 60 * 86400:
        return n(seconds / (7 * 86400), "week")
    return n(seconds / (30 * 86400), "month")


def score(f: Fact, now: float) -> float:
    age_days = max(0.0, now - max(f.created, f.last_used or 0.0)) / 86400
    recency = math.exp(-age_days / 20)  # ~14-day half-life
    return 0.5 * recency + 0.3 * f.importance + 0.2 * math.log1p(f.use_count) / math.log1p(50)


def fact_line(f: Fact, now: float) -> str:
    return f"- {f.text} ({f.source}, {ago(now - f.created)})"


def build_briefing(
    store: MemoryStore, speaker: Speaker, now: float, tz: tzinfo | None = None
) -> str:
    def when(t: float, fmt: str) -> str:
        return datetime.fromtimestamp(t, tz).strftime(fmt)

    person = store.person(speaker.person_id)
    name = person.name if person else None
    head = f"[memory] Now: {when(now, '%a %d %b %Y %H:%M')}. "
    if name:
        earlier = (
            f"; earlier name: {', '.join(person.aliases)}" if person and person.aliases else ""
        )
        head += f"You are talking to {name} ({speaker.role}{earlier})"
    else:
        head += (
            f"You are talking to the {speaker.role}, whose name you don't know yet; "
            "if they tell you, remember it"
        )
    last = store.last_turn_time(speaker.person_id)
    head += f"; last talked {ago(now - last)}." if last else "; this is your first conversation."
    if person and person.description:
        head += f" About them: {person.description}"

    ranked = sorted(
        store.current_facts(speaker, speaker.person_id), key=lambda f: score(f, now), reverse=True
    )
    prefs = [f for f in ranked if f.category in ("preference", "setting")][:5]
    facts = [f for f in ranked if f not in prefs][:10]
    follow_ups = [
        f"- {fu.text}" + (f" (due {when(fu.due, '%a %d %b')})" if fu.due else "")
        for fu in store.open_follow_ups(speaker.person_id, now)
    ]
    episodes = [
        f"- {when(e.start, '%a %d %b %H:%M')}: {_cut(e.summary, EPISODE_CHARS)}"
        for e in store.recent_episodes(speaker.person_id, 5)
    ]

    def render() -> str:
        parts = [head]
        if prefs:
            parts += ["Preferences:", *(fact_line(f, now) for f in prefs)]
        if follow_ups:
            parts += ["Follow up when it fits:", *follow_ups]
        if episodes:
            parts += ["Recent conversations:", *episodes]
        if facts:
            parts += [f"Known about {name or 'them'}:", *(fact_line(f, now) for f in facts)]
        parts.append(RULE)
        return "\n".join(parts)

    text = render()
    while len(text) > CAP_CHARS and len(facts) > MIN_FACTS:
        facts.pop()
        text = render()
    while len(text) > CAP_CHARS and follow_ups:
        follow_ups.pop()
        text = render()
    return text


def _cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"
