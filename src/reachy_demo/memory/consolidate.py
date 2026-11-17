"""Write path: extract facts from a turn, compare with similar ones, then decide.

Decisions: ADD / UPDATE / INVALIDATE / NOOP.

Spec: docs/superpowers/specs/2026-11-17-memory-design.md §7.4 (Mem0 / RoboMemory pattern).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from reachy_demo.memory.prompts import DECIDE, EXTRACT, ChatFn, ask_json
from reachy_demo.memory.store import Clock, MemoryStore, Speaker, content_words, own_name_fact

log = logging.getLogger(__name__)

CATEGORIES = frozenset({"bio", "preference", "relation", "plan", "setting", "other"})
# A turn made only of these words is small talk or a robot command: nothing to remember.
_SMALL_TALK_TEXT = """
    hi hello hey bye goodbye thanks thank ok okay yes no yeah yep nope sure cool nice great good
    morning night please sorry hmm uh um ah oh wow lol haha
    chào xin cảm ơn cám vâng dạ ừ ừm ok được rồi tốt hay quá nhé nha à ờ
    turn look left right up down center dance stop nod move head camera see what you do can
    your me again more faster slower quay nhìn trái phải lên xuống nhảy dừng lại đi thôi
"""
SMALL_TALK = frozenset(_SMALL_TALK_TEXT.split())
KEEP = re.compile(r"\b(remember|nhớ|my name|tên (tôi|mình|em|anh|chị) là)\b", re.IGNORECASE)


def worth_keeping(text: str) -> bool:
    """Cheap gate before any LLM call: skip greetings, thanks and robot commands."""
    if KEEP.search(text):
        return True
    words = re.findall(r"\w+", text.lower())
    if len(words) < 3:
        return False
    return not all(w in SMALL_TALK for w in words)


@dataclass(frozen=True)
class Change:
    action: str  # ADD | UPDATE | INVALIDATE | NOOP
    fact_id: int | None
    text: str


class Consolidator:
    def __init__(
        self, store: MemoryStore, chat: ChatFn, clock: Clock, *, timeout_s: float = 10.0
    ) -> None:
        self._store = store
        self._chat = chat
        self._clock = clock
        self._timeout_s = timeout_s

    async def consolidate(
        self,
        text: str,
        speaker: Speaker,
        *,
        answer: str = "",
        event_id: int | None = None,
        explicit: bool = False,
    ) -> list[Change]:
        """Store what's worth keeping from one user turn; `explicit` = asked to remember it."""
        if speaker.role != "owner":
            return []  # guests never write memory (spec §5.4)
        person = self._store.person(speaker.person_id)
        name = person.name if person else None
        today = datetime.fromtimestamp(self._clock()).strftime("%A %Y-%m-%d")
        system = EXTRACT.format(
            today=today,
            speaker=name or "the owner (name not known yet)",
            who=name or '"The owner"',
        )
        user = f"Speaker said: {text}\nRobot answered: {answer or '(nothing)'}"
        data = await ask_json(self._chat, system, user, timeout_s=self._timeout_s)
        if data is None:
            log.warning("memory extract failed", extra={"event_id": event_id})
            return []

        old_name = name
        if isinstance(new_name := data.get("name"), str) and new_name.strip():
            name = new_name.strip()
            self._store.set_name(speaker.person_id, name)
            # The name lives in people only: close any "the owner's name is X" fact.
            for f in self._store.current_facts(speaker, speaker.person_id):
                if own_name_fact(f.text, old_name, name):
                    self._store.invalidate_fact(f.id, valid_to=None, replaced_by=None)

        open_before = self._store.all_open_follow_ups(speaker.person_id)
        changes: list[Change] = []
        for raw in data.get("facts") or []:
            cand = _candidate(raw, name)
            if cand is None or own_name_fact(cand["text"], old_name, name):
                continue  # the speaker's name lives in people.name only
            if cand["sensitive"] and not explicit:
                log.info("memory skipped a sensitive fact (not asked to remember)")
                continue
            changes.append(await self._apply(cand, speaker, event_id))

        for raw in data.get("follow_ups") or []:
            if isinstance(raw, dict) and isinstance(raw.get("text"), str) and raw["text"].strip():
                self._store.add_follow_up(
                    speaker.person_id, raw["text"].strip(), parse_date(raw.get("due"))
                )

        self._close_follow_ups(open_before, [c.text for c in changes if c.action != "NOOP"], name)
        if event_id is not None:
            self._store.mark_consolidated(event_id)
        return changes

    async def _apply(self, cand: dict[str, Any], speaker: Speaker, event_id: int | None) -> Change:
        similar = self._store.search_facts(
            cand["text"], speaker, person_id=speaker.person_id, limit=5
        )
        decision: dict[str, Any] = {"action": "ADD"}
        if similar:
            listing = "\n".join(f"{f.id}: {f.text}" for f, _ in similar)
            got = await ask_json(
                self._chat,
                DECIDE,
                f"New: {cand['text']}\nExisting:\n{listing}",
                timeout_s=self._timeout_s,
            )
            decision = got or decision  # decide failed: keep the fact (a later dedupe merges it)
        action = str(decision.get("action", "ADD")).upper()
        raw_id = decision.get("id")
        target = (
            raw_id if isinstance(raw_id, int) and raw_id in {f.id for f, _ in similar} else None
        )
        if action in ("UPDATE", "INVALIDATE") and target is None:
            log.warning("memory decide used an id it was not given: %r", raw_id)
            action = "ADD"

        evidence = [event_id] if event_id is not None else []
        keys = (
            "subject_text",
            "text",
            "text_original",
            "category",
            "importance",
            "sensitive",
            "valid_from",
            "valid_to",
        )
        fields = {k: cand[k] for k in keys}
        with self._store.transaction():
            if action == "NOOP":
                return Change("NOOP", None, cand["text"])
            if action == "UPDATE" and target is not None:
                merged = decision.get("text")
                text = (
                    merged.strip() if isinstance(merged, str) and merged.strip() else cand["text"]
                )
                self._store.update_fact(target, text, cand["text_original"] or None)
                return Change("UPDATE", target, text)
            fact_id = self._store.add_fact(
                subject_person_id=speaker.person_id, source="said", evidence=evidence, **fields
            )
            if action == "INVALIDATE" and target is not None:
                self._store.invalidate_fact(
                    target, valid_to=cand["valid_from"], replaced_by=fact_id
                )
                return Change("INVALIDATE", fact_id, cand["text"])
            return Change("ADD", fact_id, cand["text"])

    def _close_follow_ups(self, open_before: list[Any], texts: list[str], name: str | None) -> None:
        """A new fact that answers an open follow-up closes it (word overlap, spec §7.7)."""
        ignore = content_words(name or "") | {"owner"}
        for fu in open_before:
            need = content_words(fu.text) - ignore
            if not need:
                continue
            for text in texts:
                if len(need & content_words(text)) >= min(2, len(need)):
                    self._store.close_follow_up(fu.id)
                    break


def _candidate(raw: Any, name: str | None) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not raw["text"].strip():
        return None
    try:
        importance = float(raw.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5
    category = raw.get("category")
    subject = raw.get("subject")
    original = raw.get("text_original")
    return {
        "subject_text": subject.strip()
        if isinstance(subject, str) and subject.strip()
        else (name or "owner"),
        "text": raw["text"].strip(),
        "text_original": original.strip() if isinstance(original, str) else "",
        "category": category if category in CATEGORIES else "other",
        "importance": importance,
        "sensitive": raw.get("sensitive") is True,
        "valid_from": parse_date(raw.get("valid_from")),
        "valid_to": parse_date(raw.get("valid_to")),
    }


def parse_date(value: Any) -> float | None:
    """'YYYY-MM-DD' → local midnight epoch; anything else → None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").timestamp()
    except ValueError:
        return None
