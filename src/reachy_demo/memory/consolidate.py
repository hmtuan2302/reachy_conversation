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

from reachy_demo.memory.prompts import DECIDE, DESCRIBE, EXTRACT, ChatFn, ask_json
from reachy_demo.memory.store import Clock, MemoryStore, Speaker, content_words, own_name_fact

log = logging.getLogger(__name__)

CATEGORIES = frozenset({"bio", "preference", "relation", "plan", "setting", "other"})
# Facts about who the owner is; a change to one rewrites people.description.
SELF = frozenset({"bio", "preference"})
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
        """Store what's worth keeping from one user turn; `explicit` = asked to remember it.

        Owner: facts, name, follow-ups, and the description when a bio/preference fact changed.
        Guest: only facts, stored under the guest (never the owner's).
        """
        owner = speaker.role == "owner"
        person = self._store.person(speaker.person_id)
        name = person.name if person else None
        today = datetime.fromtimestamp(self._clock()).strftime("%A %Y-%m-%d")
        system = EXTRACT.format(
            today=today,
            speaker=name or "the owner (name not known yet)",
            who=name or '"The owner"',
        )
        user = f"Speaker said: {text}\nRobot answered: {answer or '(nothing)'}"
        open_before = self._store.all_open_follow_ups(speaker.person_id) if owner else []
        if open_before:
            user += "\nOpen follow-ups:\n" + "\n".join(f"{f.id}: {f.text}" for f in open_before)
        data = await ask_json(self._chat, system, user, timeout_s=self._timeout_s)
        if data is None:
            log.warning("memory extract failed", extra={"event_id": event_id})
            return []

        old_name = name
        if owner and isinstance(new_name := data.get("name"), str) and new_name.strip():
            name = new_name.strip()
            self._store.set_name(speaker.person_id, name)
            # The name lives in people only: close any "the owner's name is X" fact.
            for f in self._store.current_facts(speaker, speaker.person_id):
                if own_name_fact(f.text, old_name, name):
                    self._store.invalidate_fact(f.id, valid_to=None, replaced_by=None)

        changes: list[Change] = []
        about_self = False
        for raw in data.get("facts") or []:
            cand = _candidate(raw, name)
            if cand is None or own_name_fact(cand["text"], old_name, name):
                continue  # the speaker's name lives in people.name only
            if cand["sensitive"] and not explicit:
                log.info("memory skipped a sensitive fact (not asked to remember)")
                continue
            change = await self._apply(cand, speaker, event_id)
            changes.append(change)
            about_self |= change.action != "NOOP" and cand["category"] in SELF

        if not owner:
            if event_id is not None:
                self._store.mark_consolidated(event_id)
            return changes
        if about_self:
            await self.describe(speaker, name)

        for raw in data.get("follow_ups") or []:
            if isinstance(raw, dict) and isinstance(raw.get("text"), str) and raw["text"].strip():
                self._store.add_follow_up(
                    speaker.person_id, raw["text"].strip(), parse_date(raw.get("due"))
                )

        # The LLM judges meaning; ids it wasn't given are ignored. Word overlap stays as backup.
        raw = data.get("answered")
        # The model sometimes writes ids as strings ("1"); bools and other junk are skipped.
        answered = (
            {int(i) for i in raw if isinstance(i, int | str) and str(i).strip().isdigit()}
            if isinstance(raw, list)
            else set()
        )
        for fu in open_before:
            if fu.id in answered:
                self._store.close_follow_up(fu.id)
        still_open = [f for f in open_before if f.id not in answered]
        self._close_follow_ups(still_open, [c.text for c in changes if c.action != "NOOP"], name)
        if event_id is not None:
            self._store.mark_consolidated(event_id)
        return changes

    async def describe(self, speaker: Speaker, name: str | None) -> None:
        """Rewrite people.description from the owner's current bio/preference facts."""
        facts = [
            f.text
            for f in self._store.current_facts(speaker, speaker.person_id)
            if f.category in SELF
        ]
        if not facts:
            return
        who = name or "the owner"
        data = await ask_json(
            self._chat,
            DESCRIBE.format(who=who),
            "\n".join(f"- {t}" for t in facts),
            timeout_s=self._timeout_s,
        )
        text = (data or {}).get("description")
        if isinstance(text, str) and text.strip():
            self._store.set_description(speaker.person_id, text)
        else:
            log.warning("memory describe failed; description unchanged")

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

    def close_mentioned(self, speaker: Speaker, *texts: str) -> None:
        """Close the owner's follow-ups that this turn asked or answered (user text or reply).

        Runs on every turn, before the small-talk gate: "yes" alone never reaches extraction,
        but the robot's reply to it ("I'm back to my own voice") shows the follow-up is done.
        """
        if speaker.role != "owner":
            return
        person = self._store.person(speaker.person_id)
        open_now = self._store.all_open_follow_ups(speaker.person_id)
        self._close_follow_ups(open_now, list(texts), person.name if person else None)

    def _close_follow_ups(self, open_before: list[Any], texts: list[str], name: str | None) -> None:
        """A text sharing enough words with an open follow-up closes it (word overlap, §7.7)."""
        # "Ask whether Tuan wants the robot ...": these words say nothing about the topic.
        ignore = content_words(name or "") | {"owner", "ask", "asks", "robot", "whether"}
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
