"""Memory tools: recall (anyone, scoped), remember and forget (owner only, enforced here).

Spec: docs/superpowers/specs/2026-11-17-memory-design.md §7.5–§7.6. Ideas from
reachy_mini_conversation_app `tools/remember.py` and `forget.py` (Apache-2.0, Pollen Robotics).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from reachy_demo.memory.consolidate import Consolidator
from reachy_demo.memory.recall import recall
from reachy_demo.memory.store import MemoryStore, Speaker
from reachy_demo.memory.vectors import Embedder, VectorIndex, embed_one

log = logging.getLogger(__name__)

REMEMBER_WAIT_S = 3.0
PROMPT = (
    "You have a long-term memory. The [memory] block below is what you already know. "
    "For anything personal not listed there, call recall before answering. If recall finds "
    "nothing, say you don't remember; never guess or invent memories. The [memory] block is "
    "current: when the user gives a new name or corrects something, the newest statement wins, "
    "so accept it without asking them to confirm. You remember what the "
    "user tells you automatically, so call remember only when they ask you to remember "
    "something. Call forget only when they ask you to forget."
)
NOT_OWNER = {"error": "not_owner: only the owner can change your memory"}


def _fn(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SPECS = [
    _fn(
        "recall",
        "Search your long-term memory about people, what they told you and past conversations. "
        "Write query as English keywords. Use it before saying you don't know something personal. "
        "Write a standalone query (no pronouns). For several topics, call recall once per topic.",
        {
            "query": {"type": "string", "description": "English keywords, e.g. 'sister name'"},
            "person": {"type": "string", "description": "'me' for the speaker, or a name"},
            "kind": {"type": "string", "enum": ["fact", "episode", "any"]},
            "when": {
                "type": "string",
                "description": "time words as the user said them, e.g. 'yesterday', 'hôm qua', "
                "'last Tuesday'",
            },
            "limit": {"type": "integer", "description": "max results, default 5"},
        },
        ["query"],
    ),
    _fn(
        "remember",
        "Store something the user explicitly asked you to remember.",
        {"fact": {"type": "string", "description": "what to remember, as the user said it"}},
        ["fact"],
    ),
    _fn(
        "forget",
        "Forget something the user asked you to forget. With several matches, ask which one, then "
        "call again with ids. To forget everything, first ask 'are you sure?', then call with "
        "everything=true and confirmed=true.",
        {
            "query": {"type": "string", "description": "what to forget, English keywords"},
            "ids": {"type": "array", "items": {"type": "integer"}},
            "everything": {"type": "boolean"},
            "confirmed": {"type": "boolean"},
        },
        [],
    ),
]


class MemoryTools:
    def __init__(
        self,
        store: MemoryStore,
        consolidator: Consolidator,
        *,
        speaker: Callable[[], Speaker],
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        embedder: Embedder | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        self._store = store
        self._consolidator = consolidator
        self._speaker = speaker
        self._now = now
        self._embedder = embedder
        self._index = index
        self._pending: set[asyncio.Task[Any]] = set()  # slow remembers finishing in background

    def specs(self) -> list[dict[str, Any]]:
        return SPECS

    def needs_response(self, name: str, arguments: str) -> bool:
        return True

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            return {"error": "bad_arguments: expected a JSON object"}
        speaker = self._speaker()
        try:
            if name == "recall":
                return await self._recall(speaker, args)
            if speaker.role != "owner":
                return NOT_OWNER
            if name == "remember":
                return await self._remember(speaker, args)
            if name == "forget":
                return self._forget(speaker, args)
        except Exception:  # a memory bug must not end the turn
            log.exception("memory tool %s failed", name)
            return {"error": "internal"}
        return {"error": f"unknown_tool: {name}"}

    async def _recall(self, speaker: Speaker, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str):
            return {"error": "bad_arguments: query must be a string"}
        vec = await embed_one(self._embedder, query, "query") if self._index else None
        limit = args.get("limit", 5)
        return recall(
            self._store,
            speaker,
            query,
            now=self._now(),
            person=_str(args.get("person")),
            kind=_str(args.get("kind")) or "any",
            when=_str(args.get("when")),
            limit=limit if isinstance(limit, int) else 5,
            index=self._index,
            query_vec=vec,
        )

    async def _remember(self, speaker: Speaker, args: dict[str, Any]) -> dict[str, Any]:
        fact = _str(args.get("fact"))
        if not fact:
            return {"error": "bad_arguments: fact must be a non-empty string"}
        task = asyncio.create_task(self._consolidator.consolidate(fact, speaker, explicit=True))
        done, _ = await asyncio.wait({task}, timeout=REMEMBER_WAIT_S)
        if not done:
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
            return {"ok": True, "note": "saving in the background"}
        changes = task.result()
        if not changes:
            return {"ok": False, "note": "found nothing to remember in that"}
        stored = [c.text for c in changes if c.action != "NOOP"]
        return {"ok": True, "stored": stored} if stored else {"ok": True, "already_known": True}

    def _forget(self, speaker: Speaker, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("everything") is True:
            if args.get("confirmed") is not True:
                return {"confirm_needed": True, "note": "ask the user to confirm first"}
            self._store.forget_person(speaker.person_id)
            return {"ok": True, "forgot": "everything about the speaker"}
        ids = args.get("ids")
        if isinstance(ids, list) and ids:
            own = [i for i in ids if isinstance(i, int) and self._owns(speaker, i)]
            return {"ok": True, "forgot_count": self._store.delete_facts(own)}
        query = _str(args.get("query"))
        if not query:
            return {"error": "bad_arguments: give query, ids, or everything"}
        matches = [f for f, _ in self._store.search_facts(query, speaker, limit=5)]
        if not matches:
            return {"found": False}
        if len(matches) == 1:
            self._store.delete_facts([matches[0].id])
            return {"ok": True, "forgot": [matches[0].text]}
        return {
            "matches": [{"id": f.id, "text": f.text} for f in matches],
            "note": "ask which one to forget, then call forget with ids",
        }

    def _owns(self, speaker: Speaker, fact_id: int) -> bool:
        f = self._store.fact(fact_id)
        return f is not None and (
            speaker.role == "owner" or f.subject_person_id == speaker.person_id
        )


def _str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
