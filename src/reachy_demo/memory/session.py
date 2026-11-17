"""The memory side of a conversation: session boundaries, the cached briefing, the write queue.

`context()` is the agent's per-round system-prompt hook (cheap: the briefing is built once per
session). `submit()` is its per-turn hook: it logs the turn at once and queues the slow LLM work
for `run()`, a background worker. Nothing here can raise into a turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from datetime import tzinfo
from typing import Any

from reachy_demo.memory.briefing import build_briefing
from reachy_demo.memory.consolidate import Consolidator, worth_keeping
from reachy_demo.memory.episodes import summarize_session
from reachy_demo.memory.prompts import ChatFn
from reachy_demo.memory.store import OWNER, Clock, MemoryStore, Speaker

log = logging.getLogger(__name__)

GAP_S = 600.0  # 10 min without a turn ends a session
PRELOAD_GAP_S = 7200.0  # reload the last raw turns only if the last session ended < 2 h ago
PRELOAD_TURNS = 5
RECOVER_S = 86400.0  # re-queue unconsolidated turns from the last day at startup

Job = tuple[
    Any, ...
]  # ("turn", event_id, user, answer, speaker) | ("summarize", session_id, person_id)


class MemorySession:
    def __init__(
        self,
        store: MemoryStore,
        consolidator: Consolidator,
        chat: ChatFn,
        *,
        clock: Clock,
        speaker: Callable[[], Speaker] = lambda: OWNER,
        gap_s: float = GAP_S,
        maxsize: int = 64,
        poll_s: float = 30.0,
        tz: tzinfo | None = None,
    ) -> None:
        self.store = store
        self.consolidator = consolidator
        self._chat = chat
        self._clock = clock
        self._speaker = speaker
        self._gap_s = gap_s
        self._poll_s = poll_s
        self._tz = tz
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize)
        self.session_id: str | None = None
        self._briefing = ""
        self._last = 0.0
        self.dropped = 0
        self._recover()  # before any turn, so nothing queued is queued twice

    # -- hot path (called by the agent) -------------------------------------------------------

    def context(self) -> str:
        """The [memory] block. A new session (and a fresh briefing) after a gap of `gap_s`."""
        now = self._clock()
        if self.session_id is None or now - self._last > self._gap_s:
            self._end_session()
            self.session_id = uuid.uuid4().hex[:12]
            try:
                self._briefing = build_briefing(self.store, self._speaker(), now, self._tz)
            except Exception:  # a broken DB must not break the conversation
                log.exception("briefing failed")
                self._briefing = ""
        self._last = now
        return self._briefing

    def submit(self, turn: list[dict[str, Any]]) -> None:
        """Log a finished turn (user, assistant and tool messages) and queue its consolidation."""
        try:
            if self.session_id is None:
                self.context()
            self._last = self._clock()
            user = str(turn[0].get("content") or "")
            answer = next(
                (
                    str(m["content"])
                    for m in reversed(turn)
                    if m["role"] == "assistant" and m.get("content")
                ),
                "",
            )
            tools = [c["function"]["name"] for m in turn for c in m.get("tool_calls") or []]
            speaker = self._speaker()
            eid = self.store.log_event(
                "turn",
                {"user": user, "answer": answer, "tools": tools},
                person_id=speaker.person_id,
                session_id=self.session_id,
            )
            if worth_keeping(user) and speaker.role == "owner":
                self._put(("turn", eid, user, answer, speaker))
            else:
                self.store.mark_consolidated(eid)
        except Exception:
            log.exception("memory submit failed")

    def preload(self) -> list[dict[str, Any]]:
        """Last raw turns as chat messages, if the previous session ended less than 2 h ago."""
        last = self.store.last_turn_time()
        if last is None or self._clock() - last > PRELOAD_GAP_S:
            return []
        messages: list[dict[str, Any]] = []
        for _, p in self.store.last_turns(PRELOAD_TURNS):
            messages += [
                {"role": "user", "content": str(p.get("user", ""))},
                {"role": "assistant", "content": str(p.get("answer", ""))},
            ]
        return messages

    # -- warm path ---------------------------------------------------------------------------

    def tick(self) -> None:
        """End the session if it has been idle for `gap_s` (the worker calls this)."""
        if self.session_id is not None and self._clock() - self._last > self._gap_s:
            self._end_session()

    def _end_session(self) -> None:
        if self.session_id is not None:
            self._put(("summarize", self.session_id, self._speaker().person_id))
            self.session_id = None

    def _put(self, job: Job) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self.dropped += 1
            log.warning(
                "memory queue full: dropped the oldest job", extra={"dropped": self.dropped}
            )
        self._queue.put_nowait(job)

    def _recover(self) -> None:
        """Queue what a crash left behind: unsummarized sessions, unconsolidated turns."""
        for sid, pid in self.store.sessions_without_episode(exclude=self.session_id):
            self._put(("summarize", sid, pid if pid is not None else OWNER.person_id))
        since = self._clock() - RECOVER_S
        for eid, _pid, p in self.store.unconsolidated_turns(since, self._queue.maxsize):
            # shortcut: replays as the owner (only speaker in Phase 1); use the logged person at M8.
            self._put(("turn", eid, str(p.get("user", "")), str(p.get("answer", "")), OWNER))

    async def run(self) -> None:
        """Background worker: process jobs one by one until cancelled."""
        while True:
            try:
                job = await asyncio.wait_for(self._queue.get(), self._poll_s)
            except TimeoutError:
                self.tick()
                continue
            try:
                await self._process(job)
            except Exception:
                log.exception("memory job failed", extra={"job": job[0]})
            finally:
                self._queue.task_done()

    async def _process(self, job: Job) -> None:
        if job[0] == "turn":
            _, eid, user, answer, speaker = job
            await self.consolidator.consolidate(user, speaker, answer=answer, event_id=eid)
        elif job[0] == "summarize":
            _, sid, pid = job
            await summarize_session(self.store, self._chat, sid, pid, self._clock)

    async def drain(self) -> None:
        """Wait until every queued job is done (the worker must be running)."""
        await self._queue.join()

    async def close(self, worker: asyncio.Task[None] | None, timeout_s: float = 5.0) -> None:
        """Summarize the open session, give the worker `timeout_s` to finish, then back up."""
        self._end_session()
        if worker is not None:
            try:
                await asyncio.wait_for(self.drain(), timeout_s)
            except TimeoutError:
                log.warning("memory jobs left for the next start (timeout at exit)")
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        if self.store.path:
            self.store.backup(self.store.path.with_name(self.store.path.name + ".bak"))
