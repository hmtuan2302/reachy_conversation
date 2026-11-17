"""`reachy-demo memory eval|reset`.

`eval`: scripted memory test set against the real LLM, on a simulated clock.
`reset`: delete the memory database (MEMORY_DB) to start an experiment fresh.

Each JSONL line is one step: {"t": ISO time, "say": text} tells the agent something;
{"t", "ask", "expect": [...], "category"} asks and scores; {"expect_none": true} scores an
abstention; {"restart": true} restarts the app (same memory DB, fresh agent). `--baseline` runs
the same script with memory off and the whole history kept in context (spec §12).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from reachy_demo.agent.agent import SYSTEM_PROMPT, Agent, ToolChatFn
from reachy_demo.agent.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMError,
    chat_events,
    chat_stream,
)
from reachy_demo.agent.tools import Toolbox
from reachy_demo.agent.tools import memory as memory_tools
from reachy_demo.memory.consolidate import Consolidator
from reachy_demo.memory.embed import load_embedder
from reachy_demo.memory.prompts import ChatFn
from reachy_demo.memory.session import MemorySession
from reachy_demo.memory.store import OWNER, MemoryStore
from reachy_demo.memory.vectors import Embedder, EmbedKind, VectorIndex, backfill

DEFAULT_FILE = "tests/replay/memory_eval.jsonl"
# shortcut: abstention and answers are scored by substring, not an LLM judge; add a judge if
# correct answers get marked wrong for wording.
ABSTAIN = (
    "don't remember", "do not remember", "don't know", "do not know", "not sure",
    "no record", "haven't told", "didn't tell", "never told", "không nhớ", "không biết",
    "chưa nói", "chưa từng", "chưa kể",
)  # fmt: skip


def norm(text: str) -> str:
    """Lowercase without diacritics, so 'Tuấn' matches 'tuan'."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def score(answer: str, step: dict[str, Any]) -> bool:
    a = norm(answer)
    if step.get("expect_none"):
        return any(norm(m) in a for m in ABSTAIN)
    if any(re.search(rf"\b{re.escape(norm(e))}\b", a) for e in step.get("expect_absent", [])):
        return False  # brought up something unrelated (noise)
    expect = step.get("expect", [])
    return not expect or any(norm(e) in a for e in expect)


class SimClock:
    def __init__(self) -> None:
        self.t = time.time()

    def __call__(self) -> float:
        return self.t


@dataclass
class Report:
    results: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    failures: list[str] = field(default_factory=list)
    briefing_ms: list[float] = field(default_factory=list)
    embed_ms: list[float] = field(default_factory=list)
    push_ms: list[float] = field(default_factory=list)
    turns: int = 0
    memory_calls: int = 0

    def lines(self) -> list[str]:
        out = []
        total = [ok for oks in self.results.values() for ok in oks]
        for cat, oks in sorted(self.results.items()):
            out.append(f"{cat:<14} {sum(oks)}/{len(oks)}  {100 * sum(oks) / len(oks):.0f}%")
        if total:
            out.append(
                f"{'overall':<14} {sum(total)}/{len(total)}  {100 * sum(total) / len(total):.0f}%"
            )
        if self.briefing_ms:
            q = sorted(self.briefing_ms)
            p95 = q[min(len(q) - 1, int(0.95 * len(q)))]
            out.append(f"briefing ms    p50 {statistics.median(q):.1f}  p95 {p95:.1f}")
        for label, ms in (("query embed ms", self.embed_ms), ("push ms", self.push_ms)):
            if ms:
                q = sorted(ms)
                p95 = q[min(len(q) - 1, int(0.95 * len(q)))]
                out.append(f"{label} p50 {statistics.median(q):.1f}  p95 {p95:.1f}")
        if self.turns:
            out.append(f"memory LLM calls per turn  {self.memory_calls / self.turns:.2f}")
        return out + [f"FAIL {f}" for f in self.failures]


class _Timed:
    """Wraps the embedder to time query embeddings (the per-turn cost)."""

    def __init__(self, inner: Embedder, sink: list[float]) -> None:
        self.model = inner.model
        self._inner = inner
        self._sink = sink

    def embed(self, texts: list[str], kind: EmbedKind) -> Any:
        t0 = time.perf_counter()
        out = self._inner.embed(texts, kind)
        if kind == "query":
            self._sink.append(1000 * (time.perf_counter() - t0))
        return out


async def evaluate(
    steps: list[dict[str, Any]],
    agent_chat: Callable[[list[dict[str, Any]] | None], ToolChatFn],
    memory_chat: ChatFn | None,
    db_path: Path | None,
    *,
    embedder: Embedder | None = None,
) -> Report:
    """Run the script. `agent_chat(tool_specs)` builds the agent's chat function.

    With `memory_chat` and `db_path`, memory is on; without, it is the long-context baseline.
    """
    report = Report()
    clock = SimClock()

    async def counted(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        report.memory_calls += 1
        assert memory_chat is not None
        async for piece in memory_chat(messages):
            yield piece

    store = MemoryStore(db_path, clock) if memory_chat and db_path else None
    session: MemorySession | None = None
    worker: asyncio.Task[None] | None = None

    def start() -> Agent:
        nonlocal session, worker
        if store is None:
            return Agent(agent_chat(None), max_turns=10**6)  # baseline: everything in context
        index = VectorIndex.load(store, embedder.model) if embedder else None
        timed = _Timed(embedder, report.embed_ms) if embedder else None
        consolidator = Consolidator(store, counted, clock, embedder=timed, index=index)
        session = MemorySession(
            store, consolidator, counted, clock=clock, embedder=timed, index=index
        )
        worker = asyncio.create_task(session.run())
        tools = Toolbox(
            memory_tools.MemoryTools(
                store, consolidator, speaker=lambda: OWNER,
                now=lambda: datetime.fromtimestamp(clock()).astimezone(),
                embedder=timed, index=index,
            )
        )  # fmt: skip
        s = session

        def timed_context() -> str:
            t0 = time.perf_counter()
            text = s.context()
            report.briefing_ms.append(1000 * (time.perf_counter() - t0))
            return text

        async def timed_related(text: str) -> str:
            t0 = time.perf_counter()
            block = await s.related(text)
            report.push_ms.append(1000 * (time.perf_counter() - t0))
            return block

        push = s.has_vectors and os.environ.get("MEMORY_PUSH", "").strip().lower() not in (
            "off", "0", "false", "no",
        )  # fmt: skip
        agent = Agent(
            agent_chat(tools.specs()),
            system_prompt=f"{SYSTEM_PROMPT} {memory_tools.PROMPT}",
            tools=tools,
            context=timed_context,
            on_turn=s.submit,
            related=timed_related if push else None,
        )
        agent.preload(s.preload())
        return agent

    agent = start()
    try:
        for step in steps:
            if "t" in step:
                clock.t = datetime.fromisoformat(step["t"]).timestamp()
            if step.get("restart"):
                if session is not None:
                    await session.close(worker)
                    agent = start()
                continue
            text = step.get("say") or step.get("ask")
            if not text:
                continue
            report.turns += 1
            try:
                answer = "".join([p async for p in agent.answer(text)])
            except LLMError as e:
                answer = f"(error: {e})"
            if session is not None:
                await session.drain()  # memory written before the next step, as in a slow chat
            if "ask" in step:
                ok = score(answer, step)
                report.results[step.get("category", "other")].append(ok)
                if not ok:
                    report.failures.append(f"[{step.get('category')}] {text} -> {answer}")
    finally:
        if session is not None:
            await session.close(worker)
    return report


def reset(db: Path, *, everything: bool, yes: bool, ask: Callable[[str], str] = input) -> int:
    """Delete the database (and its WAL files); backups too with `everything`. Asks first."""
    files = [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]
    if everything:
        files += sorted(db.parent.glob(db.name + ".bak*"))
    files = [f for f in files if f.exists()]
    if not files:
        print(f"nothing to reset: {db} does not exist")
        return 0
    print("will delete:\n" + "\n".join(f"  {f}" for f in files))
    if not yes and ask("This cannot be undone. Delete? [y/N] ").strip().lower() not in ("y", "yes"):
        print("cancelled")
        return 1
    for f in files:
        f.unlink()
    kept = sorted(db.parent.glob(db.name + ".bak*"))
    print(f"memory reset: {db}" + (f" (kept {len(kept)} backup file(s))" if kept else ""))
    return 0


async def embed_all(db: Path) -> int:
    """`memory embed`: vectors for every fact and episode summary that lacks one."""
    embedder, status = load_embedder(os.environ)
    if embedder is None:
        print(f"memory embed: no embedder: {status}", file=sys.stderr)
        return 2
    store = MemoryStore(db, time.time)
    index = VectorIndex.load(store, embedder.model)
    facts = episodes = 0
    while True:
        f, e = await backfill(store, embedder, index, 200)
        facts, episodes = facts + f, episodes + e
        if f + e == 0:
            break
    store.close()
    print(f"memory embed: {facts} facts, {episodes} episodes ({embedder.model})")
    return 0


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("memory", help="memory tools (eval, reset, embed)")
    msub = p.add_subparsers(dest="memory_command", required=True)
    e = msub.add_parser("eval", help="run the scripted memory test set against the real LLM")
    e.add_argument("file", nargs="?", default=DEFAULT_FILE)
    e.add_argument("--baseline", action="store_true", help="memory off, whole history in context")
    r = msub.add_parser("reset", help="delete the memory database (MEMORY_DB) and start fresh")
    r.add_argument("--db", help="database file (default: MEMORY_DB, else data/memory.db)")
    r.add_argument("--all", action="store_true", help="also delete the .bak backup files")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    m = msub.add_parser(
        "embed", help="embed every memory that has no vector yet (after a model change)"
    )
    m.add_argument("--db", help="database file (default: MEMORY_DB, else data/memory.db)")


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))
    if args.memory_command == "reset":
        db = args.db or os.environ.get("MEMORY_DB", "").strip() or "data/memory.db"
        return reset(Path(db), everything=args.all, yes=args.yes)
    if args.memory_command == "embed":
        db = args.db or os.environ.get("MEMORY_DB", "").strip() or "data/memory.db"
        return asyncio.run(embed_all(Path(db)))
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print("config error: DEEPSEEK_API_KEY is required", file=sys.stderr)
        return 2
    try:
        steps = [
            json.loads(line) for line in Path(args.file).read_text().splitlines() if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    base_url = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL

    def agent_chat(specs: list[dict[str, Any]] | None) -> ToolChatFn:
        return partial(chat_events, api_key=key, model=model, base_url=base_url, tools=specs)

    memory_model = os.environ.get("MEMORY_MODEL", "").strip() or DEFAULT_MODEL
    memory_chat = (
        None
        if args.baseline
        else partial(chat_stream, api_key=key, model=memory_model, base_url=base_url)
    )
    embedder = None
    if not args.baseline:
        embedder, status = load_embedder(os.environ)
        off = os.environ.get("EMBED", "").strip().lower() in ("off", "0", "false", "no")
        if embedder is None and not off:
            print(f"config error: memory eval needs the embedding model: {status}; "
                  "set EMBED=off for the keyword-only run", file=sys.stderr)  # fmt: skip
            return 2
        print(f"memory search: {status}")
    with tempfile.TemporaryDirectory() as tmp:
        report = asyncio.run(
            evaluate(steps, agent_chat, memory_chat, Path(tmp) / "eval.db", embedder=embedder)
        )
    print(("baseline (long context, no memory)" if args.baseline else "memory") + f" · {args.file}")
    for line in report.lines():
        print(line)
    return 0
