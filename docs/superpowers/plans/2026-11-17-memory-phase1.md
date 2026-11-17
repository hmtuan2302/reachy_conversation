# Memory Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reachy remembers the owner's facts and past conversations across restarts, recalls them on request, and abstains when it doesn't know.

**Architecture:** One SQLite file (`data/memory.db`) with FTS5. A session object builds a briefing once per session (SQL only), logs each finished turn, and runs a background worker that consolidates facts (DeepSeek extract → FTS similar → decide) and summarizes sessions. `recall/remember/forget` are an agent `ToolSet`.

**Tech Stack:** Python 3.12, stdlib `sqlite3` (FTS5), DeepSeek via `agent/llm.py`, pytest-asyncio. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-11-17-memory-design.md](../specs/2026-11-17-memory-design.md) §5–§7, §12

## Global Constraints
- Single user in Phase 1: every turn is `OWNER = Speaker(1, "owner")`.
- Hot path: no LLM call for memory; briefing built once per session, cap ~600 tokens (2400 chars).
- Warm path never raises into a turn; queue max 64, drop oldest.
- `memory/` never imports `agent`, `robot`, `voice`, `perception` (it takes a `ChatFn`-shaped callable).
- `MEMORY=off` → agent behaves exactly as before. `MEMORY_DB` default `data/memory.db`. `MEMORY_MODEL` default `deepseek-chat`.
- ruff, mypy strict, pytest green. Usage doc `docs/usage/memory.md`.

## Review Focus
1. Empty or stopword-only `recall` query → `{"found": false}`, never an FTS syntax error. (Task 4)
2. Quotes / FTS operators in user text (`"`, `*`, `AND`, `-`) → treated as words, no crash. (Task 1)
3. LLM returns prose or fenced JSON → parsed or retried once, then dropped, no exception. (Task 5)
4. App killed mid-session → next start summarizes the old session and re-queues unconsolidated turns. (Task 6)
5. Guest speaker (M8 seam) → no writes, scoped reads. (Tasks 1, 5, 7)

---

### Task 1: Store (`memory/store.py`)
Schema v1 (`meta, people, facts+facts_fts, episodes+episodes_fts, follow_ups, event_log`), migrations with backup, `Speaker`, `OWNER`, dataclasses `Person, Fact, Episode, FollowUp`, CRUD, scoped FTS search, `match_query()`.
Tests `tests/unit/test_memory_store.py`: fresh DB has owner row; reopen no-op; failed migration rolls back; FTS matches `tuan` ↔ `Tuấn`; FTS operators in query are safe; guest scope; invalidate keeps history; time-overlap search finds expired facts; forget_person deletes.

### Task 2: Time words (`memory/timeparse.py`)
`parse(text: str, now: datetime) -> tuple[float, float] | None` for VN+EN day words, parts of day, weeks, months, "N days ago", weekdays, ISO and d/m dates.
Tests `tests/unit/test_timeparse.py`: table tests with a fixed aware `now`.

### Task 3: Briefing (`memory/briefing.py`)
`build_briefing(store, speaker, now: float, tz=None) -> str` per spec §7.3 blocks 1–4, 6, 7; `ago(seconds) -> str`.
Tests `tests/unit/test_briefing.py`: no-name line; 5 newest episodes; sensitive excluded; cap trims facts first keeping ≥ 5, never episodes.

### Task 4: Recall (`memory/recall.py`)
`recall(store, speaker, query, *, person=None, kind="any", when=None, limit=5, now: datetime) -> dict` per spec §7.5 (time-window episodes even without word match, abstention, `[said, 3 days ago]` tags, touch use counts).
Tests `tests/unit/test_recall.py`.

### Task 5: Consolidator (`memory/prompts.py`, `memory/consolidate.py`)
`ask_json(chat, system, user, timeout_s) -> dict | None`; `worth_keeping(text) -> bool`; `Consolidator(store, chat, clock).consolidate(text, speaker, *, answer="", event_id=None, explicit=False) -> list[Change]`.
Tests `tests/unit/test_consolidate.py` with a scripted fake chat: ADD, NOOP, UPDATE, INVALIDATE (old expired + replaced_by), unknown id → ADD, invalid JSON retry then drop, guest no write, sensitive only when explicit, name sets `people.name`, follow-up closed by a matching fact.

### Task 6: Episodes + session (`memory/episodes.py`, `memory/session.py`)
`summarize_session(store, chat, session_id, person_id, clock) -> int | None`; `MemorySession(store, consolidator, chat, clock, speaker, gap_s=600)` with `context() -> str`, `submit(turn: list[dict]) -> None`, `run()`, `drain()`, `close()`, `last_turns_for_preload(max_gap_s=7200) -> list[dict]`.
Tests `tests/unit/test_memory_session.py`: briefing cached within a session, rebuilt after 10 min gap and old session summarized; drop-oldest; crash recovery; gate skips small talk; close summarizes.

### Task 7: Tools (`agent/tools/memory.py`)
`MemoryTools(store, consolidator, speaker, clock)` → `recall`, `remember`, `forget` specs and calls; owner-only in code; forget flows (one match, several, everything needs confirmed).
Tests `tests/unit/test_memory_tools.py`.

### Task 8: Agent + CLI wiring
`Agent(..., context=None, on_turn=None)`, `Agent.preload(messages)`; memory prompt rules; CLI builds memory unless `MEMORY=off`, registers `MemoryTools` in both modes, runs the worker, backs up on exit. Tests extend `tests/unit/test_agent.py`, `tests/unit/test_agent_cli.py`.

### Task 9: Eval (`memory/cli.py`, `tests/replay/memory_eval.jsonl`)
`reachy-demo memory eval [FILE] [--baseline]`: simulated clock, restarts, per-category score, abstention check. Integration test marked `@pytest.mark.integration`.

### Task 10: Docs
`docs/usage/memory.md` + README link; `docs/usage/agent.md` env vars; `.env.example`; architecture 05 §5 pointer; spec updated to what was built.
