# Context Token Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `CONTEXT_MAX_TOKENS` caps every LLM request and `MEMORY_MAX_TOKENS` caps the memory block inside it.

**Architecture:** One `est_tokens()` helper (chars/4). The briefing trims to a token cap instead of `CAP_CHARS`. The agent, each round, sends only the newest whole history turns that fit in what's left after the system prompt, memory block and current turn. The CLI reads both env values and subtracts the tool specs from the budget before handing it to `Agent`.

**Tech Stack:** Python 3.12, pytest, mypy strict, ruff, uv.

**Spec:** [docs/superpowers/specs/2026-11-17-context-budget-design.md](../specs/2026-11-17-context-budget-design.md)

## Global Constraints

- Defaults: `CONTEXT_MAX_TOKENS=8000`, `MEMORY_MAX_TOKENS=600`.
- Token estimate: `ceil(len(text) / 4)`; no new dependency.
- Never send half a turn; never drop the current turn.
- Stored history is unchanged; only what is sent is trimmed. `max_turns` stays as a ceiling.
- Done = `uv run ruff format . && uv run ruff check .`, `uv run mypy`, `uv run pytest` all green.
- The folder is not a git repo: skip commit steps (commit if it becomes one).

## Review Focus

1. Tool specs bigger than `CONTEXT_MAX_TOKENS` → config error at startup, not a silently empty history. (Task 3 test.)
2. `MEMORY_MAX_TOKENS` ≥ `CONTEXT_MAX_TOKENS` → config error; it can never fit. (Task 3 test.)
3. One huge tool result inside the current turn → current turn still sent, history empty, one warning per turn. (Task 2 test.)
4. Context grows between rounds of one turn (tool results appended) → each round re-trims. (Task 2 test.)
5. `memory_eval` baseline (`max_turns=10**6`) keeps "whole history" semantics: it must not get a budget. (Task 2: default `max_tokens=None` means no token trim.)

---

### Task 1: Token helper and briefing cap

**Files:**
- Create: `src/reachy_demo/ops/tokens.py`
- Modify: `src/reachy_demo/memory/briefing.py:13,52-110`, `src/reachy_demo/memory/session.py:37-79`
- Test: `tests/unit/test_tokens.py`, `tests/unit/test_briefing.py:3,113`, `tests/unit/test_memory_session.py`

**Interfaces:**
- Produces: `est_tokens(text: str) -> int` in `reachy_demo.ops.tokens`; `build_briefing(store, speaker, now, tz=None, *, cap_tokens: int = 600) -> str`; `MemorySession(..., memory_max_tokens: int = 600)`; constant `DEFAULT_MEMORY_TOKENS = 600` in `briefing.py` (replaces `CAP_CHARS`).

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tokens.py
def test_est_tokens_rounds_up_chars_over_4() -> None:
    assert est_tokens("") == 0
    assert est_tokens("abcd") == 1
    assert est_tokens("abcde") == 2
```

In `test_briefing.py`: change the import to `DEFAULT_MEMORY_TOKENS`, change line 113 to
`assert est_tokens(text) <= DEFAULT_MEMORY_TOKENS or text.count("fact ") == 5`, and add
`test_briefing_respects_cap_tokens`: same fixture data, `build_briefing(s, OWNER, NOW, UTC, cap_tokens=300)` → `est_tokens(text) <= 300 or text.count("fact ") == 5`, and fewer facts than with the default.

In `test_memory_session.py` add `test_session_passes_memory_max_tokens`: monkeypatch `reachy_demo.memory.session.build_briefing` to record `cap_tokens`; `MemorySession(..., memory_max_tokens=123).context()` → recorded `123`.

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/unit/test_tokens.py tests/unit/test_briefing.py tests/unit/test_memory_session.py -v`
Expected: FAIL (`ModuleNotFoundError: reachy_demo.ops.tokens`, unknown kwarg `cap_tokens`).

- [ ] **Step 3: Implement**

`est_tokens` with a `# shortcut: chars/4 is ~10-20% off for Vietnamese, use a real tokenizer if budgets get tight` comment. In `build_briefing`, both trim loops compare `est_tokens(text) > cap_tokens`. `MemorySession` stores `memory_max_tokens` and passes `cap_tokens=` at `session.py:74`.

- [ ] **Step 4: Run and see them pass**

Run: same command. Expected: PASS.

---

### Task 2: Agent token trim

**Files:**
- Modify: `src/reachy_demo/agent/agent.py:38-115`
- Test: `tests/unit/test_agent.py`

**Interfaces:**
- Consumes: `est_tokens` (Task 1).
- Produces: `Agent(..., max_tokens: int | None = None)`. `None` = no token trim (today's behaviour). `max_tokens` is the budget for system + context + history + current turn (the CLI has already removed tool specs).

- [ ] **Step 1: Write failing tests** (use `FakeChat` / `_tool_chat`; message size = `est_tokens(json.dumps(msg))`)

- `test_token_budget_drops_oldest_whole_turns`: 3 turns with ~400-char questions, `max_tokens` fitting system + 1 old turn + current → `chat.calls[-1]` holds system, turn 2, turn 3 only; `agent.history` still has all 3.
- `test_token_budget_keeps_tool_turn_whole`: history turn with a tool call + tool reply; budget fits only part of it → neither message is sent (no `role: tool` without its `tool_calls`).
- `test_current_turn_sent_even_over_budget`: `max_tokens=10`, a 2000-char question → sent messages == `[system, user question]`; `caplog` has exactly one `WARNING` containing `"context budget"`.
- `test_budget_rechecked_each_round`: tool round whose result is 2000 chars; second round's messages drop the history turn that the first round still sent.
- `test_no_max_tokens_sends_full_history`: `max_tokens=None`, 5 long turns → all sent (only `max_turns` applies).

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/unit/test_agent.py -v`
Expected: the 5 new tests FAIL (unknown kwarg `max_tokens`).

- [ ] **Step 3: Implement**

Add a private `_fit(self, fixed_tokens: int) -> list[dict[str, Any]]` that walks `self._turns` newest → oldest, adding each whole turn's `est_tokens(json.dumps(msg))` sum while `used + turn ≤ max_tokens - fixed`; stops at the first turn that doesn't fit (no skipping to older smaller ones). Use it in `answer()` where `messages` is built. Warn once per `answer()` call via `log.warning("context budget exceeded by the current turn", extra={...})` when `fixed` alone is over budget.

- [ ] **Step 4: Run and see them pass**

Run: `uv run pytest tests/unit/test_agent.py -v`. Expected: PASS, including existing tests.

---

### Task 3: Env settings, CLI wiring, docs

**Files:**
- Modify: `src/reachy_demo/agent/cli.py:215-300,348-370`, `.env.example:29-31`, `docs/usage/agent.md` (settings + troubleshooting table), `docs/usage/memory.md`
- Test: `tests/unit/test_agent_cli.py`

**Interfaces:**
- Consumes: `Agent(max_tokens=)` (Task 2), `MemorySession(memory_max_tokens=)` (Task 1), `est_tokens` (Task 1).
- Produces: `memory_from_env(api_key, base_url, memory_max_tokens: int = 600)`.

- [ ] **Step 1: Write failing tests**

- `test_run_bad_token_budget_returns_2` (parametrize `CONTEXT_MAX_TOKENS` / `MEMORY_MAX_TOKENS` × `"abc"`, `"0"`, `"-5"`): `cli.run(_args()) == 2`, the var name in stderr.
- `test_run_memory_budget_must_be_below_context`: `CONTEXT_MAX_TOKENS=500`, `MEMORY_MAX_TOKENS=600` → 2, stderr mentions both names.
- `test_run_tool_specs_over_budget_returns_2`: `CONTEXT_MAX_TOKENS=10`, `MEMORY_MAX_TOKENS=5`, with tools loaded → 2, stderr contains `"tool specs"`.
- `test_run_passes_budget_to_agent`: spy on `cli.Agent`; `CONTEXT_MAX_TOKENS=8000` → `max_tokens == 8000 - est_tokens(json.dumps(tools.specs()))` (or `8000` when no tools).

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/unit/test_agent_cli.py -v`. Expected: new tests FAIL.

- [ ] **Step 3: Implement**

In `run()`, after the `VISION_PROVIDER` check, parse both with a small local `positive_int(name, default) -> int | None` that prints `config error: {name} must be a positive integer, got {raw!r}` and returns `None` → `return 2`. Then the memory < context check (`config error: MEMORY_MAX_TOKENS must be below CONTEXT_MAX_TOKENS`). Pass `memory_max_tokens` to `memory_from_env`. In `make_agent`, `budget = context_max - (est_tokens(json.dumps(tools.specs())) if tools else 0)`; if `budget <= memory_max`, print `config error: tool specs (~N tokens) leave no room in CONTEXT_MAX_TOKENS` and return 2 (raise a small exception inside `make_agent` caught at its two call sites, or compute the budget once before the `with` block — builder's choice). Pass `max_tokens=budget`.

Docs: add the two lines to `.env.example` after `MEMORY_MODEL`:

```
CONTEXT_MAX_TOKENS=8000          # cap for each LLM request: system prompt + memory + tools + history (approx, chars/4)
MEMORY_MAX_TOKENS=600            # memory block share inside CONTEXT_MAX_TOKENS
```

Add both to the settings list in `docs/usage/agent.md` and `docs/usage/memory.md`, and the three new `config error` lines to the troubleshooting table in `agent.md`.

- [ ] **Step 4: Run the full check**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: all green.
