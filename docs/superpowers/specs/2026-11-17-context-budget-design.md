# Context token budget — short design

## Intent

Memory grows for months. The prompt sent to the LLM must not. One env setting caps the whole
prompt, so cost and latency stay predictable, and it gives a fixed space for the later
"smart context selection" work to fill.

## Goals / non-goals

- Goal: `CONTEXT_MAX_TOKENS` caps every LLM request (system prompt + memory block + tool specs +
  history + current turn). `MEMORY_MAX_TOKENS` caps the memory block inside it.
- Non-goal: choosing *which* memories go in (that is the next design), exact token counts,
  summarizing dropped history.
- Anti-goal: dropping half a turn (a tool reply without its tool call breaks the API), or
  dropping the current turn.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **A (picked)** | One total budget; memory gets its share; history fills the rest, oldest whole turn dropped first | One number controls the request size; touches `agent.py` and `briefing.py` |
| B | Memory budget only (replace `CAP_CHARS`) | Tiny change, but history stays capped by turn count, which means nothing once tool results get long |

Picked A: a turn-count cap stops meaning anything after months of use.

## Design

- **Env** (`.env.example`, `docs/usage/agent.md`, `docs/usage/memory.md`):
  `CONTEXT_MAX_TOKENS=8000`, `MEMORY_MAX_TOKENS=600`. Read in `agent/cli.py` like the other
  chat settings; a non-integer or non-positive value is a startup error. [my call]
- **Counting**: `tokens(text) = ceil(len(text) / 4)`, the estimate the code already uses.
  No tokenizer dependency (DeepSeek's is not on PyPI). Roughly 10–20 % off for Vietnamese,
  so the budget is approximate. Comment it as a `shortcut:`.
- **Memory**: `build_briefing(..., cap_tokens=600)` replaces `CAP_CHARS` with
  `cap_tokens * 4`; existing trim order unchanged (facts down to 5, then follow-ups).
  `MemorySession` takes `memory_max_tokens` and passes it on.
- **Agent**: `Agent(max_tokens=8000)` beside `max_turns` (kept as a hard ceiling). Each round,
  before calling the LLM: `fixed = system + context + tool specs (json) + current turn`;
  keep the newest whole history turns whose total fits in `max_tokens - fixed`. If nothing
  fits, send no history and log a warning once per turn; never drop the current turn.
  Stored history is not changed, only what is sent (so a shorter memory block next round lets
  more history back in).
- **Tests** (`tests/unit/test_agent.py`, `test_briefing.py`): oldest turns dropped first;
  a tool turn is kept or dropped whole; current turn always sent even over budget;
  briefing respects `cap_tokens`; env parsing rejects bad values.

## Left to the builder

Exact log field names; where the shared `tokens()` helper lives (one small function, used by
agent and briefing).
