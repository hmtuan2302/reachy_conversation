# Startup Greeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reachy says "Hi, I'm Reachy Mini, nice to meet you." once all startup has finished.

**Architecture:** Local STT models are loaded in `run` before the conversation. `converse` gains a `greeting` that is played through the existing answer and barge-in path, once listening runs.

**Tech Stack:** Python 3.12, asyncio, pytest.

**Spec:** [docs/superpowers/specs/2026-11-19-startup-greeting-design.md](../specs/2026-11-19-startup-greeting-design.md)

## Global Constraints

- Text, exactly: `Hi, I'm Reachy Mini, nice to meet you.`
- No "you:" line, no `on_answer`, and the greeting is not added to agent history or memory.
- Every feature change updates `docs/FEATURES.md` and its usage doc (`docs/usage/agent.md`).

## Review Focus

1. **A stop said during the greeting** must cut it off, the same as during an answer. Test: `test_greeting_can_be_stopped`.
2. **A turn heard during the greeting that is not a stop** is dropped, not answered later as if the user said it. Test: `test_greeting_drops_side_talk`.
3. **A TTS failure on the greeting** must not end the conversation. The answer path already catches `TTSError`/`PlayerError`. Test: `test_greeting_speech_error_keeps_listening`.
4. **A local STT load failure** exits 2 with `error:` before connecting to the robot. Test: `test_agent_local_stt_load_failure_exits_2`.
5. **The greeting is missing from history**, so the model never repeats "nice to meet you" as context. Covered by `test_greeting_prints_and_speaks_first`, which asserts the agent received no greeting.

---

### Task 1: Greeting in `converse`, local STT preload, docs

**Files:**
- Modify: `src/reachy_demo/agent/cli.py` (`converse`, `run`, new `GREETING`)
- Modify: `docs/usage/agent.md`, `docs/FEATURES.md`
- Test: `tests/unit/test_agent_cli.py`

**Interfaces:**
- Consumes: `models_from_env(env) -> LocalModels`, `SttError`, `transcribe_local(pcm, *, models=)` from `voice/local_stt.py`.
- Produces: `GREETING: str`, `converse(..., greeting: str | None = None)`.

- [ ] **Step 1: Write the failing tests** (reuse the file's `_turns`, `_agent`, `_classify` helpers)

```python
async def test_greeting_prints_and_speaks_first():
    out, spoken = io.StringIO(), []
    async def speak(t): spoken.append(t)
    agent = _agent()
    await cli.converse(_turns("A"), agent, out, speak=speak, greeting="Hi.")
    assert out.getvalue().startswith("bot: Hi.\n") and spoken[0] == "Hi."
    # the agent only ever answered "A"

async def test_greeting_can_be_stopped():          # classify says stop for "stop"; speak sleeps 0.3 s
    # turns: "stop" arrives during the greeting -> output has "  (stopped)", no "you: stop" answer

async def test_greeting_drops_side_talk():         # "hmm" during the greeting -> "(ignored: hmm)"

async def test_greeting_speech_error_keeps_listening():   # speak raises TTSError -> "speech error", then "A" is answered

def test_agent_local_stt_load_failure_exits_2(monkeypatch, capsys):   # STT_PROVIDER=local, models_from_env raises SttError("boom") -> 2, "boom" in err
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_agent_cli.py -q`
Expected: the 5 new tests FAIL, and the existing tests pass.

- [ ] **Step 3: Implement**

`converse`:
- Split `answer` so a fixed text can go through the same printing and speaking path, without "you:" and without `agent.answer`.
- `answer_with_barge_in` takes the coroutine to run instead of the text.
- Before the main loop: if `greeting`, run it, with barge-in when `classify` is set. Its follow-up, or the next `queue.get()`, becomes the first `text`.

`run`:
- Preload local STT models after the provider and key checks and before `connect_robot`. Catch `SttError`, print `error: {e}`, and return 2.
- Pass `greeting=GREETING` to both `converse` calls.
- The existing tests that patch `transcribe_local` with `STT_PROVIDER=local` must also patch `cli.models_from_env`.

- [ ] **Step 4: Run tests, lint, types**

Run: `uv run pytest tests/unit -q && uv run ruff check src tests && uv run mypy`
Expected: all pass.

- [ ] **Step 5: Docs.** In `docs/usage/agent.md`, add a "Startup greeting" line. In `docs/FEATURES.md`, update row 2 and the "Last updated" line.

- [ ] **Step 6: Commit** with the message `feat(agent): greet once startup has finished`.
