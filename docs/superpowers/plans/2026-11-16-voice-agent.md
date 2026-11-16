# Voice Agent (S2T → LLM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `reachy-demo agent (--mic | --stdin | --file PATH)` prints `you: <turn>` then the streamed `bot: <DeepSeek answer>` for each completed user turn.

**Architecture:** Existing `transcribe_stream` yields one completed turn of text (the server's smart-turn decides when). `Agent.answer` adds conversation history and streams DeepSeek tokens from `chat_stream`. The CLI wires source → S2T → queue → Agent → terminal.

**Tech Stack:** Python 3.12, httpx (SSE streaming), asyncio; pytest + httpx.MockTransport.

**Spec:** `docs/superpowers/specs/2026-11-16-voice-agent-design.md`

## Global Constraints
- ruff (line 100) + mypy strict + pytest green.
- Env: `DEEPSEEK_API_KEY` required, `LLM_MODEL` default `deepseek-chat`, `LLM_BASE_URL` default `https://api.deepseek.com`, `HF_TOKEN` required.
- History: last 10 Q/A pairs. Prompt: user's language, ≤ 3 sentences, plain text without markdown, admit uncertainty.
- LLM failure prints `bot: (error: …)` and listening continues. Missing keys → exit 2.
- No new dependencies. Usage doc updated in the same commit (`docs/usage/agent.md`, README link).

## Review Focus
- DeepSeek 401 (bad key) on the first turn: one clear error line, keeps listening, no traceback.
- Network drop mid-answer: the partial answer stays printed, an error is shown, and the half answer is NOT stored in history.
- Two turns completing while one answer streams: answered in order, none dropped.
- SSE keep-alive or comment lines (`: keep-alive`) and `[DONE]`: ignored, no crash.
- Ctrl-C during an answer: exits 0.

---

### Task 1: DeepSeek streaming client
**Files:** Create `src/reachy_demo/agent/llm.py`; Test `tests/unit/test_llm.py`
**Produces:** `class LLMError(Exception)`; `async def chat_stream(messages: list[dict[str, str]], *, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com", transport: httpx.AsyncBaseTransport | None = None) -> AsyncIterator[str]`
- [ ] Tests: `test_streams_content_deltas` (SSE with 2 deltas, a `: keep-alive` line, an empty-content delta and `[DONE]` → `["Hà ", "Nội"]`; the request has `Authorization: Bearer k`, `stream: true`, and the model and messages), `test_http_error_raises_llm_error` (401 body → LLMError matching "401"), `test_network_error_raises_llm_error` (transport raises httpx.ConnectError → LLMError).
- [ ] Run → FAIL. Implement (POST `{base_url}/chat/completions`, timeout 30 s, parse `data: ` lines). Run → PASS. Commit.

### Task 2: Agent with conversation history
**Files:** Create `src/reachy_demo/agent/agent.py`; Test `tests/unit/test_agent.py`
**Consumes:** a `ChatFn = Callable[[list[dict[str, str]]], AsyncIterator[str]]` (e.g. `functools.partial(chat_stream, api_key=…)`).
**Produces:** `SYSTEM_PROMPT: str`; `class Agent(chat: ChatFn, system_prompt: str = SYSTEM_PROMPT, max_turns: int = 10)` with `async def answer(self, text: str) -> AsyncIterator[str]` and `history: list[dict[str, str]]`.
- [ ] Tests: `test_sends_system_prompt_history_and_question` (2nd call's messages = system, user1, assistant1, user2), `test_history_keeps_last_10_pairs` (12 turns → 20 history messages, the oldest dropped), `test_failed_answer_not_stored` (chat raises LLMError after 1 token → the error propagates and history is unchanged).
- [ ] Run → FAIL. Implement. Run → PASS. Commit.

### Task 3: `agent` CLI pipeline + docs
**Files:** Create `src/reachy_demo/agent/cli.py`; Modify `src/reachy_demo/__main__.py`, `.env.example`, `README.md`, `docs/architecture/05-agent-tools-memory.md` (note at top); Create `docs/usage/agent.md`; Test `tests/unit/test_agent_cli.py`
**Consumes:** `transcribe_stream`, `mic_chunks`, `stdin_chunks`, `wav_chunks`, `Agent`, `chat_stream`, `LLMError`.
**Produces:** `add_parser(sub)`; `async def converse(turns: AsyncIterator[str], agent: Agent, out: TextIO) -> None`; `def run(args) -> int`.
- [ ] Tests: `test_converse_prints_turns_and_answers_in_order` (2 turns → `you: A\nbot: a1\nyou: B\nbot: b1\n`), `test_llm_error_prints_and_continues` (1st answer raises LLMError("401") → `bot: (error: 401)`, the 2nd turn is still answered), `test_turns_arriving_while_answering_are_queued` (slow answer; 3 turns all answered in order), `test_run_without_deepseek_key_returns_2`, `test_run_without_hf_token_returns_2`.
- [ ] Run → FAIL. Implement: a producer task puts turns into an `asyncio.Queue`; the consumer answers one at a time; a `None` sentinel at the end of the source. Wire the subcommand. Run → PASS.
- [ ] Manual: `uv run reachy-demo agent --file data/girl-saying-hello_E_major.wav` → `you: Hello.` + a DeepSeek greeting.
- [ ] Docs + commit.
