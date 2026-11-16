# Live Translate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `reachy-demo translate --mic | --file x.wav` prints a live vi<->en translation in the terminal.

**Architecture:** A standalone `voice/translate.py` opens an OpenAI Realtime WebSocket, streams 24 kHz PCM16 in, and yields `Partial`/`Final` text events. Audio sources (`audio/sources.py`) and a terminal printer (`voice/translate_cli.py`) sit on either side. No robot or conversation code is touched.

**Tech Stack:** Python 3.12, `websockets`, `numpy`, stdlib `wave`, new optional extra `mic` = `sounddevice`. pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-11-16-live-translate-design.md`

## Global Constraints

- Ruff (line-length 100) + mypy strict + pytest all green (see `src/CLAUDE.md`).
- Realtime input audio: PCM16, 24 kHz, mono.
- Server turn detection ON for this tool.
- API key from `OPENAI_API_KEY` (env/.env); never logged; no hardcoded credentials.
- Realtime event names live only in `voice/translate.py` (`[verify]` against current OpenAI docs first).
- Inject dependencies (WS url, key) through constructors; no module singletons. No `print` outside the CLI module.
- `sounddevice` is optional (`mic` extra); ask before adding. Already approved by user in the spec.
- Shortcuts with a known limit get a `# shortcut: <limit>, <when to upgrade>` comment.

## Review Focus

- Empty WAV / zero-length audio: exit 0 with no output, no hang.
- WAV that is not 16-bit mono (stereo, 8-bit, 44.1 kHz): converted or rejected with a clear message, not garbage.
- Missing `OPENAI_API_KEY`: exit 2 with a message naming the variable.
- WS closes or errors mid-stream: error printed, exit 1, mic stream closed.
- Ctrl-C during `--mic`: clean exit 0, mic released.
- No `sounddevice` installed with `--mic`: message pointing at `uv sync --extra mic`.

## Prerequisite finding (resolve before Task 1)

`src/` has no `__init__.py`, `config.py` or `__main__.py` in git (only stale `.pyc` in `src/__pycache__`), and `pyproject.toml` says `packages = ["reachy_demo"]` while the code sits in `src/`. `tests/integration/test_cli.py` runs `python -m reachy_demo`. This plan assumes the package is importable as `reachy_demo` and has a `__main__.py`. If those files were lost, restore them first; do not recreate from guesswork.

---

### Task 1: Realtime translate client

**Files:**
- Create: `src/voice/translate.py`
- Test: `tests/unit/test_translate.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Partial: text: str`
  - `@dataclass(frozen=True) class Final: text: str`
  - `async def translate_stream(pcm24k: AsyncIterator[bytes], *, api_key: str, url: str = REALTIME_URL) -> AsyncIterator[Partial | Final]`
  - `REALTIME_URL: str` constant.

- [ ] **Step 1: Spike the protocol.** Read the current OpenAI Realtime docs (use context7 / web) for: the endpoint and model for speech translation or transcription, the `session.update` payload (instructions "translate between Vietnamese and English, output only the translation", server VAD on, text output), the audio append event, and the text delta/done event names. Write the confirmed names in a comment block at the top of `translate.py`. If no direct translation mode exists, fall back to: transcription session + a text-only response per utterance with the translate instruction.
- [ ] **Step 2: Write failing test** `test_stream_yields_partials_then_final` using an in-process `websockets.serve` fake that, after receiving an audio-append message, sends two delta events then a done event. Assert the output is `[Partial("Hel"), Partial("Hello"), Final("Hello")]` (partials are cumulative).
- [ ] **Step 3: Write failing test** `test_sends_auth_header_and_session_update`: fake server records headers and first message; assert `Authorization: Bearer <key>` and that the first message is the session config with server VAD on.
- [ ] **Step 4: Write failing test** `test_ws_close_mid_stream_raises`: server closes after the first delta; assert `translate_stream` raises `ConnectionError` (wrap `websockets.ConnectionClosedError`).
- [ ] **Step 5: Run** `uv run pytest tests/unit/test_translate.py -v`. Expected: 3 FAIL (module missing).
- [ ] **Step 6: Implement** `translate_stream`: one task pumps `pcm24k` chunks into base64 append events, the main loop reads server events and yields. Run both under `asyncio.TaskGroup` so failure of either cancels the other.
- [ ] **Step 7: Run tests.** Expected: 3 PASS. Then `uv run ruff check . && uv run mypy`.
- [ ] **Step 8: Commit** `feat(voice): realtime translate stream client`

### Task 2: Audio sources (file and mic)

**Files:**
- Create: `src/audio/sources.py`
- Modify: `pyproject.toml` (add `mic = ["sounddevice>=0.5"]` to `[project.optional-dependencies]`; add `sounddevice.*` to the mypy `ignore_missing_imports` list)
- Test: `tests/unit/test_sources.py`, fixture `tests/fixtures/hello_16k.wav` (1 s generated sine, committed)

**Interfaces:**
- Produces:
  - `async def wav_chunks(path: Path, *, realtime: bool = True, chunk_ms: int = 100) -> AsyncIterator[bytes]`: yields 24 kHz mono PCM16 bytes; raises `ValueError` for non-PCM16 WAV.
  - `async def mic_chunks(*, chunk_ms: int = 100) -> AsyncIterator[bytes]`: yields 24 kHz mono PCM16 from the default input; raises `RuntimeError("install the mic extra: uv sync --extra mic")` if `sounddevice` is missing; always closes the stream on exit.
  - `def to_24k_mono_pcm16(samples: np.ndarray, rate: int) -> bytes` (stereo is averaged to mono; linear-interpolation resample).

- [ ] **Step 1: Write failing tests:** `test_to_24k_length` (1 s of 16 kHz in -> 48000 bytes out), `test_stereo_is_downmixed`, `test_wav_chunks_total_bytes` (fixture 1 s -> 48000 bytes total, each chunk <= 4800 bytes, `realtime=False`), `test_empty_wav_yields_nothing`, `test_8bit_wav_rejected` (`ValueError`).
- [ ] **Step 2: Run:** expected FAIL.
- [ ] **Step 3: Implement** the three functions. Linear-interp resample with `np.interp`; mark `# shortcut: linear resample, no anti-alias filter; upgrade if accuracy suffers on 44.1k+ sources`. `wav_chunks` sleeps `chunk_ms` between chunks when `realtime`. `mic_chunks` bridges the `sounddevice.RawInputStream` callback to an `asyncio.Queue` via `loop.call_soon_threadsafe`.
- [ ] **Step 4: Run tests.** Expected PASS (mic function untested; hardware, mark with a manual check in Task 3).
- [ ] **Step 5: Commit** `feat(audio): wav and mic sources at 24k pcm16`

### Task 3: CLI command and terminal printer

**Files:**
- Create: `src/voice/translate_cli.py`
- Modify: `src/__main__.py` (add a `translate` subcommand that calls `translate_cli.run(args)`), `.env.example` (no new vars; confirm `OPENAI_API_KEY` present)
- Test: `tests/unit/test_translate_cli.py`

**Interfaces:**
- Consumes: `translate_stream`, `Partial`, `Final` (Task 1); `wav_chunks`, `mic_chunks` (Task 2).
- Produces:
  - `def render(events: AsyncIterator[Partial | Final], out: TextIO) -> Awaitable[None]` (partial: `"\r" + text` padded to clear the line; final: text then `"\n"`).
  - `def run(args: argparse.Namespace) -> int` (exit codes: 0 ok, 1 stream error, 2 usage/config).
  - CLI: `reachy-demo translate (--mic | --file PATH)`, mutually exclusive, one required.

- [ ] **Step 1: Write failing tests:** `test_render_partial_then_final` (StringIO output equals `"\rHel\rHello\nHello\n"`-style per the rule above; assert the last line is `Hello\n` and no stray partial remains); `test_run_without_api_key_returns_2` (empty env, stderr names `OPENAI_API_KEY`); `test_run_missing_file_returns_2`; `test_run_stream_error_returns_1` (monkeypatch `translate_stream` to raise `ConnectionError`).
- [ ] **Step 2: Run:** expected FAIL.
- [ ] **Step 3: Implement** `render` and `run`. `run` reads the key from env after loading `.env` the same way the existing `config.py` does (reuse it; do not add a second loader). `KeyboardInterrupt` -> return 0.
- [ ] **Step 4: Wire** the `translate` subcommand into `__main__.py` following the existing `check-config` pattern.
- [ ] **Step 5: Run** `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`. Expected: all green.
- [ ] **Step 6: Manual check (needs real key, cannot be automated):** `uv sync --extra mic && uv run reachy-demo translate --file tests/fixtures/hello_16k.wav`, then `--mic`, speaking English then Vietnamese. Expected: partials update in place, a final line per utterance, translated into the other language. Ctrl-C exits cleanly. Record the result in the PR/commit message.
- [ ] **Step 7: Commit** `feat(cli): translate command (mic/file) with live terminal output`

---

## Self-review notes

- Spec coverage: mic/file input (T2/T3), vi<->en instruction (T1 Step 1), partial+final output (T1/T3), server VAD on (T1), realtime-paced file (T2), optional `mic` extra (T2), fake-WS test (T1), no robot code touched (all).
- Types consistent: `Partial`/`Final`/`translate_stream`/`wav_chunks`/`mic_chunks` are named identically across tasks.
- Not covered on purpose (non-goals): robot mic, TTS, other languages, offline models.
