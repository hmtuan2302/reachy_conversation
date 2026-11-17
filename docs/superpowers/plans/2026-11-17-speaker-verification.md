# Speaker Verification and Owner Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "Only follow my command" turns on owner mode. In owner mode, each spoken turn is checked against the owner's voiceprint: guests are ignored, unsure speakers are asked to repeat, and `dance` and `look_toward_sound` refuse non-owners.

**Architecture:**
- `voice/speaker_id.py` holds the pure logic. It keeps the speech audio heard since the last turn, scores it with an injected `embed` function, and returns a verdict.
- The sherpa-onnx CAM++ extractor is the real `embed`, loaded only by the CLI.
- `agent/tools/owner.py` holds the mode state, the `owner_mode` tool and `gated_turns()`. `gated_turns()` wraps the transcript stream before `converse`.
- Tools get `speaker=mode.speaker`. It returns `OWNER` whenever owner mode is off, so today's behaviour is unchanged.

**Tech Stack:** Python 3.12, sherpa-onnx (`speaker` extra, already declared), numpy, pytest, mypy strict, ruff, uv.

**Spec:** [docs/superpowers/specs/2026-11-17-speaker-verification-design.md](../specs/2026-11-17-speaker-verification-design.md)

## Global Constraints

- **Model:** `models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`, from `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` (the "recongition" typo is in the real URL). Path comes from `SPEAKER_MODEL`.
- **Voiceprint:** `data/voice/owner.npy`, the mean of L2-normalised embeddings, normalised again.
- **Verdict:**
  - cosine ≥ `SPEAKER_HIGH`: `"owner"`
  - cosine < `SPEAKER_LOW`: `"guest"`
  - otherwise, or less than 1.0 s of buffered speech: `"unsure"`
- **Unsure reply, exact text:** `I'm not sure that's you, please say it again.`
- **No new dependencies besides `sherpa-onnx`**, which is already the `speaker` extra. Install with `uv sync --extra speaker`.
- **Owner mode:** starts off on every run and is never persisted. Typed input (`--type`) always counts as the owner.
- **When owner mode is off,** behaviour is identical to today (every speaker is `OWNER`).
- **Missing sherpa-onnx, model or voiceprint:** the agent still runs, and `owner_mode` returns `{"error": "not_enrolled: run reachy-demo enroll first"}`.
- **Done** = `uv run ruff format . && uv run ruff check .`, `uv run mypy`, `uv run pytest -m "not integration"` all green.
- Each commit updates `docs/FEATURES.md` and the usage doc when it changes user-facing behaviour (src/CLAUDE.md workflow).
- Leave the user's unrelated uncommitted changes alone: `git add` only the files listed in each task.

## Review Focus

1. **Silence between turns:** speech that was never transcribed (a cough, side talk) stays in the buffer and pollutes the next score. Expected: the buffer keeps only the last `KEEP_S = 8.0` s. (Task 1 test `test_buffer_keeps_last_seconds`.)
2. **Mode off still scores:** every turn must call `take()`, even with owner mode off. Otherwise stale audio from earlier turns is scored when the mode is turned on. (Task 3 test `test_mode_off_still_takes_each_turn`.)
3. **Turning the mode on from a guest voice:** an unverified voice saying "only follow me" must not lock the owner out. Expected: `NOT_OWNER`. (Task 3 test `test_guest_cannot_toggle`.)
4. **No voiceprint, or sherpa-onnx not installed:** the agent must start, and asking for owner mode must give a clear spoken error. It must not crash. (Task 3 test `test_not_enrolled`; Task 5 `speaker_gate_from_env` returns `None` with a printed reason.)
5. **WAV at a rate other than 16 kHz** (my_voice1.wav is 48 kHz): enrollment must score the same voice high. The extractor gets the true rate and resamples internally. (Task 2 integration test `test_enroll_scores_second_owner_clip_high`.)

---

### Task 1: Speaker gate (pure logic)

**Files:**
- Create: `src/reachy_demo/voice/speaker_id.py`
- Test: `tests/unit/test_speaker_id.py`

**Interfaces:**
- Produces:
  - `Embed = Callable[[np.ndarray, int], np.ndarray]`: float32 samples in [-1, 1] and the sample rate in; one embedding out.
  - `Verdict = Literal["owner", "guest", "unsure"]`
  - `normalize(v: np.ndarray) -> np.ndarray`
  - `voiceprint(embeddings: list[np.ndarray]) -> np.ndarray`: mean of the normalised embeddings, normalised again.
  - `pcm16_to_float(pcm: bytes) -> np.ndarray`
  - `class SpeakerGate(embed: Embed, print_: np.ndarray, high: float, low: float, min_s: float = 1.0, keep_s: float = KEEP_S)` with:
    - `feed(chunk: bytes, speaking: bool) -> None`: 24 kHz PCM16; keeps only `speaking` chunks.
    - `take() -> tuple[Verdict, float | None]`: scores and clears the buffer. The score is `None` when under `min_s`.
  - Constants `KEEP_S = 8.0` and `MIN_S = 1.0`.

- [ ] **Step 1: Write failing tests**

Use a fake `embed` that returns a fixed vector chosen by the first sample's sign, so scores are known. Each test below follows the pattern of the first.

```python
def fake_embed(x: np.ndarray, rate: int) -> np.ndarray:
    assert rate == 24_000
    return np.array([1.0, 0.0]) if x[0] > 0 else np.array([0.6, 0.8])  # cos 1.0 / 0.6

def test_owner_guest_unsure() -> None:
    gate = SpeakerGate(fake_embed, np.array([1.0, 0.0]), high=0.7, low=0.4)
    gate.feed(POS_1S * 2, speaking=True)
    assert gate.take() == ("owner", pytest.approx(1.0))
    gate.feed(NEG_1S * 2, speaking=True)
    assert gate.take()[0] == "unsure"      # 0.6 is between low and high
    # a guest: SpeakerGate(..., high=0.9, low=0.7) with NEG audio gives "guest"
```

- `test_too_short_is_unsure`: 0.5 s of speech gives `("unsure", None)`, and `embed` is never called.
- `test_silence_not_kept`: `feed(..., speaking=False)` adds nothing.
- `test_take_clears`: a second `take()` with no new audio gives `("unsure", None)`.
- `test_buffer_keeps_last_seconds`: feed 10 s; the embed input length is `8 * 24_000`.
- `test_voiceprint_is_unit_mean`: `voiceprint([[2,0],[0,3]])` ≈ `[0.7071, 0.7071]`.

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_speaker_id.py -v`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `speaker_id.py`. Store chunks in a `deque` trimmed to `keep_s`, the same way as `SpeechBuffer` in `agent/tools/voice.py`. Do not import sherpa-onnx here.
- [ ] **Step 4: Run** the same command. Expected: PASS. Then run ruff and mypy on the file.
- [ ] **Step 5: Commit** `feat(speaker): speaker gate scores speech since the last turn`

### Task 2: sherpa-onnx extractor and `reachy-demo enroll`

**Files:**
- Modify: `src/reachy_demo/voice/speaker_id.py` (add the loader)
- Create: `src/reachy_demo/voice/enroll_cli.py`
- Modify: `src/reachy_demo/__main__.py` (register `enroll`)
- Modify: `.env.example`: replace `GATE_THRESHOLD=...` with `SPEAKER_MODEL=`, `SPEAKER_HIGH=` and `SPEAKER_LOW=`, plus comments.
- Test: `tests/unit/test_enroll_cli.py`, `tests/integration/test_speaker_model.py`

**Interfaces:**
- Consumes: `Embed`, `voiceprint`, `normalize` from Task 1.
- Produces:
  - `DEFAULT_MODEL = Path("models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx")`
  - `VOICEPRINT = Path("data/voice/owner.npy")`
  - `load_sherpa_embed(model: Path) -> Embed`: raises `ImportError` or `FileNotFoundError`.
  - `read_wav(path: Path) -> tuple[np.ndarray, int]`: mono float32 and the rate. Stereo is averaged.
  - `enroll_cli.add_parser(sub)` and `enroll_cli.run(args, embed: Embed | None = None) -> int`

**CLI:** `reachy-demo enroll --file A.wav [--file B.wav ...] [--test X.wav ...] [--out PATH]`
- Embeds each `--file` and saves the voiceprint.
- Prints `self-score: 0.NN`: the first half vs the second half of the first file, each scored against the other.
- For each `--test` file, prints `test X.wav: 0.NN`.
- Exits with code 2 and a message on a missing file or a clip under 3 s.

Use `sherpa_onnx.SpeakerEmbeddingExtractor(SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1))`. For each call, run `create_stream()`, then `accept_waveform(rate, samples)`, then `input_finished()`, then `np.array(extractor.compute(stream))`. Check the exact names against the installed version (`help(sherpa_onnx.SpeakerEmbeddingExtractor)`) before coding.

- [ ] **Step 1: Write failing unit tests** with a fake `embed` passed to `run`:
  - `test_enroll_writes_voiceprint`: two tmp WAVs give a saved `.npy` whose norm ≈ 1, return 0, and `self-score:` in stdout.
  - `test_enroll_rejects_short_clip`: a 2 s WAV returns 2, and no file is written.
  - `test_enroll_prints_test_scores`: `--test` prints one `test ` line per file.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_enroll_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `read_wav`, `load_sherpa_embed` and `enroll_cli`, then register `enroll` in `__main__.py` like `speak`. It needs no API keys.
- [ ] **Step 4: Run** the unit tests. Expected: PASS.
- [ ] **Step 5: Install and download.** Run `uv sync --extra speaker`, then `curl -L -o models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx <URL above>`.
- [ ] **Step 6: Write an integration test** (`@pytest.mark.integration`, skip if the model or WAVs are missing):
  - `test_enroll_scores_second_owner_clip_high`: enroll `data/my_voice1.wav` and score `data/my_voice_2.wav`. The owner's score must be greater than the score of `data/girl-saying-hello_E_major.wav`.
  - It also asserts that one 3 s embed takes under 0.2 s.
- [ ] **Step 7: Measure.** Run `uv run reachy-demo enroll --file data/my_voice1.wav --test data/my_voice_2.wav data/girl-saying-hello_E_major.wav`. Set `DEFAULT_HIGH` and `DEFAULT_LOW` in `speaker_id.py` from the printed scores:
  - `DEFAULT_HIGH`: about midway between the owner and guest scores, never above the owner's `my_voice_2` score minus 0.05.
  - `DEFAULT_LOW`: about 0.15 below `DEFAULT_HIGH`.
  - Record the measured numbers in a comment.
- [ ] **Step 8: Commit** `feat(speaker): enroll the owner's voice with CAM++ (sherpa-onnx)`. Add the measured scores to the message. Do not commit `models/` or `data/`; both are git-ignored.

### Task 3: Owner mode, `owner_mode` tool, gated turns

**Files:**
- Create: `src/reachy_demo/agent/tools/owner.py`
- Test: `tests/unit/test_owner_mode.py`

**Interfaces:**
- Consumes: `SpeakerGate`, `Verdict` (Task 1); `OWNER` and `Speaker` from `memory/store.py`.
- Produces:
  - `GUEST = Speaker(0, "guest")`. It is a placeholder that no memory row uses; guest turns never reach tools in owner mode.
  - `class OwnerMode(gate: SpeakerGate | None)`:
    - `.on: bool = False`
    - `.last: Verdict = "owner"`
    - `.speaker() -> Speaker`: `OWNER` if not `on` or `last == "owner"`, else `GUEST`.
  - `class OwnerTools(mode: OwnerMode)`: a ToolSet with one tool, `owner_mode`, with params `{"on": boolean}` (required).
  - `PROMPT`: tells the LLM to call `owner_mode(on=true)` on "only follow my command" (any wording or language) and `on=false` on "stop following only me / follow everyone".
  - `NOT_OWNER`, `NOT_ENROLLED`, `UNSURE = "I'm not sure that's you, please say it again."`
  - `async def gated_turns(turns: AsyncIterator[str], mode: OwnerMode, out: TextIO, say: Callable[[str], Awaitable[object]] | None = None) -> AsyncIterator[str]`

**`gated_turns` behaviour, per turn:**
1. Get the verdict and score: `mode.gate.take()`, or `("owner", None)` if there is no gate. Set `mode.last`.
2. Print `  (voice: <verdict> <score|->)`.
3. If owner mode is off, or the verdict is `owner`, yield the turn.
4. If the verdict is `guest`, print `  (ignored guest: <text>)`.
5. If the verdict is `unsure`, print `  (unsure: <text>)`, then `await say(UNSURE)` if `say` is set.

Add this comment where the unsure reply is spoken: `# shortcut: may overlap an answer that is still playing; route through converse if it does on stage.`

**`OwnerTools.call` behaviour:**
- gate is `None`: return `NOT_ENROLLED`.
- `mode.last != "owner"`: return `NOT_OWNER`.
- Otherwise set `mode.on` and return `{"ok": True, "owner_mode": on}`.
- Bad args: `{"error": "bad_arguments: on must be true or false"}`.

- [ ] **Step 1: Write failing tests**, using a fake gate whose `take()` pops scripted verdicts:
  - `test_mode_off_yields_all`: guest, unsure and owner verdicts all come through.
  - `test_mode_on_drops_guest`: owner mode on, a guest turn is not yielded, and `ignored guest` is printed.
  - `test_mode_on_unsure_asks_again`: `say` is called once with `UNSURE`, and nothing is yielded.
  - `test_mode_off_still_takes_each_turn`: 3 turns with owner mode off make 3 `take()` calls.
  - `test_speaker_follows_mode`: `speaker()` is `OWNER` when owner mode is off even with `last="guest"`, and `GUEST` when it is on with `last="guest"`.
  - `test_toggle_on_and_off`: an owner verdict, then `call("owner_mode", '{"on": true}')` gives `mode.on` True, then false.
  - `test_guest_cannot_toggle`: `last="unsure"` returns `NOT_OWNER` and `on` is unchanged.
  - `test_not_enrolled`: `OwnerMode(None)` returns `NOT_ENROLLED`.
  - `test_no_gate_counts_as_owner`: typed input yields every turn with owner mode on.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_owner_mode.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `owner.py`.
- [ ] **Step 4: Run** the tests. Expected: PASS. Then run ruff and mypy.
- [ ] **Step 5: Commit** `feat(speaker): owner mode drops guests and asks unsure speakers again`

### Task 4: `dance` and `look_toward_sound` are owner-only in owner mode

**Files:**
- Modify: `src/reachy_demo/agent/tools/head.py` (`HeadTools.__init__`, `call`)
- Modify: `src/reachy_demo/agent/tools/sound.py` (`SoundTools.__init__`, `call`)
- Test: `tests/unit/test_head_tools.py`, `tests/unit/test_sound_tools.py`

**Interfaces:**
- Consumes: `Speaker` and `OWNER`.
- Produces:
  - `HeadTools(motion, dances=None, move_s=MOVE_S, speaker: Callable[[], Speaker] = lambda: OWNER)`
  - `SoundTools(sound, motion, clock=time.monotonic, speaker: Callable[[], Speaker] = lambda: OWNER)`
  - `NOT_OWNER = {"error": "not_owner: only the owner can ask for that right now"}` in each module.

Only `dance` (in head.py) and `look_toward_sound` are gated. `turn_head`, `tilt_head`, `roll_head` and `stop_head` stay open.

- [ ] **Step 1: Write failing tests:**
  - `test_dance_refused_for_guest`: `speaker=lambda: GUEST` returns `NOT_OWNER`, and nothing is queued.
  - `test_stop_head_allowed_for_guest`: a guest can still stop the head.
  - `test_look_toward_sound_refused_for_guest`
  - Existing tests pass unchanged, through the default `speaker`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_head_tools.py tests/unit/test_sound_tools.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** The check goes at the top of the gated branch.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(speaker): dance and look_toward_sound refuse guests in owner mode`

### Task 5: Wire into `reachy-demo agent`, plus docs

**Files:**
- Modify: `src/reachy_demo/agent/cli.py`: `_tap`, `run`, and a new `speaker_gate_from_env()`
- Modify: `docs/usage/agent.md` (new section "Owner mode"), `docs/FEATURES.md` (new row 13, take "owner-only voice" out of "Not built yet", Last updated, branch table), `docs/usage/memory.md` (update the line "Until speaker ID exists, everyone counts as the owner")
- Modify: the `# shortcut: everyone counts as the owner until speaker ID (M9) exists` comment in `cli.py`. Remove it, because `mode.speaker` replaces `lambda: OWNER`.
- Test: `tests/unit/test_agent_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `speaker_gate_from_env() -> SpeakerGate | None`. It reads `SPEAKER_MODEL`, `SPEAKER_HIGH`, `SPEAKER_LOW` and `data/voice/owner.npy`. It prints `  (owner mode: available, high H low L)` or `  (owner mode: off: <reason>)` and returns `None` when it can't load. It never raises.

**Wiring in `run`:**
1. Build `gate = None if args.type else speaker_gate_from_env()`, then `mode = OwnerMode(gate)`.
2. Pass `gate` to `_tap`, which calls `gate.feed(chunk, speaking)` like `speech`.
3. Set `turns = gated_turns(turns, mode, sys.stdout, say=speak)`.
4. Always add `OwnerTools(mode)` and `owner.PROMPT`.
5. Pass `speaker=mode.speaker` to `MemoryTools` (through `memory_from_env`), `VoiceTools`, `HeadTools` and `SoundTools`, replacing `lambda: OWNER`.

- [ ] **Step 1: Write failing tests:**
  - `test_speaker_gate_from_env_missing_voiceprint`: with a tmp cwd and no `owner.npy`, it returns `None` and prints `owner mode: off`.
  - `test_speaker_gate_from_env_bad_threshold`: `SPEAKER_HIGH=abc` returns `None` and prints a config message.
  - `test_tap_feeds_gate`: speech chunks reach `gate.feed`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_agent_cli.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the wiring.
- [ ] **Step 4: Full check.** Run `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -m "not integration"`. Expected: all green.
- [ ] **Step 5: Live smoke test** with the owner, on the same mic used to enroll:
  1. Run `uv run reachy-demo agent --stdin` (or `--mic`).
  2. Say "only follow my command". Expected: `(voice: owner 0.xx)` and the tool reply.
  3. Play `data/girl-saying-hello_E_major.wav` near the mic. Expected: `(ignored guest: …)`.
  4. Say "stop following only me". Expected: normal mode again.

  Record the scores you see in `docs/usage/agent.md` as tuning guidance.
- [ ] **Step 6: Write the docs.**
  - The FEATURES row: **Owner mode (speaker verification)**. Settings: `SPEAKER_MODEL`, `SPEAKER_HIGH`, `SPEAKER_LOW`. Doc: `agent.md § Owner mode`.
  - The usage section covers enrollment, the two phrases, what guests and unsure speakers get, how to tune thresholds from the `(voice: …)` lines, and that the face check is not built yet.
- [ ] **Step 7: Commit** `feat(speaker): owner mode in the voice agent` with code and docs together.
