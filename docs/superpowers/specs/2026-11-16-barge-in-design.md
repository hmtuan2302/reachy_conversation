# Voice agent, stage 3: barge-in with a spoken stop command

Small change to `reachy-demo agent`. The user can stop Reachy mid-answer by saying a stop command. Noise and other speech don't stop it.

## Decided by the user
- The mic keeps streaming while Reachy thinks (LLM) or speaks (TTS + playback).
- A local VAD ducks *upcoming* playback to 25% volume while the user talks.
- Only a stop command cancels the answer. Noise, random speech or a new question do not.
- On a stop: cancel the active answer (LLM stream, TTS, playback), mark its audio stale, flush queued speaker audio, and go back to listening. Late audio from the old answer is ignored. A short tail that already reached the speaker is acceptable.
- **Behaviour A** (picked): "stop" alone cancels and listens. "Stop, what's the weather in Hanoi?" cancels and answers the rest as the next turn.
- **Non-stop speech while Reachy is busy is dropped**, not queued (changed from today's queueing). The volume comes back and the answer continues.
- **Intent by LLM** (user's proposal, picked over a keyword list): the server VAD already ends each user turn (`transcription.completed`). Each turn heard while Reachy is busy goes to DeepSeek, which classifies it.

## Approaches
- **Local VAD.** A: numpy energy gate (PICKED: no dependency, no latency; a false trigger only costs a needless duck, because ducking never cancels). B: Silero VAD ONNX (better speech/noise separation, but needs a model file and more code). Upgrade to B if ducking fires too often on noise (`shortcut:` comment).
- **Stop detection.** A: keyword list at the start of the transcript (instant, but misses phrasings and can misfire on "stop signs…"). B: DeepSeek classifier returning `{stop, rest}` (PICKED by the user: any phrasing, any language, splits the command from the follow-up; costs ~0.3–0.5 s while playback is already ducked).
- **Cancelling.** One sensible way: each answer runs as one asyncio task. Cancelling it closes the LLM and TTS streams and kills the player. A generation counter makes any chunk from an older answer a no-op.

## Design
- **Mic tap** (`audio/vad.py`, new): `EnergyVAD(threshold_rms, onset_ms=150, hangover_ms=VAD_HANGOVER_MS)` with `feed(chunk) -> bool` (speaking or not). The agent wraps the source with an async tee that feeds each chunk to the VAD before it goes to `transcribe_stream`. The tee follows the same pattern as `Recording.tee`. Threshold from env `VAD_RMS` (default 500; `mic_check.py` shows speech is usually rms > 300). `VAD_THRESHOLD` (0–1) is kept for the later Silero model.
- **Ducking** (`audio/player.py`): `play_and_save` takes an optional `gain: Callable[[], float]` and scales each chunk by it just before writing to the player. Playback is **paced**: chunks are written in real time with about 200 ms lead, not as fast as Gemini sends them. Without pacing, the whole answer sits in ffplay's buffer and ducking would change nothing. The gain is 0.25 while the VAD says speaking, otherwise 1.0. The WAV is saved at full volume.
- **Stop** (`play_and_save`): if its task is cancelled, it kills the player at once, saves what it received, and re-raises `CancelledError`.
- **Intent** (`agent/intent.py`, new): `classify(text, chat) -> Intent(stop: bool, rest: str)`. One non-streamed DeepSeek call with a fixed prompt that asks for JSON `{"stop": bool, "rest": str}`, using the existing `chat_stream` and joining the pieces. Timeout 2 s. Any error, timeout or bad JSON gives `Intent(stop=False, rest="")` (fail small: never cancel by mistake; the turn is then dropped).
- **Turn loop** (`agent/cli.py` `converse`): while an answer task is running, each new turn is classified:
  - `stop`: cancel the answer task, bump the generation, print `  (stopped)`. If `rest` is non-empty, it is the next turn.
  - not `stop`: drop it and print `  (ignored: <text>)` so the user sees it was heard. The answer continues.

  When Reachy is idle, turns go straight to the agent with no classify call.
- **History:** a cancelled answer is not added to the agent history (already true: `Agent.answer` saves only on completion). The stop turn itself isn't added either.
- **Env:** `VAD_RMS` (new) and `VAD_HANGOVER_MS` (already in config). Added to `.env.example`.
- **Tests:** EnergyVAD onset/hangover on synthetic PCM; `play_and_save` gain scaling, pacing and kill on cancel (fake player `cat > file`); `classify` with MockTransport (stop, stop+rest, not stop, bad JSON, timeout); `converse` with fake turns: a stop mid-answer cancels it and answers `rest`, non-stop speech mid-answer is dropped (not answered later), idle turns skip classify, late chunks from a cancelled answer are dropped.
- **Docs:** `docs/usage/agent.md` gets an "Interrupting" section; README; `.env.example`.

## Left to the builder
- The exact classifier prompt wording (it must handle Vietnamese and English and return JSON only).
- The pacing loop details (monotonic clock, 20–100 ms chunks).

## Risks
- **Stop latency** = end of your sentence (server VAD, ~0.5–1.2 s) + classify (~0.3–0.5 s). Reachy is ducked to 25% the whole time, so it isn't talking over you at full volume.
- **Echo without a headset:** Reachy's own voice can trigger ducking, and its transcript reaches the classifier. That costs a classify call but should not give `stop`. With a headset, none of this happens.
- **A real question asked mid-answer without a stop word is lost.** The user must say "stop, …" or wait for the answer to end. A failed classify call also drops the turn.
- **ffplay tail:** about 200 ms of lead plus ffplay's own buffer can still be heard after a stop.
- Existing divergence, not changed here: `src/CLAUDE.md` rule 7 says server turn detection is off, but `transcribe.py` already uses `server_vad` for turn ends. This design keeps the current code.
