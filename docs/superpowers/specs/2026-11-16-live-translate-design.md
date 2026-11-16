# Live transcribe + translate (laptop mic / file)

## Intent
Speak into the laptop mic, or play an audio file, and see a live translation in the terminal. Standalone dev/demo tool; unrelated to the robot mic or the conversation loop.

## Goals
- Input: laptop mic or a WAV file (`--mic` / `--file x.wav`).
- Vietnamese <-> English: the output is whichever of the two languages the speaker is not using (vi speech -> en text, en speech -> vi text).
- Partial text updates in place, then a final line per utterance.

## Non-goals
Robot mic, speaking the translation, other languages, offline/local models (first version), saving transcripts, GUI.

## Anti-goal
Touching or depending on the robot/conversation code, so it can't break the demo.

## Constraints
- Audio goes to OpenAI (user OK'd for now). Key from existing `.env` config.
- Reuse deps: `websockets`, `numpy`. New dep: `sounddevice` (optional extra `mic`).
- Python 3.12, ruff/mypy strict as in `pyproject.toml`.

## Approaches
- A. OpenAI Realtime WebSocket (PICKED): least code, best latency/quality; needs key + network.
- B. Local sherpa-onnx ASR + local MT: private/offline, but two models, weaker translation, more glue. Possible later.
- C. VAD + chunked API calls: simple, but not truly realtime.

## Design
- `src/voice/translate.py`: async `translate_stream(pcm_chunks) -> AsyncIterator[Event]`; opens a Realtime WS session instructed "translate vi<->en, output translation only"; sends 24 kHz PCM16; yields `Partial(text)` / `Final(text)`.
- Sources (same module, small): mic via `sounddevice` (resample to 24 kHz); WAV via stdlib `wave`, resample, paced at realtime.
- CLI: `reachy-demo translate (--mic | --file PATH)` in `__main__.py`; terminal printer uses `\r` for partials.
- Server turn detection ON for this tool (no manual commit needed), unlike the main conversation session.
- Test: fake WS server + short WAV fixture asserts Partial/Final sequence.

## Left to the builder
Exact Realtime event names and the model's translation mode (verify against current docs at build time), resampling method, chunk size.

## Calls I made (not yours)
Auto-detect source language; terminal-only output; file sent at realtime pace; optional `mic` extra.
