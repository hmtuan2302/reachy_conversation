# Local speech-to-text with smart turn detection

Date: 2026-11-19. Status: approved.

## Intent

Run voice activity detection and speech-to-text on the laptop instead of Pollen's HF realtime server, and end a turn only when the user has finished speaking, not at every pause. Vietnamese and English, one language per sentence. The provider is chosen in `.env`.

## Goals

- `STT_PROVIDER=hf | local` in `.env`. Empty or `hf` keeps today's behaviour.
- `local`: Silero VAD, then Smart Turn v3, then faster-whisper `large-v3-turbo`. No network after the models are downloaded.
- The language is auto-detected for each sentence, Vietnamese or English.
- Works for `reachy-demo transcribe` and `reachy-demo agent`, with every input source (`--mic`, `--stdin`, `--file`).

## Non-goals

- Mixing languages inside one sentence.
- Streaming partial transcripts while the user is still speaking.
- Replacing `EnergyVAD`, which drives barge-in and ducking.
- Moonshine or other STT engines.

## Anti-goals

- Cutting the user off during a thinking pause. That is the point of Smart Turn.
- Breaking the HF provider or its tests.

## Constraints

- RTX 3060 Laptop, 6 GB of GPU memory, shared with VieNeu TTS on GPU. Whisper runs as `int8_float16` (about 1.5 GB).
- The audio sources give 24 kHz 16-bit PCM. Silero, Smart Turn and Whisper all want 16 kHz.
- Heavy dependencies go in an optional extra (`uv sync --extra stt`), like `vieneu` and `speaker`.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **A (picked)** | faster-whisper `large-v3-turbo` on GPU | One model, built-in language detection, best Vietnamese. Uses about 1.5 GB of GPU memory. |
| B | Moonshine-tiny-vi + Moonshine-base-en + a language check | All on CPU, accuracy close to A. A wrong language check gives a wrong sentence, most often on short words. Three models. |
| C | Whisper `small` on CPU | No GPU, but 1–2 s per sentence and weaker Vietnamese. |

The user picked A.

## Design

New module `src/reachy_demo/voice/local_stt.py` with the same interface as the HF one:

```python
async def transcribe_local(pcm24k: AsyncIterator[bytes]) -> AsyncIterator[str]
```

The flow:

1. Resample 24 kHz to 16 kHz (exact 3:2 ratio).
2. Silero VAD on 512-sample frames (32 ms). It reuses the ONNX model and wrapper that faster-whisper already ships, so there is no extra download.
3. Speech starts at Silero probability ≥ `VAD_THRESHOLD`. The audio is buffered, with 300 ms of lead-in kept.
4. On silence ≥ `STT_PAUSE_MS` (default 200), run Smart Turn v3.2 (ONNX, CPU) on the last 8 s of the turn. It uses Whisper features from faster-whisper's `FeatureExtractor`.
   - If it judges the turn complete (≥ `STT_TURN_THRESHOLD`, default 0.5), transcribe now.
   - If not, keep listening. After `STT_MAX_SILENCE_MS` (default 2000) of silence, end the turn anyway.
5. faster-whisper transcribes with `language=None` (auto-detect), `beam_size=1` and `vad_filter=False`. The language is limited to `vi` or `en`: if detection picks something else, the more likely of the two is used.
6. Yield the text. Whisper runs in a worker thread, so the event loop is never blocked.

Wiring:

- `transcribe_stream` callers (`agent/cli.py`, `transcribe_cli.py`) go through one helper that picks the provider from `STT_PROVIDER`.
- `HF_TOKEN` is required only when the provider is `hf`.
- Models load once at startup and warm up on 1 s of silence, like VieNeu.

New `.env` settings: `STT_PROVIDER`, `STT_MODEL` (default `large-v3-turbo`), `STT_DEVICE` (`cpu | gpu`, empty means GPU if found), `STT_PAUSE_MS`, `STT_TURN_THRESHOLD`, `STT_MAX_SILENCE_MS`.

The Smart Turn v3.2 ONNX file is downloaded from Hugging Face (`pipecat-ai/smart-turn-v3`) into `models/` on first use, like `OBJECT_MODEL`.

## Tests

- Unit: the turn state machine with fake VAD, Smart Turn and Whisper. It must end a turn on "complete", keep listening on "incomplete", and end anyway at the max-silence timeout.
- Unit: provider selection from `STT_PROVIDER`.
- Integration (skipped when models are missing): a Vietnamese WAV and an English WAV each give text in the right language.

## Left to the builder

- Exact names of the shipped Silero files and how faster-whisper exposes them.
- How to limit the language to vi/en (`detect_language` probabilities, or the info returned by `transcribe`).
- Updating the docs: `docs/FEATURES.md` row 1 and `docs/usage/transcribe.md`.
