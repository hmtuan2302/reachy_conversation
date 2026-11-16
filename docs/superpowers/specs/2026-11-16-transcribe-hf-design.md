# Live transcribe (speech -> text in the spoken language) on the Hugging Face server

Replaces the translation goal of `2026-11-16-live-translate-design.md`. Small change.

## Intent
User clarified that they want speech-to-text: Vietnamese speech -> Vietnamese text, English speech -> English text. No translation. They have an HF token, not an OpenAI key, so use the server the Reachy conversation app uses.

## Spike result (2026-11-16)
- `POST https://pollen-robotics-reachy-mini-realtime-url.hf.space/session`, header `X-Reachy-Mini-Authorization: Bearer <HF_TOKEN>`, body `{}` -> `{connect_url, state: "granted", ...}`.
- WS `connect_url` with `Authorization: Bearer <HF_TOKEN>`; OpenAI-compatible protocol.
- `session.update` with input 24 kHz PCM, `transcription {language: "auto"}`, `turn_detection {type: server_vad, create_response: false}` -> only `conversation.item.input_audio_transcription.completed` per utterance, with no LLM reply or audio. About 0.5 s after the speech ends. English verified; Vietnamese not yet tested.

## Approaches
- A. HF server, transcription only (PICKED): token already in `.env`, language auto-detected, one line per utterance.
- B. OpenAI translation endpoint (current code): wrong task (translates) and needs an OpenAI key. Remove it; git history keeps it.
- C. Local ASR (sherpa-onnx/whisper): offline, but needs a model download and CPU, and Vietnamese quality is uncertain. Possible later.

## Design
- Rename the feature to `transcribe`: `voice/transcribe.py`, `voice/transcribe_cli.py`, CLI `reachy-demo transcribe (--mic | --file PATH)`. Delete `voice/translate*.py` and their tests. `audio/sources.py` stays as is.
- `transcribe_stream(pcm24k, *, hf_token, session_url=SESSION_URL) -> AsyncIterator[str]` yields one completed transcript per utterance. It allocates the session (httpx), connects, sends session.update, and pumps audio. If the server sends `...transcription.delta`, those are ignored (shortcut; add live partials later if wanted).
- End of a file: append 2 s of silence so the last utterance closes, then wait up to 5 s for the final transcript and close. Mic runs until Ctrl-C.
- Output: one line per utterance, in the spoken language.
- Env: `HF_TOKEN` (already in `.env`), added to `.env.example`.
- Errors as before: allocator HTTP error or state != granted, handshake failure, socket close or server `error` -> ConnectionError (exit 1); source/WAV/mic errors surface (exit 2); missing HF_TOKEN -> exit 2.
- Tests: fake allocator (httpx MockTransport) + fake WS server. Rename the usage doc to `docs/usage/transcribe.md`.

## Risks
- Audio goes to Pollen's HF endpoint, under their quota or terms.
- Vietnamese recognition is unverified until the user tries it.
