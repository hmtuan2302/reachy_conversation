# Voice agent, stage 2: text → speech (Gemini, streamed, saved)

Small change. Completes the pipeline S2T → LLM → **T2S**.

## Decided by the user
- Model `gemini-3.8-flash-tts`, Gemini API key (in `.env` as `GEMINI_API_KEY`).
- Streaming mode: play the speech while it is being generated.
- When generation ends, save the audio to a file.

## Spike (2026-11-16)
- `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent?alt=sse`, header `x-goog-api-key`, body `contents[].parts[].text` + `generationConfig {responseModalities:["AUDIO"], speechConfig.voiceConfig.voice:"Kore"}`.
- Each SSE event carries `candidates[0].content.parts[].inlineData.data` = base64 raw PCM (`audio/l16; rate=24000; channels=1`).
- Vietnamese: first audio 1.06 s, 5.2 s of speech done in 2.2 s. English: first audio 0.93 s. Faster than real time.
- Round-trip check: the TTS audio run through `reachy-demo transcribe` read back the same text (vi and en).
- Playback: Windows `ffplay.exe` fed raw PCM on stdin works from WSL (exit 0; audible playback still to be confirmed by the user). sounddevice output isn't installed by default.

## Approaches
- **When speech starts.** A: speak the whole answer once the LLM finishes (PICKED: answers are ≤ 3 sentences and DeepSeek finishes in ~0.5–1 s, so the cost is small). B: send each sentence to TTS as soon as the LLM finishes it (faster start on long answers, but more requests and ordering logic). Do B later if long answers feel slow (`shortcut:` comment).
- **Where it plays.** A: pipe PCM into Windows `ffplay.exe` (PICKED: native Windows audio, the same route as the mic, no extra dependency). B: sounddevice through WSLg (needs the `mic` extra and WSLg output; the mic path was unreliable). The player command can be changed with `TTS_PLAYER`.

## Design
- `src/reachy_demo/voice/tts.py`: `speak_stream(text, *, api_key, model="gemini-3.8-flash-tts", voice="Kore", transport=None) -> AsyncIterator[bytes]` yields 24 kHz mono PCM16 chunks. HTTP/network/malformed errors → `TTSError`. Gemini details live only here.
- `src/reachy_demo/audio/player.py`: `play_and_save(chunks, *, path, player_cmd) -> Path`. It starts the player subprocess, writes each chunk to its stdin as it arrives, waits for playback to end, then writes `path` as a WAV (always saved, even if the player fails or is disabled). Default player: `ffplay.exe -autoexit -nodisp -loglevel error -f s16le -ar 24000 -ch_layout mono -i -`. `TTS_PLAYER=none` means save only.
- Saved files: `data/speech/<YYYYmmdd-HHMMSS>-<n>.wav`, one per answer (`data/` is git-ignored). The path is printed dimmed after the answer: `  (saved data/speech/…wav)`.
- `reachy-demo agent`: after `bot: …` finishes printing, it speaks it. The next turn is answered after playback ends. `--no-speak` keeps the text-only behavior.
- `reachy-demo speak "text"`: a new command to test TTS alone (prints the saved path).
- Env: `GEMINI_API_KEY` (required when speaking), `TTS_MODEL` (default `gemini-3.8-flash-tts`), `TTS_VOICE` (default `Kore`), `TTS_PLAYER`. Added to `.env.example`.
- Errors fail small: a TTS error prints `  (speech error: …)` and listening continues. The text answer is already on screen.
- Tests: Gemini SSE parsing via httpx MockTransport (chunks decoded in order, 4xx → TTSError), play_and_save with a fake player command (`cat > file`) and the WAV contents, agent CLI speaking after the answer, speech error keeps listening, and `--no-speak`.
- Docs: `docs/usage/agent.md` (speech section), new `docs/usage/speak.md`, README, `.env.example`.

## Risks
- **Echo:** with laptop speakers the mic hears the bot and may transcribe it as a new turn. With your headset it doesn't. Echo handling or barge-in is a later stage.
- The Gemini key was pasted in chat; rotate it when convenient.
