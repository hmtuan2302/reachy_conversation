# Speak: text → speech (Gemini)

Turn text into speech with Gemini `gemini-3.8-flash-tts`. It plays while it streams in and is saved as a WAV when done. The voice agent ([agent.md](agent.md)) uses the same path to say its answers. Use this command to test speech on its own.

```bash
$ uv run reachy-demo speak "Xin chào, tôi là Reachy."
saved data/speech/20261116-085850-1.wav
```

## Setup

In `.env` at the repo root:

```
GEMINI_API_KEY=...              # from aistudio.google.com
# TTS_VOICE=Kore                # optional voice (e.g. Kore, Puck)
# TTS_MODEL=gemini-3.8-flash-tts
# TTS_PLAYER=none               # save only, don't play
```

Playback uses Windows `ffplay.exe` (from `winget install Gyan.FFmpeg.Essentials`) on the Windows default output, e.g. your headset. Run commands from the repo folder.

## How it works

- Streaming request (`streamGenerateContent?alt=sse`). Gemini sends raw 24 kHz mono 16-bit audio in small chunks, and each chunk is passed to the player as it arrives: about 1 s to the first sound, and generation runs faster than real time.
- When generation ends, everything received is written to `data/speech/<YYYYmmdd-HHMMSS>-<n>.wav` (24 kHz mono 16-bit). It's saved even if playback failed, and a partial file is saved if the stream broke.
- Vietnamese and English both work. Check with `uv run reachy-demo transcribe --file data/speech/<file>.wav`, which reads the saved audio back as text.
- Play a saved file again: `ffplay.exe -autoexit -nodisp "$(wslpath -w data/speech/<file>.wav)"`.

## Errors

| Message | Fix |
|---|---|
| `config error: GEMINI_API_KEY is required` (exit 2) | Add the key to `.env` |
| `speech error: Gemini TTS returned 400/403` (exit 1) | The key is wrong or not allowed for this model |
| `speech error: Gemini TTS returned 429` (exit 1) | Quota or rate limit. Wait and retry |
| `speech error: cannot start player 'ffplay.exe' ...` (exit 1) | Install Windows ffmpeg, or set `TTS_PLAYER=none`. The WAV is still saved |

Code: [src/reachy_demo/voice/tts.py](../../src/reachy_demo/voice/tts.py), [src/reachy_demo/audio/player.py](../../src/reachy_demo/audio/player.py), [src/reachy_demo/voice/speak_cli.py](../../src/reachy_demo/voice/speak_cli.py).
