# Live translate (laptop mic or WAV file)

Speak English or Vietnamese, or play a WAV file, and see the translation in the terminal as you speak.

## Setup (once)

```bash
cd ~/reachy_conversation
unset VIRTUAL_ENV            # if conda is active
uv sync --extra mic          # --extra mic is only needed for --mic
cp .env.example .env         # then put your key in .env: OPENAI_API_KEY=sk-...
```

`translate` only needs `OPENAI_API_KEY`. `ROBOT_HOST` is not required.

**WSL2 mic:** WSL2 reaches the Windows mic through WSLg's PulseAudio. If `--mic` says it can't open the microphone, check that `pactl info` works and that Windows lets apps use the microphone (Settings → Privacy → Microphone).

## Use

```bash
# Speak Vietnamese, read English (default --to en)
uv run reachy-demo translate --mic

# Speak English, read Vietnamese
uv run reachy-demo translate --mic --to vi

# Translate a recording (streamed at real-time speed, like a live mic)
uv run reachy-demo translate --file meeting.wav --to en
```

Stop the mic with **Ctrl-C**. File mode exits by itself when the file ends.

## Options

| Option | Meaning |
|---|---|
| `--mic` | Laptop default microphone |
| `--file PATH` | WAV file: 16-bit PCM, any sample rate, mono or stereo |
| `--to en\|vi` | Language you want to **read** (default `en`) |

Use either `--mic` or `--file`, not both.

## Converting other audio formats

Only 16-bit PCM WAV is accepted. To convert mp3, m4a or other formats, use ffmpeg:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 24000 -sample_fmt s16 out.wav
```

## Exit codes and common errors

| Code | Message | Fix |
|---|---|---|
| 0 | | Done, or you pressed Ctrl-C |
| 2 | `OPENAI_API_KEY is required` | Add the key to `.env` |
| 2 | `need 16-bit PCM WAV` / `not a readable WAV file` | Convert with ffmpeg (above) |
| 2 | `install the mic extra` | `uv sync --extra mic` |
| 2 | `cannot open microphone` | Check the mic / WSL audio (see Setup) |
| 1 | `translation connection failed: ... 401` | The key is wrong or expired |
| 1 | `translation socket closed. <reason>` | Server-side problem. Read the reason and retry |

## How it works

The audio is converted to 24 kHz mono PCM16 and sent in 200 ms chunks to OpenAI's Realtime translation endpoint (`gpt-realtime-translate`). Translated text fragments are printed as they arrive. Audio leaves your machine and is billed to your OpenAI key.

## Limits

- You choose the direction with `--to`. The source language isn't auto-detected, because the endpoint translates into one fixed language per session.
- Output is a continuous stream of text. There are no line breaks between sentences.
- Code: [src/voice/translate.py](../../src/voice/translate.py), [src/voice/translate_cli.py](../../src/voice/translate_cli.py), [src/audio/sources.py](../../src/audio/sources.py).
