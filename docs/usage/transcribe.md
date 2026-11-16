# Live transcribe (speech → text)

Speak into the laptop mic, or play a WAV file, and get a line of text for each sentence, in the language you spoke. Vietnamese is written as Vietnamese and English as English. The language is detected automatically for each sentence.

```
$ uv run reachy-demo transcribe --file data/OSR_us_000_0010_8k.wav
The birch canoe slid on the smooth planks.
Glue the sheet to the dark blue background.
...
```

## Setup (once)

```bash
cd ~/reachy_conversation
unset VIRTUAL_ENV            # if conda is active
uv sync --extra mic          # --extra mic is only needed for --mic
```

Put your Hugging Face token in `.env`, in the repo root:

```
HF_TOKEN=hf_...
```

If you've run `huggingface-cli login`, the token is in `~/.cache/huggingface/token`. Only `HF_TOKEN` is needed. `ROBOT_HOST` and `OPENAI_API_KEY` are not.

**Run commands from the repo folder.** `.env` is read from the folder you run the command in.

**WSL2 mic:** WSL2 reaches the Windows mic through WSLg's PulseAudio. If `--mic` says `cannot open microphone`, check that `pactl info` works and that Windows lets apps use the microphone (Settings → Privacy → Microphone).

## Use

```bash
uv run reachy-demo transcribe --mic                 # talk; Ctrl-C to stop
uv run reachy-demo transcribe --file meeting.wav    # stops by itself at the end
uv run reachy-demo transcribe --file meeting.wav > meeting.txt   # save to a file
```

- A line appears about 0.5 s after you stop speaking. You get one line per sentence or pause, and nothing while you're still talking.
- File mode streams at real-time speed, so a 30 s file takes about 35–40 s.

## Options

| Option | Meaning |
|---|---|
| `--mic` | Laptop default microphone |
| `--file PATH` | WAV file: 16-bit PCM, any sample rate, mono or stereo |

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
| 2 | `HF_TOKEN is required` | Add `HF_TOKEN=hf_...` to `.env` |
| 2 | `no such file` | Check the path |
| 2 | `need 16-bit PCM WAV` / `not a readable WAV file` | Convert with ffmpeg (above) |
| 2 | `install the mic extra` | `uv sync --extra mic` |
| 2 | `cannot open microphone` | Check the mic / WSL audio (see Setup) |
| 1 | `session allocator returned 401 ... token_invalid` | The token is wrong or expired. Make a new one on huggingface.co → Settings → Access Tokens |
| 1 | `transcription connection failed` / `socket closed` | Network or server problem. Retry |

## How it works

1. The command asks Pollen's Reachy session service (`pollen-robotics-reachy-mini-realtime-url.hf.space`) for a server, using your HF token. This is the same backend that `reachy_mini_conversation_app` uses.
2. Audio is converted to 24 kHz mono PCM16 and streamed in 200 ms chunks over the realtime WebSocket.
3. The server detects each sentence, transcribes it with language `auto`, and returns the text. It's set to transcribe only (`create_response: false`), so the AI doesn't reply.

Your audio leaves your machine and goes to Pollen's Hugging Face endpoint.

## Limits

- No live partial text while you speak. Each line appears when the sentence ends.
- English is verified. Vietnamese should work, since the speech recognizer is multilingual, but hasn't been checked yet. Please try it with `--mic`.
- Code: [src/voice/transcribe.py](../../src/voice/transcribe.py), [src/voice/transcribe_cli.py](../../src/voice/transcribe_cli.py), [src/audio/sources.py](../../src/audio/sources.py).
