# Speak: text → speech (Gemini or local VieNeu)

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
# SAVE_SPEECH=off              # play only, don't save a WAV
```

Playback uses Windows `ffplay.exe` (from `winget install Gyan.FFmpeg.Essentials`) on the Windows default output, e.g. your headset. Run commands from the repo folder.

## Local model: VieNeu-TTS v3 Turbo

Runs [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) on this machine. It needs no API key and is Vietnamese-first (Apache-2.0). It runs on CPU through ONNX Runtime at about 0.5× real time on a recent i5, and uses CUDA automatically if it finds it.

```bash
uv sync --extra vieneu          # installs the `vieneu` package; the model downloads on the first run
```

In `.env`:

```
TTS_PROVIDER=vieneu             # default: gemini
# TTS_VOICE=Mai Anh             # a preset voice name; empty = the model's default
```

The first run downloads about 0.5 GB. The ONNX files go to `models/vieneu/` as real files, because `onnxruntime` 1.24+ won't load weights through the Hugging Face cache's symlinks. Measured on this laptop (CPU): loading takes about 6–12 s once per process. Generation speed varies from run to run: on 2026-11-17 it ran 1.1–1.6× *slower* than real time (7 s of speech took about 9.6 s).

With `vieneu`, `GEMINI_API_KEY` is not needed and `TTS_MODEL` is ignored. Both `speak` and `agent` use it. The model loads once per process (it takes a few seconds on the first sentence). Its 48 kHz output is converted to the same 24 kHz WAV and player format as Gemini. A background thread keeps generating while earlier audio plays. Because the model can be slower than real time, playback waits until enough audio is buffered to finish without gaps. The wait is based on the measured speed and the answer's length (about 18 characters per second of speech). First sound comes after about 1–1.5 s for a short answer and 2.5–4 s for a 3-sentence one. Before this, each chunk played as soon as it arrived and speech came out choppy. The wait is in `CHARS_PER_S` and `MARGIN_S` in [vieneu_tts.py](../../src/reachy_demo/voice/vieneu_tts.py).

Test cloning with a fixed WAV: `CLONE_AUDIO=data/OSR_us_000_0010_8k.wav uv run reachy-demo speak "Xin chào"` speaks in that file's voice. A missing file gives `config error: CLONE_AUDIO file not found` (exit 2). For cloning your voice by asking the agent, see [agent.md](agent.md#voice-cloning).

To list the voices: `uv run python -c "from vieneu import Vieneu; print(Vieneu().list_preset_voices())"`.

## How it works

- Streaming request (`streamGenerateContent?alt=sse`). Gemini sends raw 24 kHz mono 16-bit audio in small chunks, and each chunk is passed to the player as it arrives: about 1 s to the first sound, and generation runs faster than real time.
- When generation ends (unless `SAVE_SPEECH=off`), everything received is written to `data/speech/<YYYYmmdd-HHMMSS>-<n>.wav` (24 kHz mono 16-bit). It's saved even if playback failed, and a partial file is saved if the stream broke.
- Vietnamese and English both work. Check with `uv run reachy-demo transcribe --file data/speech/<file>.wav`, which reads the saved audio back as text.
- Play a saved file again: `ffplay.exe -autoexit -nodisp "$(wslpath -w data/speech/<file>.wav)"`.

## Errors

| Message | Fix |
|---|---|
| `config error: GEMINI_API_KEY is required` (exit 2) | Add the key to `.env` |
| `speech error: Gemini TTS returned 400/403` (exit 1) | The key is wrong or not allowed for this model |
| `speech error: Gemini TTS returned 429` (exit 1) | Quota or rate limit. Wait and retry |
| `speech error: TTS_PROVIDER=vieneu needs the package` (exit 1) | `uv sync --extra vieneu` |
| `speech error: VieNeu TTS failed: ...` (exit 1) | The model failed to load or run. Check the voice name and the network on the first download |
| `speech error: cannot start player 'ffplay.exe' ...` (exit 1) | Install Windows ffmpeg, or set `TTS_PLAYER=none`. The WAV is still saved |

Code: [src/reachy_demo/voice/tts.py](../../src/reachy_demo/voice/tts.py), [src/reachy_demo/voice/vieneu_tts.py](../../src/reachy_demo/voice/vieneu_tts.py), [src/reachy_demo/audio/player.py](../../src/reachy_demo/audio/player.py), [src/reachy_demo/voice/speak_cli.py](../../src/reachy_demo/voice/speak_cli.py).
