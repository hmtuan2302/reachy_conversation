# Live transcribe (speech → text)

Speak into the laptop mic, or play a WAV file, and get a line of text for each sentence, in the language you spoke. Vietnamese is written as Vietnamese and English as English. The language is detected automatically for each sentence.

```
$ uv run reachy-demo transcribe --file data/OSR_us_000_0010_8k.wav
The birch canoe slid on the smooth planks.
Glue the sheet to the dark blue background.
...
```

By default it uses Pollen's Hugging Face server. Set `STT_PROVIDER=local` to run it on the laptop instead (see [Local provider](#local-provider-stt_providerlocal)).

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

**WSL2 `--mic` (one-time; not needed for `--stdin`):** WSL2 reaches the Windows mic through WSLg's PulseAudio server (`/mnt/wslg/PulseServer`). PortAudio only uses ALSA, so install the ALSA→Pulse bridge and make it the default. Without it, `--mic` fails with `cannot open microphone: Error querying device -1`.

```bash
sudo apt install -y libasound2-plugins
printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > ~/.asoundrc
uv run python -c "import sounddevice as sd; print(sd.query_devices())"   # must list "default"
```

**Which mic is used (e.g. Soundcore Bluetooth headphones):** WSL can't see devices by name. It always records the *Windows default input*, which appears in WSL as `RDPSource`. To use your headset:

1. Windows: Settings → System → Sound → Input → choose the headset mic (often "Headset" or "Hands-Free"), and check that the level bar moves when you speak.
2. Check in WSL and record a 5 s test:
   ```bash
   sudo apt install -y pulseaudio-utils
   pactl info | grep "Default Source"                # RDPSource
   parecord --channels=1 --rate=24000 /tmp/t.wav & sleep 5; kill $!; paplay /tmp/t.wav
   ```
3. Restart `transcribe` after you change the Windows default.

Bluetooth headsets switch to the lower-quality hands-free mode while the mic is in use. That's fine for speech, but a laptop or USB mic is clearer.

If it still fails, allow microphone access for apps in Windows (Settings → Privacy & security → Microphone), then run `wsl --shutdown` from Windows and reopen the terminal.

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
| `--mic` | Default mic as seen by WSL (through WSLg) |
| `--stdin` | Raw 24 kHz mono 16-bit PCM piped in. Best quality on WSL; see below |
| `--file PATH` | WAV file: 16-bit PCM, any sample rate, mono or stereo |

Use exactly one of `--mic`, `--stdin`, `--file`.

## Best quality on WSL: record in Windows, pipe into WSL

`--mic` goes through WSLg's PulseAudio bridge. To skip it, let Windows ffmpeg read the mic directly (DirectShow) and pipe raw PCM into the app. You don't need the ALSA, Pulse or `.asoundrc` setup for this.

1. Install ffmpeg on **Windows** once: `winget install Gyan.FFmpeg.Essentials`. Then open a new WSL terminal, and `which ffmpeg.exe` should find it.
2. Find your mic's exact name:
   ```bash
   ffmpeg.exe -hide_banner -list_devices true -f dshow -i dummy 2>&1 | grep '(audio)'
   # "Headset (soundcore R60i NC)" (audio)
   ```
3. Transcribe:
   ```bash
   ffmpeg.exe -hide_banner -loglevel error -f dshow -audio_buffer_size 50 \
     -i audio="Headset (soundcore R60i NC)" -ac 1 -ar 24000 -f s16le - \
     | uv run reachy-demo transcribe --stdin
   ```
   Press Ctrl-C to stop. `-audio_buffer_size 50` keeps the delay low (50 ms instead of DirectShow's default 500 ms).

**Test a recording end to end** (record 5 s, listen, transcribe):
```bash
ffmpeg.exe -y -hide_banner -loglevel error -f dshow -i audio="Headset (soundcore R60i NC)" \
  -t 5 -ac 1 -ar 24000 "$(wslpath -w data)\\test.wav"     # speak now
ffplay.exe -autoexit -nodisp "$(wslpath -w data/test.wav)"  # hear it in your headset
uv run reachy-demo transcribe --file data/test.wav          # should print what you said
```
If you hear nothing in step 2, the problem is in Windows (see below), not in this app. To check the mic with no WSL involved at all, record with the Windows **Sound Recorder** app.

**Check that the mic isn't silent** (speak during the 3 s):
```bash
ffmpeg.exe -hide_banner -loglevel error -f dshow -i audio="Headset (soundcore R60i NC)" -t 3 -ac 1 -ar 24000 -f s16le - \
  | uv run python -c "import sys,numpy as n; x=n.frombuffer(sys.stdin.buffer.read(),'<i2').astype(float); print('rms',round(float(n.sqrt((x**2).mean())))) "
```
Speech is usually rms > 300. **rms near 0 means Windows is giving apps silence.** Fix it in Windows:
- Settings → Privacy & security → Microphone: turn on **Microphone access**, **Let apps access your microphone** and **Let desktop apps access your microphone**. Blocked apps get silence, not an error.
- Settings → System → Sound → your headset → Input: check that it isn't muted, the volume is up, and the test bar moves when you speak.
- Bluetooth: the headset's mic only works in its "Headset"/hands-free mode. Use the "Headset (...)" device, not "Headphones (...)".

Or pipe any other source the same way, e.g. a file with Linux ffmpeg: `ffmpeg -re -i talk.mp3 -ac 1 -ar 24000 -f s16le - | uv run reachy-demo transcribe --stdin`.

## Converting other audio formats

Only 16-bit PCM WAV is accepted. To convert mp3, m4a or other formats, use ffmpeg:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 24000 -sample_fmt s16 out.wav
```

**Mic diagnostic for `--mic`:** `uv run python scripts/mic_check.py` shows the live level for 8 s, saves `data/mic_check.wav`, and transcribes it.

## Debug recording (saved on exit)

When you use `--mic` or `--stdin`, everything the app sent for transcription is saved when it exits: on Ctrl-C, on an error, or at the end of the input:

```
recording saved: data/recordings/20261116-091347-mic.wav
```

Use it to check what the app actually heard:
```bash
ffplay.exe -autoexit -nodisp "$(wslpath -w data/recordings/<file>.wav)"   # listen
uv run reachy-demo transcribe --file data/recordings/<file>.wav            # transcribe it again
```
If it plays back as silence, the mic route isn't delivering sound (see the mic setup above). `--file` input isn't copied, because it's already on disk. `data/` is git-ignored. Delete old recordings yourself.

## Exit codes and common errors

| Code | Message | Fix |
|---|---|---|
| 0 | | Done, or you pressed Ctrl-C |
| 2 | `HF_TOKEN is required` | Add `HF_TOKEN=hf_...` to `.env` |
| 2 | `no such file` | Check the path |
| 2 | `need 16-bit PCM WAV` / `not a readable WAV file` | Convert with ffmpeg (above) |
| 2 | `install the mic extra` | `uv sync --extra mic` |
| 2 | `cannot open microphone: Error querying device -1` | No input device visible. On WSL2, do the mic setup above |
| 2 | `cannot open microphone: ... Wait timed out` | The mic didn't start, even after one automatic retry. Run it again. If it keeps failing, run `wsl --shutdown` from Windows |
| 1 | `session allocator returned 401 ... token_invalid` | The token is wrong or expired. Make a new one on huggingface.co → Settings → Access Tokens |
| 1 | `transcription connection failed` / `socket closed` | Network or server problem. Retry |

## Local provider (`STT_PROVIDER=local`)

Runs on the laptop, with no network after the first model download and no `HF_TOKEN`. Vietnamese or English, one language per sentence (mixed sentences are not supported).

```bash
uv sync --extra stt              # plus the extras you already use, e.g. --extra mic --extra vieneu-gpu
```

In `.env`:

```
STT_PROVIDER=local
STT_MODEL=large-v3-turbo         # faster-whisper model name or a local folder
STT_DEVICE=                      # cpu | gpu; empty = GPU if found, else CPU
```

The first run downloads Whisper `large-v3-turbo` (about 1.6 GB, to the Hugging Face cache) and `smart-turn-v3.2-cpu.onnx` (to `models/smart-turn/`). Loading takes a few seconds at startup.

**How a turn ends:**

1. Silero VAD checks every 32 ms whether you are speaking (speech starts at `VAD_THRESHOLD`, default 0.5).
2. After `STT_PAUSE_MS` of silence, Smart Turn v3.2 listens to the last 8 s of your voice (intonation, rhythm, trailing "ờ…", "and…") and scores 0 to 1 how likely you are finished.
3. A score of at least `STT_TURN_THRESHOLD` ends the turn: Whisper transcribes it, choosing Vietnamese or English by itself.
4. Below that, it keeps listening. Speaking again continues the same turn; `STT_MAX_SILENCE_MS` of silence ends it anyway.

Sounds shorter than 250 ms of speech (a cough, a click) are ignored.

**Tuning:**

| Setting | Default | Raise it when | Lower it when |
|---|---|---|---|
| `STT_PAUSE_MS` | 200 | Tiny gaps between words trigger checks | You want replies a little sooner |
| `STT_TURN_THRESHOLD` | 0.5 | Reachy cuts you off while you think | Replies wait too long after a finished sentence |
| `STT_MAX_SILENCE_MS` | 2000 | You need longer thinking pauses | "Unfinished-sounding" sentences wait too long |

After you stop speaking: about 0.2 s plus Whisper when Smart Turn says done, about 2 s plus Whisper when it doesn't.

**Measured** (RTX 3060 Laptop, WSL2): Whisper about 0.55 s per sentence on GPU (`int8_float16`), Smart Turn about 130 ms on CPU.

**GPU libraries:** Whisper on GPU needs cuBLAS 12 and cuDNN 9. The app loads them from the pip `nvidia-*` packages that the `vieneu-gpu` extra installs with PyTorch. Without them, startup fails with `error: Whisper failed to load on cuda: ... Try STT_DEVICE=cpu`. Set `STT_DEVICE=cpu` (slower, about 1–3 s per sentence), or install the `vieneu-gpu` extra.

## How it works

With `STT_PROVIDER=hf` (the default):

1. The command asks Pollen's Reachy session service (`pollen-robotics-reachy-mini-realtime-url.hf.space`) for a server, using your HF token. This is the same backend that `reachy_mini_conversation_app` uses.
2. Audio is converted to 24 kHz mono PCM16 and streamed in 200 ms chunks over the realtime WebSocket. The app adds 0.5 s of silence before your audio and 2 s after it. The server's speech detector misses speech that starts at the very first sample (a short "hello" clip returned nothing without the lead-in), and it needs the trailing silence to know the last sentence has ended.
3. The server detects each sentence, transcribes it with language `auto`, and returns the text. It's set to transcribe only (`create_response: false`), so the AI doesn't reply.

Your audio leaves your machine and goes to Pollen's Hugging Face endpoint.

## Limits

- No live partial text while you speak. Each line appears when the sentence ends.
- English is verified. Vietnamese should work, since the speech recognizer is multilingual, but hasn't been checked yet. Please try it with `--mic`.
- Local provider: the turn detector has no published Vietnamese accuracy figures. Short one- or two-word clips are the least reliable.
- Code: [src/reachy_demo/voice/local_stt.py](../../src/reachy_demo/voice/local_stt.py), [src/reachy_demo/voice/turns.py](../../src/reachy_demo/voice/turns.py), [src/reachy_demo/voice/transcribe.py](../../src/reachy_demo/voice/transcribe.py), [src/reachy_demo/voice/transcribe_cli.py](../../src/reachy_demo/voice/transcribe_cli.py), [src/reachy_demo/audio/sources.py](../../src/reachy_demo/audio/sources.py).
