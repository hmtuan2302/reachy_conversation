# Voice agent: ask by voice, hear the answer

Speak a question. The agent prints the answer in the language you spoke, says it aloud, and saves the spoken answer as a WAV file. Pipeline: **speech → text (HF server) → LLM (DeepSeek) → speech (Gemini)**.

```
$ uv run reachy-demo agent --file data/girl-saying-hello_E_major.wav
you: Hello.
bot: Hello! How can I help you today?
  (saved data/speech/20261116-085901-1.wav)      ← spoken through your headset, then saved
```

Add `--no-speak` for text answers only.

It remembers the conversation (the last 10 questions and answers), so follow-ups work:

```
you: Thủ đô của Việt Nam là gì?
bot: Thủ đô của Việt Nam là Hà Nội.
you: Dân số ở đó khoảng bao nhiêu?
bot: Dân số Hà Nội khoảng 8 đến 8,5 triệu người, ...
you: And what's the weather usually like there in winter?
bot: Hanoi winters are usually cool and quite humid, ...
```

## Setup (once)

The same setup as [transcribe](transcribe.md), plus a DeepSeek key. In `.env` at the repo root:

```
HF_TOKEN=hf_...              # speech → text (Pollen's Hugging Face server)
DEEPSEEK_API_KEY=sk-...      # the LLM; get one at platform.deepseek.com
# LLM_MODEL=deepseek-chat    # optional: deepseek-chat (default, fastest) | deepseek-v4-pro (stronger, thinks first, slower)
# LLM_BASE_URL=https://api.deepseek.com   # optional: any OpenAI-compatible endpoint
GEMINI_API_KEY=...           # text → speech (not needed with --no-speak); aistudio.google.com
# TTS_VOICE=Kore             # optional Gemini voice (e.g. Kore, Puck)
# TTS_MODEL=gemini-3.8-flash-tts
# TTS_PLAYER=none            # optional: save WAVs only, don't play. Default plays through Windows ffplay.exe
```

Playback uses **Windows `ffplay.exe`**, which comes with `winget install Gyan.FFmpeg.Essentials` (see [transcribe.md](transcribe.md)). It plays on the Windows default output, e.g. your headset.

Run commands from the repo folder, because `.env` is read from the folder you run in.

## Use

Pick one input, the same as `transcribe`:

```bash
# Best on WSL: Windows records the mic, and the audio is piped into WSL (see transcribe.md)
ffmpeg.exe -hide_banner -loglevel error -f dshow -audio_buffer_size 50 \
  -i audio="Headset (soundcore R60i NC)" -ac 1 -ar 24000 -f s16le - \
  | uv run reachy-demo agent --stdin

uv run reachy-demo agent --mic                # mic through WSLg
uv run reachy-demo agent --file question.wav  # a recorded question
```

Press **Ctrl-C** to stop. File mode stops by itself when the file ends and the last answer is printed.

## How it works

```
mic / stdin / file ─► speech → text ─► agent (DeepSeek) ─► terminal ─► Gemini TTS ─► headset
                      (one line per    (+ last 10 Q/A)                (streamed)     + data/speech/*.wav
                       finished turn)
```

1. **Your turn ends when you've finished speaking, not at every pause.** The Hugging Face server's smart turn detection decides. A pause of up to about 1.2 s in the middle of a sentence is kept as one turn (tested in English), and only the finished turn goes to the LLM.
2. **The agent** sends a short system prompt, the last 10 Q/A pairs and your question to DeepSeek, and prints the answer as it streams (about 0.5 s to the first word).
3. **Style:** it answers in your language (Vietnamese or English), in at most 3 sentences unless you ask for more, as plain text with no markdown (so it can be spoken later), and it says when it isn't sure.
4. **Speech:** when the text answer is complete, it goes to Gemini (`gemini-3.8-flash-tts`, streaming). Audio plays as it arrives (about 1 s to the first sound) and is saved to `data/speech/<date-time>-<n>.wav` when generation ends. It's saved even if playback fails.
5. **One answer at a time.** While Reachy is answering or speaking, only a stop command does anything (see below). Anything else you say then is ignored, not answered later.

## Interrupting

While Reachy answers, the mic keeps listening.

- **You start talking:** playback drops to 25% volume about 0.2 s later and comes back when you stop. Nothing is cancelled yet.
- **You say a stop command:** "stop", "wait", "enough", "dừng lại", "thôi", or any phrasing. The answer is cancelled (text and speech), queued audio is flushed, and Reachy listens again:
  ```
  you: Tell me about the history of Hanoi.
  bot: Hanoi has been ...
    (stopped)
  ```
- **"Stop, <question>":** it stops, then answers the question: `stop, what time is it?`
- **Anything else** said while Reachy talks (fillers, side talk, a question without a stop word) is dropped and shown as `  (ignored: …)` after the answer.

The stop takes effect after you finish the sentence: turn end (about 0.5–1.2 s) plus the stop check by DeepSeek (about 0.3–0.5 s). Reachy stays at 25% volume meanwhile. A short tail (about 0.2 s) that already reached the speaker can still be heard. If the stop check fails or takes over 2 s, it doesn't stop.

Tune when it ducks with `VAD_RMS` in `.env` (default 500). `uv run python scripts/mic_check.py` prints your mic's `rms`: set `VAD_RMS` above the noise level and below your speech. A headset avoids Reachy hearing itself.

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

## Errors

| What you see | Meaning / fix |
|---|---|
| `config error: DEEPSEEK_API_KEY is required` (exit 2) | Add the key to `.env` |
| `config error: HF_TOKEN is required` (exit 2) | Add the HF token to `.env` |
| `bot: (error: DeepSeek returned 401: ...)` | The key is wrong or revoked. It **keeps listening**, so fix `.env` and restart |
| `bot: (error: DeepSeek returned 402: ...)` | Out of DeepSeek balance. Top up at platform.deepseek.com |
| `bot: ... (error: DeepSeek request failed: ...)` | Network dropped mid-answer. The partial answer stays on screen but isn't remembered, and listening continues |
| `config error: GEMINI_API_KEY is required` (exit 2) | Add the Gemini key, or run with `--no-speak` |
| `  (speech error: Gemini TTS returned 4xx: ...)` | The Gemini key is wrong (400/403) or the quota is used up (429). The text answer is already shown, and it keeps listening |
| `  (speech error: cannot start player 'ffplay.exe' ...)` | Install Windows ffmpeg, or set `TTS_PLAYER`. The WAV is still saved |
| `transcription stopped: ...` (exit 1) | Speech → text failed (network or token). See [transcribe.md](transcribe.md) |
| `error: ...` (exit 2) | Bad file or mic problem. See [transcribe.md](transcribe.md) |

## Limits

- Speech starts after the whole text answer is ready (answers are short, so this adds under 1 s). There's no interrupting the bot while it speaks.
- **Echo:** with laptop speakers the mic can hear the bot and treat it as a new question. Use headphones.
- No tools: it can't move the robot or remember things between runs. Conversation memory lasts only while the command runs.
- Piping audio **faster than real time** (e.g. `ffmpeg` without `-re`) can join separate sentences into one turn. A live mic and `--file` are paced in real time.
- Your audio goes to Pollen's Hugging Face server, your questions go to DeepSeek, and the answers go to Google Gemini for speech.
- Code: [src/reachy_demo/agent/](../../src/reachy_demo/agent/) (`llm.py` DeepSeek client, `agent.py` history and prompt, `cli.py` pipeline), [src/reachy_demo/voice/tts.py](../../src/reachy_demo/voice/tts.py), [src/reachy_demo/audio/player.py](../../src/reachy_demo/audio/player.py).
