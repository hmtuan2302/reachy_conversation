# Voice agent: ask by voice, get an answer (text)

Speak a question and the agent answers it in the terminal, in the language you spoke. This is stage 1 of the pipeline **speech → text → LLM → speech**; speaking the answer aloud comes in a later stage.

```
$ uv run reachy-demo agent --file data/girl-saying-hello_E_major.wav
you: Hello.
bot: Hello! How can I help you today?
```

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
```

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
mic / stdin / file ─► speech → text ─► agent (DeepSeek) ─► terminal
                      (one line per    (+ last 10 Q/A)    (later: text → speech)
                       finished turn)
```

1. **Your turn ends when you've finished speaking, not at every pause.** The Hugging Face server's smart turn detection decides. A pause of up to about 1.2 s in the middle of a sentence is kept as one turn (tested in English), and only the finished turn goes to the LLM.
2. **The agent** sends a short system prompt, the last 10 Q/A pairs and your question to DeepSeek, and prints the answer as it streams (about 0.5 s to the first word).
3. **Style:** it answers in your language (Vietnamese or English), in at most 3 sentences unless you ask for more, as plain text with no markdown (so it can be spoken later), and it says when it isn't sure.
4. **One answer at a time.** If you finish another turn while it's answering, that turn waits and is answered next.

## Errors

| What you see | Meaning / fix |
|---|---|
| `config error: DEEPSEEK_API_KEY is required` (exit 2) | Add the key to `.env` |
| `config error: HF_TOKEN is required` (exit 2) | Add the HF token to `.env` |
| `bot: (error: DeepSeek returned 401: ...)` | The key is wrong or revoked. It **keeps listening**, so fix `.env` and restart |
| `bot: (error: DeepSeek returned 402: ...)` | Out of DeepSeek balance. Top up at platform.deepseek.com |
| `bot: ... (error: DeepSeek request failed: ...)` | Network dropped mid-answer. The partial answer stays on screen but isn't remembered, and listening continues |
| `transcription stopped: ...` (exit 1) | Speech → text failed (network or token). See [transcribe.md](transcribe.md) |
| `error: ...` (exit 2) | Bad file or mic problem. See [transcribe.md](transcribe.md) |

## Limits

- Text only: the answer isn't spoken yet (the next stage).
- No tools: it can't move the robot or remember things between runs. Conversation memory lasts only while the command runs.
- Piping audio **faster than real time** (e.g. `ffmpeg` without `-re`) can join separate sentences into one turn. A live mic and `--file` are paced in real time.
- Your audio goes to Pollen's Hugging Face server, and your questions go to DeepSeek.
- Code: [src/reachy_demo/agent/](../../src/reachy_demo/agent/) (`llm.py` DeepSeek client, `agent.py` history and prompt, `cli.py` pipeline).
