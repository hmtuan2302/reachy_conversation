# Voice agent: ask by voice, hear the answer

Speak a question. The agent prints the answer in the language you spoke, says it aloud, and saves the spoken answer as a WAV file. Pipeline: **speech → text (HF server) → LLM (DeepSeek) → speech (Gemini)**.

```
$ uv run reachy-demo agent --file data/girl-saying-hello_E_major.wav
you: Hello.
bot: Hello! How can I help you today?
  (saved data/speech/20261116-085901-1.wav)      ← spoken through your headset, then saved
```

Add `--no-speak` for text answers only.

It remembers the conversation (the last 10 questions and answers), so follow-ups work. It also remembers you **across runs**: see [memory.md](memory.md).

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
GEMINI_API_KEY=...           # text → speech (not needed with --no-speak or TTS_PROVIDER=vieneu); aistudio.google.com
# TTS_PROVIDER=vieneu        # local VieNeu-TTS v3 Turbo instead of Gemini, see speak.md
# TTS_VOICE=Kore             # optional Gemini voice (e.g. Kore, Puck)
# TTS_MODEL=gemini-3.8-flash-tts
# TTS_PLAYER=none            # optional: save WAVs only, don't play. Default plays through Windows ffplay.exe
# SAVE_SPEECH=off            # optional: don't save each answer to data/speech/ (no "(saved ...)" line)
# MEMORY=off                 # optional: no long-term memory (see memory.md)
# CONTEXT_MAX_TOKENS=8000    # optional: cap for each LLM request (system prompt + memory + tools + history), approx chars/4
# MEMORY_MAX_TOKENS=600      # optional: the memory block's share of that cap
# CONTEXT_DEBUG=on           # optional: print the context the model gets (see "Debugging the context")
```

Each request stays under `CONTEXT_MAX_TOKENS`: when it would go over, the oldest whole turns of the conversation are left out (never half a turn, never your current question). Nothing is deleted, so a later, smaller request can send them again.

### Debugging the context

Set `CONTEXT_DEBUG=on` in `.env` (or `CONTEXT_DEBUG=on uv run reachy-demo agent --type`) to see what the model is given, printed before and during each answer:

```
you: I'm thinking about cooking seafood for dinner tonight.
bot: 
  ┌ memory block
  │ [memory] Now: Mon 17 Nov 2026 20:00. You are talking to Tuan (owner); last talked 1 day ago.
  │ Known about Tuan:
  │ - Tuan has a sister named Mai. (said, 2 weeks ago)
  │ Use recall for anything else. ...
  └
bot: 
  ┌ related memories
  │ [related memories — use only if it fits the conversation]
  │ - fact: Tuan is allergic to shrimp. [said, 4 days ago]
  └
bot: Careful, you're allergic to shrimp, so maybe grilled fish?
```

| Block | Shown |
|---|---|
| `memory block` | When it changes, so once per conversation (it's built at the start) |
| `related memories` | Each turn that has some (needs memory search by meaning, see [memory.md](memory.md#related-memories-every-turn)) |
| `tool <name>` | Every tool result as it comes back, e.g. `tool recall` with what memory search found, or `tool turn_head` |

It only prints; nothing changes in what's sent, stored or spoken. Off by default. `bot: ` is printed again after each block so the answer stays on its own line.

With memory's related memories on (`MEMORY_PUSH`, see [memory.md](memory.md#related-memories-every-turn)), up to 3 short memory lines are sent in front of your message. They count as part of your current turn, and startup checks that 200 tokens are left for them.

Playback uses **Windows `ffplay.exe`**, which comes with `winget install Gyan.FFmpeg.Essentials` (see [transcribe.md](transcribe.md)). It plays on the Windows default output, e.g. your headset.

Run commands from the repo folder, because `.env` is read from the folder you run in.

## Use

Pick one input: the same three as `transcribe`, plus `--type`:

```bash
# Best on WSL: Windows records the mic, and the audio is piped into WSL (see transcribe.md)
ffmpeg.exe -hide_banner -loglevel error -f dshow -audio_buffer_size 50 \
  -i audio="Headset (soundcore R60i NC)" -ac 1 -ar 24000 -f s16le - \
  | uv run reachy-demo agent --stdin

uv run reachy-demo agent --mic                # mic through WSLg
uv run reachy-demo agent --file question.wav  # a recorded question
uv run reachy-demo agent --type               # type instead of speaking
```

### Typing instead of speaking (`--type`)

Each line you type and send with **Enter** is one turn, just like a finished sentence when speaking. Empty lines are skipped. **Ctrl-D** ends the input (after the last answer), and **Ctrl-C** quits at once.

```bash
uv run reachy-demo agent --type --robot            # type commands, the robot moves and answers aloud
uv run reachy-demo agent --type --robot --no-speak # text answers only
```

- **What you skip:** speech-to-text and the mic. `HF_TOKEN` isn't needed, and nothing is saved to `data/recordings`.
- **Speaking still works:** answers are spoken unless you add `--no-speak`.
- **Stop:** typing `stop` (or `thôi`) while Reachy answers or moves works like saying it.
- **Ducking:** playback isn't turned down while you type.
- **Display:** your line appears twice, once as you type it and once as `you: …`.
- **Piping works too:** `printf 'turn left\ndance\n' | uv run reachy-demo agent --type --robot` runs a fixed script of turns.

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

## Moving the head and dancing (`--robot`)

With `--robot`, the agent can move Reachy's head, dance and look through its camera when you ask it to: "turn left", "look up", "quay sang phải và nhìn xuống", "dance and tell me a joke", "what is this?".

```bash
uv sync --extra robot --extra mic --extra embed   # once: Reachy Mini SDK, dances library, laptop mic, memory search by meaning
uv run reachy-demo agent --mic --robot         # robot at ROBOT_HOST in .env (use its IP)
uv run reachy-demo agent --type --robot        # or type the commands
```

`uv sync` installs exactly the extras you list and removes the others: a plain `uv sync` removes the SDK again, and leaving out `--extra embed` turns memory search by meaning off (`memory search: keywords only (install: …)`). Always list every extra you use, e.g. add `--extra vieneu` for local TTS.

### The tools

| Tool | Arguments | What it does |
|---|---|---|
| `turn_head` (yaw) | `left` / `right` / `center` | +20° / −20° / 0°, in 1 s |
| `tilt_head` (pitch) | `up` / `down` / `center` | −15° / +15° / 0°, in 1 s |
| `roll_head` (roll, lean toward a shoulder) | `left` / `right` / `center` | +15° / −15° / 0°, in 1 s |
| `dance` | `move`: one of the 20 moves below (omit for a random one); `repeat` | 2–5 s per pass, at most 15 s |
| `stop_head` | (none) | ends a dance now and drops queued moves |

- Head moves go to a **fixed angle only**. Each one changes its own axis and keeps the others, so "right and down" ends at yaw −20°, pitch +15°. Asking twice ("more left") stays at +20°.
- Change the angles in `POSES` in [src/reachy_demo/agent/tools/head.py](../../src/reachy_demo/agent/tools/head.py).

### Dances

The dances are the moves of Pollen's `reachy_mini_dances_library`, the same ones the reference app's `dance` tool uses. They're installed with the `robot` extra. Without the library, the `dance` tool isn't offered.

`simple_nod`, `head_tilt_roll`, `side_to_side_sway`, `dizzy_spin`, `stumble_and_recover`, `headbanger_combo`, `interwoven_spirals`, `sharp_side_tilt`, `side_peekaboo`, `yeah_nod`, `uh_huh_tilt`, `neck_recoil`, `chin_lead`, `groovy_sway_and_roll`, `chicken_peck`, `side_glance_flick`, `polyrhythm_combo`, `grid_snap`, `pendulum_swing`, `jackson_square`

- The model picks the move from its description ("do a dizzy dance" → `dizzy_spin`), or a random one when you just say "dance".
- A dance moves the whole head: tilt and turn, **position** (sway, bounce, peck forward) and **both antennas**.
- **Length:** one pass is 2.1–5.3 s at the library's 114 BPM. `repeat` adds passes, up to 15 s in total. Longer requests are cut to 15 s.
- **How it plays:** our motion loop plays it, not the SDK's `play_move`, so it obeys the busy rules, stop and exit below. It eases in from the current pose and ends at neutral.

### Safety limits

Whatever a move computes, no pose outside these limits is ever sent:

| Axis | Limit |
|---|---|
| yaw / pitch / roll | ±45° / ±30° / ±25° |
| head position x / y / z | ±20 / ±40 / ±40 mm |
| antennas | ±70° |

These are the widest values the 20 library moves use (sampled), so the dances play unclipped. The head tools stay at their smaller fixed angles. Limits and the loop are in [src/reachy_demo/robot/motion.py](../../src/reachy_demo/robot/motion.py), and the library wrapper is in [src/reachy_demo/robot/dances.py](../../src/reachy_demo/robot/dances.py).

### One motion loop

All moves go into **one queue**, played by **one loop** ([motion.py](../../src/reachy_demo/robot/motion.py)). The loop is the only code that sends poses (head and antennas) to the robot: about 50 `set_target()` calls a second. Two moves can never fight over the head.

- A tool call returns at once with `started …`. The agent doesn't wait for the head, so Reachy can talk while it moves.
- Moves run one at a time, in order. At most 4 can wait. Another one gets `busy: too many moves queued`.
- At startup the motors are enabled and the head and antennas ease to neutral (2 s), so the loop starts from a known pose. Neutral antennas are the SDK's ±10° (less shaking than straight up).

**Rules while moving:**

| Case | Rule |
|---|---|
| Moves from one request ("turn right and look down") | Queued, run in order |
| A dance after short moves ("look up, then dance") | Queued after them |
| Any move or dance **while a dance runs or waits** | Refused with `busy: dancing, say stop first`. Reachy says so aloud. |
| Stop | Always accepted |

### What you hear

| You say | What happens |
|---|---|
| "turn right", "quay sang phải và nhìn xuống", "dance" | Reachy moves and says nothing (empty `bot:` line) |
| "turn left and tell me a joke", "dance and tell me a joke" | Reachy moves **and** answers at the same time. The model sets `reply: true` on the move, which triggers one more LLM call for the words. |
| "turn left" while dancing | "I'm still dancing, say stop first." |

Measured with DeepSeek and a fake robot: for "dance and tell me a joke", the joke text was ready about 2 s after the turn ended, while the dance was running. Speech then adds about 1 s of Gemini TTS.

### Stopping

- **While Reachy answers:** "stop", "dừng lại", "thôi" cancels the answer and the moves.
- **While Reachy only moves** (for example after "dance"): every turn you say first goes through the same fast stop check (about 0.3–0.5 s). "Stop", "enough" or "thôi" stops the moves without asking the agent. Anything else goes to the agent as usual.
- **"Stop dancing"** can also reach the agent, which calls `stop_head`.
- **What stops:** a dance ends at once and the head eases back to neutral (0.5 s). A 1 s head move already in progress finishes, so head moves always end on their fixed angle. Queued moves are dropped. Right after a stop, new moves are accepted.
- One stop stops everything: "stop talking" also ends a dance.

### Seeing (camera)

With `--robot`, Reachy can also look. Put a cup in front of it and ask "what is this?", "what am I holding?", "bạn thấy gì?" or "look at this".

```
you: What is this in front of you?
bot: I see a white ceramic mug on the table.
```

How it works:
1. The agent calls the `camera` tool with your question (in English).
2. The tool takes the current frame from the robot's head camera (1280×720 JPEG over WebRTC).
3. The frame and question go to the **vision model** (pick the provider below).
4. The answer comes back as text, and Reachy says it in your language.

**Pick the vision provider in `.env`:**

```
VISION_PROVIDER=hf             # default: the Reachy Hugging Face realtime server (HF_TOKEN)
VISION_PROVIDER=deepseek       # DeepSeek chat with images (DEEPSEEK_API_KEY)
# VISION_MODEL=deepseek-chat   # deepseek only; default deepseek-chat
# VISION_BASE_URL=...          # deepseek only; any OpenAI-compatible endpoint (default LLM_BASE_URL)
```

| Provider | Model | Question → answer text | Notes |
|---|---|---|---|
| `hf` | Pollen's realtime server (model not disclosed) | about 5.0 s | The same server and `HF_TOKEN` as speech-to-text. What the reference app uses. |
| `deepseek` | `deepseek-chat` | about 2.7 s | Fastest. Served as `deepseek-flash` without reasoning. |
| `deepseek` | `deepseek-flash` | slower | Reasons first. Also sees images. |
| `deepseek` | `deepseek-v4-pro` | n/a | **Can't see images.** It answers "I'm unable to view the image". |

Times were measured 2026-11-16 with a test image and the full agent round trip. The terminal shows the provider at startup: `  (camera on, vision: deepseek deepseek-chat)`. An unknown `VISION_PROVIDER` stops the app with a config error (exit 2).

- **Speed:** about 2.7–5 s from your question to the answer text, depending on the provider (table above), then the spoken answer.
- **Follow-ups** ("what color is it?") may be answered from the last look without taking a new photo. Say "look again" to force a new one.
- **Aim:** the camera sees what's in front of the head. Put the object about 30–60 cm in front of it, in good light. You can say "look down" first for things on the table.
- **No camera:** if the camera stream can't start, you see `  (no camera: the robot can move but not see)` at startup. Head moves and dances still work.
- **Setup:** with `VISION_PROVIDER=hf`, `HF_TOKEN` is needed with `--robot`, even with `--type`. With `deepseek`, `--type --robot` needs no `HF_TOKEN`.

### Errors and exit

- If sending a pose fails (Wi-Fi drop, robot asleep), you see `  (head error: …)`. The queue is dropped, the loop is free again (it never stays `busy`), and the next move restarts it. Reachy doesn't say it failed: it already answered when the move was queued.
- On exit (Ctrl-C, an error or the end of input), whatever runs, even a dance, ends at once. The head eases to neutral, then is lowered into the robot's rest pose (2 s, antennas folded, the same pose as the SDK's `goto_sleep()`). Only then are the motors disabled, so the head doesn't drop. It doesn't wait for a dance to finish.
- **Motors stay off after exit.** The head rests in the sleep pose, and the next app or script must call `mini.enable_motors()` before it moves anything. `agent --robot` does that at startup. Your own scripts need it too, or the robot ignores their moves.

## Voice cloning

With `TTS_PROVIDER=vieneu` ([speak.md](speak.md#local-model-vieneu-tts-v3-turbo)), Reachy can answer in your own voice.

```
you: clone giọng của tôi đi
bot: Được rồi, từ giờ mình sẽ nói bằng giọng của bạn.     ← spoken in your voice
...
you: đừng dùng giọng tôi nữa
bot: Ok, mình quay lại giọng cũ nhé.                       ← back to the preset voice
```

- The mic keeps the last 10 s of your speech in memory, counting only moments when you were talking. "Clone my voice" (in any wording, Vietnamese or English) makes the agent call `clone_voice`, which saves that audio to `data/voice/clone.wav`.
- It needs at least 3 s of your speech. If you've just started talking, Reachy asks you to say a bit more, and then you ask again. Talking longer before asking gives a better clone, because VieNeu uses up to 8 s.
- While `data/voice/clone.wav` exists, every answer is spoken in that voice, even after a restart. "Stop using my voice" calls `stop_cloning_voice`, which deletes the file. Deleting the file by hand does the same.
- The first answer in a new voice takes about 1.5 s longer, while VieNeu encodes the clip. After that it's as fast as a preset voice.
- Use a headset: if the speaker's sound reaches the mic, Reachy's own voice can end up in the sample.
- Owner-only (like memory changes). Until speaker ID exists, everyone counts as the owner.
- To test without recording yourself, set a fixed WAV in `.env`: `CLONE_AUDIO=data/OSR_us_000_0010_8k.wav`. Then "clone my voice" uses that file instead of the mic, so it also works with `--type`, and "stop cloning" turns it off as usual. Any sample rate works, and VieNeu uses up to 8 s. Leave it empty to clone your real voice from the mic.
- Not available with Gemini TTS or `--no-speak`. With `--type` and no `CLONE_AUDIO`, there's no mic sample, so it answers "heard 0 s".

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
| `config error: HF_TOKEN is required` (exit 2) | Add the HF token to `.env` (speech-to-text, and the camera with `VISION_PROVIDER=hf`), or use `--type` |
| `config error: VISION_PROVIDER must be one of hf, deepseek` (exit 2) | Fix `VISION_PROVIDER` in `.env` |
| `config error: CONTEXT_MAX_TOKENS must be a positive integer` (or `MEMORY_MAX_TOKENS`) (exit 2) | Use a whole number above 0 |
| `config error: MEMORY_MAX_TOKENS must be below CONTEXT_MAX_TOKENS` (exit 2) | Lower `MEMORY_MAX_TOKENS` or raise `CONTEXT_MAX_TOKENS` |
| `config error: tool specs (~N tokens) leave no room in CONTEXT_MAX_TOKENS` (exit 2) | Raise `CONTEXT_MAX_TOKENS`; the tools alone use about N tokens |
| `config error: related memories (MEMORY_PUSH) need ~200 more tokens …` (exit 2) | Raise `CONTEXT_MAX_TOKENS` or set `MEMORY_PUSH=off` |
| `  (no camera: the robot can move but not see)` | The WebRTC camera stream didn't start. Check the robot's network. Head moves still work |
| Reachy says it can't see, or that it's too dark | The frame was empty, or the vision model failed (or took over 15 s). Check the light and what's in front of the camera. With `deepseek`, check that `VISION_MODEL` sees images (not `deepseek-v4-pro`) |
| `bot: (error: DeepSeek returned 401: ...)` | The key is wrong or revoked. It **keeps listening**, so fix `.env` and restart |
| `bot: (error: DeepSeek returned 402: ...)` | Out of DeepSeek balance. Top up at platform.deepseek.com |
| `bot: ... (error: DeepSeek request failed: ...)` | Network dropped mid-answer. The partial answer stays on screen but isn't remembered, and listening continues |
| `config error: GEMINI_API_KEY is required` (exit 2) | Add the Gemini key, run with `--no-speak`, or set `TTS_PROVIDER=vieneu` ([speak.md](speak.md)) |
| `  (speech error: Gemini TTS returned 4xx: ...)` | The Gemini key is wrong (400/403) or the quota is used up (429). The text answer is already shown, and it keeps listening |
| `  (speech error: cannot start player 'ffplay.exe' ...)` | Install Windows ffmpeg, or set `TTS_PLAYER`. The WAV is still saved |
| `config error: ROBOT_HOST is required` (exit 2) | Set the robot's IP in `.env`, or drop `--robot` |
| `  (head error: …)` | A head move failed (robot asleep, Wi-Fi drop). Queued moves are dropped and the chat goes on |
| `robot error: cannot connect to ...` (exit 1), other cause | Robot off, wrong IP, or not on the same Wi-Fi. Under WSL2 use the IP with mirrored networking |
| `robot error: ... No module named 'reachy_mini'` (exit 1) | Run `uv sync --extra robot --extra mic --extra embed` (plus any other extras you use) |
| `transcription stopped: ...` (exit 1) | Speech → text failed (network or token). See [transcribe.md](transcribe.md) |
| `error: ...` (exit 2) | Bad file or mic problem. See [transcribe.md](transcribe.md) |

## Limits

- Speech starts after the whole text answer is ready (answers are short, so this adds under 1 s). There's no interrupting the bot while it speaks.
- **Echo:** with laptop speakers the mic can hear the bot and treat it as a new question. Use headphones.
- Tools: the memory tools `recall`, `remember` and `forget` always ([memory.md](memory.md)), plus the head and camera tools with `--robot`.
- Piping audio **faster than real time** (e.g. `ffmpeg` without `-re`) can join separate sentences into one turn. A live mic and `--file` are paced in real time.
- Your audio goes to Pollen's Hugging Face server, your questions go to DeepSeek, and the answers go to Google Gemini for speech.
- Code: [src/reachy_demo/agent/](../../src/reachy_demo/agent/) (`llm.py` DeepSeek client, `agent.py` history, prompt and tool loop, `cli.py` pipeline, `tools/head.py` head tools, `tools/memory.py` memory tools), [src/reachy_demo/robot/adapter.py](../../src/reachy_demo/robot/adapter.py), [src/reachy_demo/voice/tts.py](../../src/reachy_demo/voice/tts.py), [src/reachy_demo/audio/player.py](../../src/reachy_demo/audio/player.py).
