# Features built so far

A one-page summary of everything `reachy-demo` can do. Details, setup and errors are in each feature's usage doc under [docs/usage/](usage/).

**Keep this file current:** every commit that adds or changes a feature updates its row here, and its usage doc.

Last updated: 2026-11-17 (branch `feat/video-tracking`, face tracking, mimic, object tracking and find object added).

## Commands

| Command | What it does |
|---|---|
| `reachy-demo check-config` | Loads `.env` and reports config errors |
| `reachy-demo transcribe --mic \| --stdin \| --file F` | Live speech to text |
| `reachy-demo speak "text"` | Text to speech, played and saved as WAV |
| `reachy-demo agent [--mic \| --stdin \| --file F \| --type] [--robot] [--no-speak]` | The voice agent |
| `reachy-demo enroll --file F [--test X ...]` | Saves the owner's voiceprint for owner mode, prints scores |
| `reachy-demo memory eval \| embed \| reset` | Scripted memory test, vector backfill, wipe the memory DB |

## Features

| # | Feature | What you get | Key settings | Doc |
|---|---|---|---|---|
| 1 | **Live transcribe** | One line of text per finished sentence, Vietnamese or English (auto-detected). Input from WSLg mic, Windows `ffmpeg.exe` piped to `--stdin` (best on WSL), or a WAV. Uses Pollen's HF realtime server. Saves what it heard to `data/recordings/` on exit. | `HF_TOKEN` | [transcribe.md](usage/transcribe.md) |
| 2 | **Voice agent** | Speech → DeepSeek → spoken answer. Answers in your language, max 3 sentences, plain text. Keeps the last 10 Q/A. `--type` to type instead of speak. | `DEEPSEEK_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | [agent.md](usage/agent.md) |
| 3 | **Text to speech** | Gemini streaming TTS (about 1 s to first sound), or local **VieNeu-TTS v3 Turbo** on CPU (no key). VieNeu buffers first, so speech plays without gaps (first sound after about 1–4 s). Each answer saved to `data/speech/`. Played through Windows `ffplay.exe`. | `GEMINI_API_KEY`, `TTS_PROVIDER`, `TTS_VOICE`, `TTS_PLAYER`, `SAVE_SPEECH` | [speak.md](usage/speak.md) |
| 4 | **Barge-in** | When you talk, playback ducks to 25%. A spoken stop ("stop", "thôi", "dừng lại", any wording) cancels text, speech and moves. A bare stop phrase stops at once. Longer turns go to a fast DeepSeek check. "Stop, <question>" stops, then answers; "stop tracking and mimic me" stops, then switches. Other side talk is dropped. | `VAD_RMS` | [agent.md § Interrupting](usage/agent.md#interrupting) |
| 5 | **Head motion and dances** (`--robot`) | Tools `turn_head`, `tilt_head`, `roll_head` (fixed angles), `dance` (20 Pollen library moves, max 15 s), `stop_head`. One motion loop (about 50 Hz) is the only sender of poses. Every pose is clamped to safety limits. Moves are queued, and new moves are refused while dancing. Can move and talk at the same time. On exit the head goes to the sleep pose, then motors turn off. | `ROBOT_HOST` | [agent.md § Moving the head](usage/agent.md#moving-the-head-and-dancing---robot) |
| 6 | **Camera / vision** (`--robot`) | "What is this?": the `camera` tool grabs a frame and asks a vision model. HF server about 5 s, DeepSeek about 2.7 s. | `VISION_PROVIDER` (`hf` / `deepseek`), `VISION_MODEL` | [agent.md § Seeing](usage/agent.md#seeing-camera) |
| 7 | **Voice cloning** | "Clone my voice" saves your last 10 s of speech (3 s minimum) to `data/voice/clone.wav`. Answers then use your voice (VieNeu only). "Stop using my voice" deletes the file. | `TTS_PROVIDER=vieneu`, `CLONE_AUDIO` | [agent.md § Voice cloning](usage/agent.md#voice-cloning) |
| 8 | **Long-term memory** | SQLite `data/memory.db` keeps people (name, earlier names, description), facts (changes tracked, never overwritten), one summary per conversation, and follow-ups. A `[memory]` block is built at the start of each conversation. Tools `recall`, `remember`, `forget`. Sensitive facts are stored only on request. If you restart within 2 h, the last 5 turns are reloaded. | `MEMORY`, `MEMORY_DB`, `MEMORY_MODEL`, `MEMORY_MAX_TOKENS` | [memory.md](usage/memory.md) |
| 9 | **Search by meaning + related memories** | Local `embeddinggemma-300m` (ONNX, CPU). Hybrid recall combines vectors, keywords and recency, and works across Vietnamese and English. Each turn can push up to 3 related memories (similarity ≥ 0.55), and they are never stored. | `EMBED`, `EMBED_MODEL`, `MEMORY_PUSH` | [memory.md § How Reachy finds](usage/memory.md#how-reachy-finds-whats-relevant) |
| 10 | **Context budget** | Each LLM request stays under a token cap. When over, the oldest whole turns are dropped. `CONTEXT_DEBUG=on` prints the memory block, related memories and tool results. | `CONTEXT_MAX_TOKENS`, `CONTEXT_DEBUG` | [agent.md § Debugging the context](usage/agent.md#debugging-the-context) |
| 11 | **Sound direction (DoA)** (`--robot --mic`) | Stores the direction of your last sentence (30 s) and of the last loud sound (2 min), from the robot's mic array. The `look_toward_sound` tool turns head and body toward it (body ±120°, head ±45°, 60°/s). If the spot is out of reach, Reachy says so. | `SOUND_RMS` | [sound-direction.md](usage/sound-direction.md) |
| 12 | **Memory testing** | Hand-test scenarios with SQLite helpers, plus a scripted eval on a simulated clock. Last result: 54/55. | `MEMORY_DB` | [memory-testing.md](usage/memory-testing.md) |
| 13 | **Owner mode (speaker verification)** | "Only follow my command" turns it on: each sentence is scored against your voiceprint (CAM++ via sherpa-onnx, CPU, about 50 ms). Guests are ignored, unsure voices are asked to repeat, and `dance` / `look_toward_sound` refuse guests. "Stop following only me" turns it off. Only your voice can switch it. "Am I your boss?" calls `who_is_speaking`, which answers from the voice check. `reachy-demo enroll --file F` saves the voiceprint. | `SPEAKER_MODEL`, `SPEAKER_HIGH`, `SPEAKER_LOW` | [agent.md § Owner mode](usage/agent.md#owner-mode-speaker-verification) |
| 14 | **Face tracking** (`--robot`) | "Track me" / "follow me": tool `track_face` follows your face with head and body until you say stop. Faces come from the robot's own detector (YuNet on the robot, no laptop CPU). With no face in view, or when someone talks from more than 35° away, it turns toward the voice (DoA). Head turns alone up to 25°, then the body turns too. Other moves are refused while it follows. Owner only in owner mode. | `ROBOT_HOST` | [face-tracking.md](usage/face-tracking.md) |
| 15 | **Mimic** (`--robot`, camera) | "Mimic me" / "bắt chước tôi": tool `mimic_me` copies your head turn, nod, tilt and lean like a mirror, about 1:1, until you say stop. YuNet finds your face, then MediaPipe reads the head angles on a crop. Smoothed by a one-euro filter and a spring (max 150°/s). Limits: ±35° turn and ±22° nod from facing you (the camera must keep seeing you), ±25° tilt, ±20 mm lean. Owner only in owner mode. | `MIMIC_MODEL`, extra `perception` | [mimic.md](usage/mimic.md) |
| 16 | **Object tracking** (`--robot`, camera) | "Track my iPhone" / "follow the cup": tool `track_object` finds the object with YOLOX-nano (80 everyday classes). With several candidates, the vision model picks "mine". An OpenCV CSRT tracker then follows it at 15–25 Hz, and the detector corrects it every 1 s. It turns head and body like face tracking until you say stop, and gives up after 10 s unseen. Owner only in owner mode. | `OBJECT_MODEL`, `VISION_PROVIDER`, extra `perception` | [object-tracking.md](usage/object-tracking.md) |
| 17 | **Find object** (`--robot`, camera) | "Find my phone" / "tìm cái cốc": tool `find_object` makes one smooth sweep (front → left 160° → right 160° → front, body and head together) and runs YOLOX on 3×3 tiles while moving. On a candidate it brakes smoothly and checks a still frame: a lone confident hit is accepted, otherwise the vision model checks it. A look-alike is skipped and the sweep resumes. Reachy centres the object, describes it (vision model), says where it is and asks "is that it?". "No" resumes the sweep past it. At most 2 sweeps (~40 s, `SEARCH_SWEEP_S`), then "couldn't find it". No memory of sightings yet. Owner only in owner mode. | `OBJECT_MODEL`, `VISION_PROVIDER`, `SEARCH_SWEEP_S` | [find-object.md](usage/find-object.md) |

## Not built yet

From the roadmap in [src/CLAUDE.md](../src/CLAUDE.md): greeting, emotion-aware replies, attention arbiter, object sightings in memory, face check for unsure voices in owner mode, point-and-recognize. Later memory phases are also open: object locations, reminders, routines, and forgetting old memories over time.

## Branches

| Branch | Holds |
|---|---|
| `main` | Features 1–13 |
| `feat/doa-speaker-verification` | Merged into `main` |
| `feat/video-tracking` | `main` + face tracking, mimic, object tracking and find object (features 14–17), not merged yet |
