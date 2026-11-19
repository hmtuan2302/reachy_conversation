# Reachy Mini Agentic Demo — Product & Design

Date: 2026-11-16 · Status: draft for approval

Legend: **[decided]** = said by the owner. **[call]** = a call I (the designer) made; the owner can overrule it.

---

## 1. What this is and why

A single Python app, running on the owner's laptop, that drives a **Reachy Mini Wireless** over Wi‑Fi. It puts on **one continuous live demo** where all ten abilities run together **[decided]**. The robot greets people, talks in real time, reads emotions, tracks and switches attention, finds objects, mimics head movement, reacts to sounds, remembers things, obeys only its owner's voice, and looks where the owner points.

**Problem it solves:** it shows, in one unbroken run, that a small, cheap robot can act like an aware, attentive agent and not a voice assistant on a stand. **Purpose: a demo only [decided].** It is not a product for daily use.

**Success looks like:** the owner runs the full script (§7) in front of an audience, start to finish, with no restart, no crash, and no visible "dead air" longer than ~3 s.

## 2. Goals, non-goals, anti-goals

**Goals**
- All 10 features live in one process at the same time, in one continuous show **[decided]**.
- Built feature by feature in the owner's order 1→10 **[decided]**, with each step demo-able before the next.
- Runs on a **CPU-only laptop**. Third-party APIs are allowed, and so are light local CPU models **[decided]**. **[changed 2026-11-19]** The laptop has an NVIDIA GPU (~4 GB VRAM free beside local STT and VieNeu TTS), so local GPU models are allowed too (VieNeu TTS, YOLOE in memory Phase 2).
- A lean codebase written **from scratch** that the owner fully understands **[decided]**. It borrows ideas, not code wholesale, from `opensource/apps/reachy_mini_conversation_app`.

**Non-goals**
- Multi-user support, accounts, a web UI, personality profiles, an MCP tool store.
- Running on the robot's own CPU.
- Production hardening (auto-update, telemetry, broad hardware support).
- Languages other than English **[call]**. English is assumed for STT, TTS and voice ID enrollment.

**Anti-goals** (failing these makes the demo a failure even if every feature "works")
- The robot obeys a stranger's spoken command.
- The head jitters or fights between two behaviours (e.g. face tracking vs. sound turn).
- Long silent freezes while a cloud call runs, with no "thinking" cue.
- A crash in one feature (e.g. the object detector) takes down the voice conversation.

## 3. Constraints

| Constraint | Consequence |
|---|---|
| Reachy Mini **Wireless**, app on laptop over Wi‑Fi **[decided]** | Camera, mic and speaker are streamed via the SDK's WebRTC media client. Adds ~100–300 ms latency, and Wi‑Fi quality matters at the venue. |
| Laptop has **no GPU** **[decided]**, **[changed 2026-11-19]**: it has an NVIDIA GPU, ~4 GB VRAM free beside STT and TTS | Small ONNX models still run on CPU. Local GPU models (VieNeu TTS, YOLOE) share the GPU; heavy vision reasoning still goes to a cloud API. |
| SDK `reachy_mini` 1.13 (installed in `~/reachy_mini/reachy_mini_env`) | Provides `get_frame()`, `get_audio_sample()`, `push_audio_sample()`, `set_target()`, `goto_target()`, `look_at_image()`, `look_at_world()`, `get_DoA()`. Daemon REST: `GET /api/state/doa`. |
| DoA reports angle **0 = left, π/2 = front, π = right** | The mic array covers the front half-plane only. "Turns around" means turning left or right toward the side the sound came from. It cannot tell front from back (see Risk R4). |
| Live venue | Noisy room, multiple voices, variable lighting. |

## 4. Approaches considered

1. **Build on the reference conversation app.** Fastest to features 1–2 because it already solves barge-in, echo and motion blending. It carries a large codebase the owner doesn't need (profiles, web UI, MCP).
2. **New lean app from scratch — CHOSEN [decided].** Own realtime client, own motion loop on the SDK, own perception loop. The owner understands every line. It costs extra time re-solving barge-in, echo handling and motion blending. Mitigation: read the reference's `moves.py`, `huggingface_realtime.py` and `audio/` for the techniques, then reimplement them minimally.
3. **Step-by-step pipeline** (VAD → voice ID → STT → LLM → TTS). Voice ID and vocal emotion become trivial, but there's a 1.5–3 s reply delay, which undercuts feature 2 (realtime voice). Rejected.

## 5. Architecture (the parts of the "how" fixed by this design)

```
                ┌──────────── laptop (one Python process, asyncio) ─────────────┐
 Reachy  audio ─┼─► Audio In ─► local VAD ─► Voice Gate (owner?) ─► Realtime LLM │
 Wireless       │                  │                                 (cloud)    │
  (SDK/WebRTC)  │                  └─► Vocal-emotion (optional)        │  tools  │
         video ─┼─► Perception loop (CPU, ~5–10 Hz):                    ▼        │
                │     faces+expression · hands/pointing · objects   Tool layer  │
         DoA  ──┼─► Sound watcher (loud-onset + DoA)                    │        │
                │                     │                                  │        │
                │              World State (latest percepts) ◄──────────┘        │
                │                     │                                           │
                │              Attention Arbiter (priority) ─► Motion loop 50 Hz ─┼─► head/antennas/body
                │              Memory store (JSON/SQLite)                         │
 speaker ◄──────┼── Audio Out (paced chunks, ducking on barge-in)                 │
                └────────────────────────────────────────────────────────────────┘
```

**Components**
- **Realtime voice client [call]:** OpenAI Realtime API (speech-to-speech, with function calling) over WebSocket, so the agent drives everything through tools. **Automatic server turn detection is OFF.** Local VAD cuts each utterance, the Voice Gate decides, then the app sends `input_audio_buffer.commit` (owner) or `clear` (anyone else). This is how feature 9 works with realtime voice.
- **Guests can chat; only the owner can command [decided].** Every utterance reaches the LLM, tagged with who said it.
- **Voice Gate (feature 9):** the owner enrolls with 3–5 short phrases (~30 s total). A speaker-embedding model on CPU (sherpa-onnx speaker-verification model, available in `opensource/vad/sherpa-onnx`) compares each finished utterance to the owner's voiceprint with a cosine threshold (~50–100 ms). The app then adds a short text item `[speaker: owner|guest]` and commits the audio, so the model knows who is speaking before it replies **[call]**.
- **Enforced in code, not only in the prompt [call]:** the tool layer checks the speaker tag of the current turn. **Owner-only tools:** every tool that moves the robot or changes state: find object, look at / track, mimic, look where pointed, move/dance/emotion on request, remember/forget. **Open to everyone:** conversation, self-introduction, recall ("what do you remember about…"), read text / "what is this" only when the *owner* points (pointing is an action). Reactive behaviours (face tracking, emotion mirroring, sound investigation) are autonomous and not gated. When a guest asks for an owner-only tool, the tool returns a refusal and the robot says something like "Sorry, only my owner can ask me that."
- **Barge-in [call]:** anyone can interrupt, since guests can chat. A local VAD onset while the robot is speaking stops playback.
- **Perception loop:** a single thread grabs frames and runs light CPU models **[call: MediaPipe family + a small ONNX object detector]**:
  - face detection + landmarks → face position, head pose (for mimic), facial expression
  - hand landmarks → index-finger ray for pointing
  - small object detector (e.g. YOLO-nano class, COCO) → candidate boxes for "find X"
  - each model runs only when a behaviour needs it, to save CPU
- **Cloud vision [call]:** an OpenAI vision-capable model is called on demand with a JPEG for (a) "which candidate is *my* phone?" / semantic search when the detector lacks the class, (b) "what is this?", (c) reading text.
- **World State:** a plain object with the latest percepts and timestamps. Behaviours read it, and perception writes it.
- **Attention Arbiter:** exactly one behaviour owns the head at a time, picked by priority. A higher priority interrupts, and when it ends the lower one resumes. Expressive moves (emotion, antenna wiggle, speech wobble) are *layered offsets*, never owners. Priority order (highest first) **[call, owner saw it and did not object]**:
  1. Owner's direct command (find, look-at, go-to-pose)
  2. Sudden loud sound → turn and investigate
  3. Owner pointing
  4. Mimic mode (when asked)
  5. Track the face of whoever is speaking/nearest
  6. Idle breathing / glance around
- **Motion loop:** 50 Hz `set_target()` with smoothing and speed limits. It blends the owner's pose with the layered offsets.
- **Memory (feature 8):** a local JSON/SQLite store with tools `remember(fact)`, `recall(query)`, `forget(query)`. It auto-stores (a) facts the owner states, (b) last-seen object locations from feature 5/10, (c) the owner's name. A short summary is injected into the system prompt at session start **[call]**. `opensource/memory/RoboMemory-Code` is future work, not MVP.

## 6. Features — MVP vs future

| # | Feature | MVP (must work in the show) | Future |
|---|---|---|---|
| 1 | Greeting & self-intro | When the owner **or a guest** says "introduce yourself" or something similar **[decided]**, the robot looks at the speaker, waves its antennas, greets them, and gives a short self-introduction (name, what it can do), then asks their name. No automatic greeting when a face appears **[call: the voice request replaces it]**. | Greets known people by name (face ID); proactive greeting on arrival. |
| 2 | Realtime voice | Speech-to-speech conversation, first audio of a reply ≤ 1.5 s after the user stops; the owner can interrupt (barge-in). | Local fallback models when offline. |
| 3 | Emotion-aware | Facial expression (happy / sad / surprised / neutral / angry) from the perception loop is sent to the LLM as context. The LLM picks its tone and an expressive move; the head tilts and antennas match the mood. | Vocal emotion from prosody; fusing face and voice. |
| 4 | Object tracking & attention switching | Follow a named object or face; switch target on command ("look at the cup", "now look at me") or via an arbiter priority change. | Autonomous interest-based switching. |
| 5 | Semantic object search | "Find my phone": scan a fixed pattern of 5–7 head poses. At each pose run the detector, send candidates to cloud vision to confirm, then look at the best one and report "on the table, to your left". Store the location in memory. | Searching rooms with body rotation; multi-object queries. |
| 6 | Mimic head movement | "Copy me": the robot mirrors the owner's head yaw/pitch/roll from face landmarks, smoothed and clamped to safe limits, until "stop". | Mimicking body and antenna gestures. |
| 7 | Sound DoA / something falls | A loud sudden sound (energy onset above an ambient threshold) + DoA angle → turn toward it, run a camera check, and say what it sees ("looks like a book fell"). | Sound-type classification (glass vs. knock). |
| 8 | Robot memory | Remember/recall/forget facts and object locations across restarts; "where did you see my keys?" works. | RoboMemory-style episodic/spatial memory. |
| 9 | Obey only me | Everyone can chat; only owner-verified utterances can trigger owner-only tools. Guests get a polite spoken refusal. | Voice ID + face ID combined; multiple trusted users. |
| 10 | Point & recognize/read | Owner points + asks "what is this?" / "read this": hand ray → region of the image → head looks there → cloud vision names it or reads the text aloud. | Continuous pointing without a voice trigger. |

**Build order:** 1 → 10 as the owner specified **[decided]**, with two adjustments **[call]**:
- **Step 0 — connection spike on this WSL2 laptop** (the demo laptop **[decided]**): connect to the Wireless from WSL2, get camera frames, mic audio, speaker playback and DoA, and measure round-trip audio latency and frame rate. If WSL2 networking (NAT, mDNS discovery, WebRTC) fails or lags, fall back to WSL2 mirrored networking mode, then to running natively on Windows. This must pass before step 1.
- The *plumbing* for feature 9 (manual turn commit, speaker tag, tool-level check that starts as "everyone is owner") is built together with feature 2. Turning the gate on later is then a model swap, not a redesign of the audio path.

**Demo date:** not given. The plan is milestone-based. If time runs short, the "Future" column is cut first; then F10 and F6 become scripted (narrow, rehearsed cases) rather than general.

## 7. Demo script (the acceptance run)

1. A guest says "Introduce yourself" → **greeting and self-intro** (F1). The owner says "Hi, I'm <name>, remember that" → the robot **remembers** it (F8).
2. Free chat for ~30 s between the owner and a guest, with one interruption (F2).
3. The owner smiles / frowns → the robot's tone and pose change (F3).
4. A guest says "dance!" → polite refusal. The owner says it → it obeys (F9).
5. "Look at the cup." … "Now look at me." (F4)
6. "Copy me" → mimic for ~10 s → "stop" (F6).
7. "Find my phone" → search → reports where (F5).
8. A helper drops a book on one side → the robot turns, looks, comments (F7).
9. The owner points at a printed sign → "read this" (F10).
10. "Where was my phone?" → answers from memory (F8). Goodbye.

## 8. Acceptance criteria (measurable)

Measured in the demo room, over **10 full rehearsal runs** of §7, unless stated otherwise.

| Area | Criterion |
|---|---|
| Whole show | ≥ 9/10 runs complete with no restart and no crash; process up ≥ 30 min continuously. |
| Latency | Voice reply first audio ≤ 1.5 s (p90). Tool-based answers (find/read) give a spoken "thinking" cue ≤ 1 s and a final answer ≤ 8 s. |
| F1 | "Introduce yourself" (and 3 paraphrases, from both owner and guest) → greeting + self-intro starts within 1.5 s, in ≥ 9/10 tries. |
| F2 | Barge-in (any speaker) stops robot speech within 0.5 s, in ≥ 9/10 tries. |
| F3 | On a 5-expression test, posed clearly by the owner, ≥ 4/5 correctly labelled, and the response visibly changes. |
| F4 | Tracks a moved object with the head for ≥ 10 s without losing it; switches target within 1.5 s of the command. |
| F5 | Phone placed at 5 different spots in the field of reach: found and correctly located in ≥ 4/5, within ≤ 15 s each. |
| F6 | Mirror lag ≤ 300 ms; no motion beyond safe joint limits. |
| F7 | 10 drops (left/right/front): turns to the correct side ≥ 8/10, within 1 s; false triggers from normal speech ≤ 1 per 10 min. |
| F8 | Facts and object locations survive an app restart; 5/5 recall queries correct. |
| F9 | Owner's commands executed ≥ 95% (≥ 40 tries); commands from 3 different guests refused ≥ 95% (≥ 40 tries); guests' plain chat still answered ≥ 95%. |
| F10 | 5 objects + 3 printed text cards: correct target ≥ 6/8; text read correctly (≥ 90% of words) on cards ≥ 24 pt. |
| Arbitration | No visible head "fight" (oscillation > 2 direction changes/s) across all 10 runs. |
| Resilience | Killing or failing any one perception model or cloud call → the robot says it can't do that right now, and the conversation continues. |

## 9. Risks

| # | Risk | Type | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Voice ID errors in a noisy room (rejects the owner, or a guest slips through) | Tech | Breaks F9 live | Enroll in the venue; tune the threshold on rehearsal data; use a close-talk position; require minimum utterance length ≥ 1 s. |
| R2 | Manual turn commit + local VAD adds latency or cuts the owner off | Tech | F2 feels slow | Tune VAD end-of-speech hangover (~400–600 ms); measure p90 latency early (step 2). |
| R3 | CPU overload from running several vision models at once on a no-GPU laptop | Tech | Lag, dropped frames, audio glitches | Run models only when needed; cap fps; audio has its own thread/priority; profile at each build step. |
| R4 | DoA covers the front half only, and non-speech sounds may give noisy angles | Tech | F7 turns the wrong way | Use onset energy + DoA averaged over the impact window; script drops to the left/right sides; investigate with a camera sweep if unsure. |
| R5 | Wi‑Fi at the venue is congested | Ops | Video/audio lag, disconnects | Bring a dedicated router/hotspot; reconnect logic; rehearse on that network. |
| R6 | Cloud API latency or outage | Ops | Silent freeze | Thinking cues, timeouts, spoken fallback; check internet before the show. |
| R7 | Speaker echo: the robot hears itself as user speech | Tech | Self-interruption, gate confusion | Mute/duck the gate while the robot is speaking unless VAD is strong; borrow the reference app's barge-in technique. |
| R8 | Head behaviours fight (tracking vs. sound vs. pointing) | Product | Looks broken | A single arbiter owns the head (§5); acceptance criterion on oscillation. |
| R9 | Rebuilding from scratch takes longer than expected | Schedule | Later features not ready | Each feature is demo-able on its own; cut "future" items first; the reference app is the cheat sheet. |
| R10 | Pointing ray from a 2D camera is ambiguous in depth | Tech | F10 looks at the wrong thing | Use the region near the fingertip along the ray; confirm with cloud vision; script the object to be near the hand. |
| R12 | Demo laptop runs WSL2: NAT, mDNS discovery and WebRTC can fail or add latency | Tech | Can't connect, or laggy audio/video | Step 0 spike; WSL2 mirrored networking; connect by IP, not name; fallback to native Windows. |
| R13 | Voice ID misreads the owner as a guest (or vice versa) mid-show | Tech | Owner's command refused, or a guest's obeyed | Tune the threshold on rehearsal data; on a refusal the owner can just repeat; operator key to force "owner" for the next turn. |
| R11 | API cost and keys | Ops | Minor | Budget cap on the API account; it's a demo. |

## 10. Left to the builder

- Exact model choices within the stated families, and their thresholds (VAD, voice ID, onset energy, expression confidence).
- Code layout, threading vs. asyncio split, config format (`.env` assumed).
- The exact head-scan pattern for F5, smoothing filters, joint safety limits (from SDK docs).
- Wording of the system prompt and spoken lines.
- Test approach: unit tests for the arbiter, voice gate and memory; manual rehearsal for the rest.
- A ≤ 10-minute on-site calibration step (re-enroll voice, measure ambient noise for VAD/onset thresholds, Wi‑Fi latency check) **[call]**, so tuning doesn't depend on long access to the room.
- A small "operator" keyboard shortcut set (force a feature, reset, mute) as a backstage safety net **[call]**.
