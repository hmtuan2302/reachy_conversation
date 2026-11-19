# Vision-language-action (VLA) use cases for Reachy Mini

Date: 2026-11-18
Scope: what a VLA-style "see → understand → act" loop can do on Reachy Mini, how each use case maps onto the existing design in [`docs/architecture/`](../architecture/README.md), and the order to build them in this repo (`reachy_demo`).

---

## 0. TL;DR

1. **Reachy Mini has no arms.** Its actions are head pose (6-DoF), body yaw, two antennas and speech. Here a VLA drives **attention and expression** (where to look, how to react, what to say), not manipulation. The action space is small, safe and cheap to record.
2. **Start without training anything.** Use a VLM (slow loop, 1–2 Hz) that picks from a fixed list of action tools, plus local CPU models (fast loop, 10–50 Hz) that do the smooth tracking. Most use cases below work this way with the perception and motion design that already exists.
3. **Train a real VLA policy only for continuous, reactive behaviour**, meaning gaze toward the speaker and imitation, plus the VLA-only use cases in §4.2 (active listening, dancing with a partner, camera operator, reflex games and others). Use SmolVLA via LeRobot. Training and inference need a GPU box: the demo laptop is CPU-only (WSL2).
4. **First build:** "find my ___" (U1) plus object memory (U2). Both reuse the voice agent, `memory/` and the F5 search design, and they make the strongest demo.

---

## 1. What "action" means on this robot

| Output | Range (soft limits, [04 §4](../architecture/04-motion-and-attention.md)) | Rate | Owner in the design |
|---|---|---|---|
| Head pose (yaw, pitch, roll, x, y, z) | yaw ±60°, pitch −30°/+25°, roll ±20° | 50 Hz motion loop | Arbiter base target + offsets |
| Body yaw | ±160° | 50 Hz | Arbiter base target |
| Antennas | per SDK | 50 Hz | Expressive moves (offsets) |
| Speech | — | per turn | Voice agent (`agent/`, `voice/`) |

Rule from [04](../architecture/04-motion-and-attention.md): `robot.set_target()` has **one caller**, the motion loop. Any VLA output, whether a tool call or a learned policy, becomes a **behaviour** (base target) or an **offset**. It never calls the SDK directly. That keeps smoothing, limits and the watchdog in force.

---

## 2. Use cases

Ranked by usefulness and how much existing work they reuse. The "Design hook" column links each use case to a feature ID in the architecture docs where one exists.

| # | Use case | Input → action | Design hook | New work |
|---|---|---|---|---|
| U1 | **"Find my ___"** | spoken goal + frames → body/head scan → centre the object → say where it is | F5 `Search` ([03 §5](../architecture/03-perception.md)) | detector + `confirm`/`locate` vision calls |
| U2 | **Object memory** ("where did I leave…") | periodic glances → sightings in memory → answer later | `sightings` table, memory research P2 | background glance behaviour, sighting writer |
| U3 | **Look at whoever is speaking** | DoA + faces → head turns to the speaker, nods, antennas "listening" | `FaceTrack` + DoA bias | candidate for a learned policy (§4) |
| U4 | **"Look at this"** | pointing or held-up object → head frames it → answer (read a label, identify a part) | F10 `PointFollow` | hand landmarker, crop before the cloud call |
| U5 | **Reading / homework buddy** | page under the camera → read, explain, gaze at the line being discussed | `describe(mode=read_text)` | a "look down" pose, line-level boxes |
| U6 | **Focus / posture coach** | slouch, phone use, absence → small gesture first, speech only if ignored | face landmarks (head pitch) | timer behaviour, nudge policy |
| U7 | **Gesture commands** | wave = wake, thumbs up = yes, palm = stop talking, point = look there | hand landmarker | gesture classifier on landmarks |
| U8 | **Task-step tracker (cooking, assembly)** | watches the work area, checks a step is done, warns ("the pan is smoking") | `Investigate`, cloud `describe` | step list in context, event-driven checks |
| U9 | **Mirror game / imitation** | owner's head pose → mirrored head pose; "Simon says" on top | F6 `Mimic` | best first **training** project (§4) |
| U10 | **Telepresence pointing** | a remote caller says "look at the whiteboard" → find and frame it | `Search` + `LookAt(label)` | remote command input |
| U11 | **Greet people by name** | known face → greeting with context from memory | person memory (research P1/P4) | face ID: **biometric data**, local and opt-in only |
| U12 | **Pet / plant / door watcher** | scheduled scans: "did the cat eat?", "has the package arrived?" | `Search` poses + `describe` | scheduler, keep frames on the device |

### What each one gives the user

- **U1, U2, U10** answer "where is it?" A stationary robot with a wide camera is good at this, and memory makes the answer useful after the object is out of view.
- **U3, U7, U9** make the robot feel alive. They are about timing and smoothness, which is where a learned policy beats hand-written rules.
- **U4, U5, U8** are active vision: the robot moves to get a better view **before** answering. This is the VLA idea that matters most for an armless robot.
- **U6, U12** are long-running watchers. Their value comes from staying quiet until they need to speak.

---

## 3. Architecture: two loops, no training

```
                 ┌──────────── slow loop, 1–2 Hz ────────────┐
 mic → S2T ──►   │ LLM / VLM (DeepSeek, HF realtime vision)  │ ──► tool call
 frame (JPEG) ─► │ "what should I do next?"                  │       │
                 └───────────────────────────────────────────┘       ▼
                                                     Arbiter: start / stop behaviour
                 ┌──────────── fast loop, 10–50 Hz ──────────┐       │
 frame ───────►  │ MediaPipe faces/hands, small detector     │ ──► behaviour tick ──► motion loop ──► set_target
                 └───────────────────────────────────────────┘
```

- **Slow loop.** The LLM picks from a short, fixed list of tools. It never outputs raw joint values. Vision goes through `perception/vision.py` (`ask_image_hf` or `ask_image_chat`).
- **Fast loop.** Behaviours (`FaceTrack`, `PointFollow`, `Search`, `LookAt`, `Mimic`) turn perception results into head targets every tick. These are the parts a learned policy could replace later.
- **Action tools the LLM sees** (names follow the behaviours in [04 §3](../architecture/04-motion-and-attention.md); the final signatures belong in [05](../architecture/05-agent-tools-memory.md)):

| Tool | Starts | Returns |
|---|---|---|
| `search(query, coco_hint?)` | `Search` | `{found, where_text, confidence}` |
| `look_at(label)` | `LookAt(label)` | `{ok}` |
| `look_at_pointed()` | `PointFollow` | `{description}` |
| `mimic(on)` | `Mimic` | `{ok}` |
| `camera(question)` | — (one frame) | `{text}` (exists: `agent/tools/camera.py`) |
| `emote(name)` | an expressive move offset | `{ok}` |

Use cases U1, U2, U4, U5, U8, U10 and U12 need only this. U3, U6, U7 and U9 also work this way using rules. §4 is for making them better.

---

## 4. When to train a real VLA policy

### 4.1 Where a VLM fails and a VLA does not

A VLM in the slow loop fails a task when one or more of these is true:

| Test | Why the VLM fails | What a VLA does instead |
|---|---|---|
| **Fast**: the task needs a reaction in < 200 ms | One VLM decision takes 1–3 s (measured: 1.1–1.5 s for one image, [03 §4](../architecture/03-perception.md)) | Outputs an action every 50–100 ms |
| **Continuous**: the output is a smooth trajectory, not a choice | Tool calls pick from a list; raw joint values from an LLM are jerky and unreliable | Outputs action chunks (a short trajectory) directly |
| **Closed loop**: each move changes the next frame | Every correction is another 1–3 s round trip | Corrects every frame |
| **Wordless style**: you can show it but not describe it ("nod like a good listener") | A prompt can't specify timing and feel | Learns the style from demonstrations |

Be honest about the alternative. If the target is a **fixed class** and the "how" is simple, a detector plus a rule (PID, 1-euro filter) beats both, and that is what `FaceTrack` and `Mimic` already do. A VLA earns its place when the task is fast **and** either open-vocabulary (any object named by voice) **or** a learned style.

### 4.2 VLA-only use cases

| # | Use case | Why not a VLM | Cheaper alternative to try first | Where the training data comes from |
|---|---|---|---|---|
| V1 | **Active listening**: nods, head tilts and antenna twitches timed to the speaker's pauses and tone | Backchannels must land within ~200 ms of a pause; style is wordless | VAD-pause rule + random nod | **Automatic**: film people listening, MediaPipe gives their head pose as labels |
| V2 | **Dance with a partner**: move in time with the music *and* with the person dancing in front | Beat sync needs < 100 ms; style is wordless | Beat tracker + fixed dance moves (`robot/dances.py`) | **Automatic**: partner's head pose from MediaPipe, plus the audio |
| V3 | **Video-call camera operator**: keep a moving person framed like a camera operator would (lead room, smooth pans, anticipates where they will go) | Closed loop at 10+ Hz; "good framing" is a style | `FaceTrack` with a centre offset | Teleop: you steer the head during recorded calls |
| V4 | **Track anything by name**: "keep watching the balloon / the cat / my toddler" while it moves fast | Open-vocabulary target and fast motion; the COCO detector doesn't know "balloon", the VLM is too slow | VLM finds it once, then an open-vocabulary tracker (e.g. SAM 2) follows it **[verify]** CPU cost | Teleop + language labels |
| V5 | **Gaze following (joint attention)**: glance where the person is looking, then back to them, with natural timing | Needs the person's gaze ray per frame and quick, short glances | Head-pose ray from the face landmarker + `look_at_image` | Teleop, or recordings of two people talking |
| V6 | **Reflex games**: peekaboo (react at the instant of the reveal), follow a ball rolled across the desk, chase a laser dot | Reaction time < 150 ms | Detector + PID for the ball / dot | Teleop while playing |
| V7 | **Startle and recover**: flinch back from a fast approaching hand, then lean in curiously | Must react within ~100 ms; the recovery is a style | Optical-flow threshold + a canned flinch move | Teleop, plus an animator (V8) |
| V8 | **Character from an animator**: an animator puppeteers Reachy through many situations; the policy reproduces that character's motion style | Personality in motion can't be written as a prompt | Hand-keyframed `moves.py` | Teleop by the animator |
| V9 | **Conducting**: you wave your hand like a conductor, Reachy's head and antennas follow your tempo and size of gesture | Continuous tempo tracking | Hand-landmark speed → tempo rule | **Automatic**: hand landmarks map to target motion |
| V10 | **Fine viewpoint for small detail**: micro-adjust the head to beat glare or get a tiny label readable, then hand the frame to the VLM | Many small closed-loop corrections; the VLM would need one 1–3 s round trip each | Scan a 3×3 grid of small offsets, keep the sharpest frame | Self-supervised: the reward is "the VLM / OCR can read it" |

Three patterns stand out:
- **V1, V2, V9 label themselves.** MediaPipe on recorded humans produces the action labels, like U9 imitation. Start here.
- **V3–V8 need teleoperation.** Build one teleop recorder (phone or mouse → head target, logged with frames in LeRobot format) and reuse it for all of them.
- **V4 and V10 are hybrids.** The VLM decides *what* (which object, is it readable), the policy does *how* (track it, adjust the view). This is the System 2 / System 1 split from §3 with a learned System 1.

### 4.3 Training plan

Full end-to-end plan for object tracking (design, data, training, sim test, deployment): [2026-11-18-vla-tracking-pipeline.md](2026-11-18-vla-tracking-pipeline.md).

Train only when a hand-written behaviour is visibly worse. Start with U9 (imitation) and U3 (speaker gaze), then V1 (active listening), since it reuses the same automatic labelling.

| | U9 Imitation (do first) | U3 Speaker gaze |
|---|---|---|
| Observation | frame (+ instruction "mirror me") | frame + DoA angle + VAD flag |
| Action | head yaw/pitch/roll targets, 10 Hz | head target + nod/antenna offsets, 10 Hz |
| Labels | **automatic**: the MediaPipe head pose of the owner, mirrored (the current `Mimic` rule) | teleoperation: you steer the head with a phone or mouse while people talk |
| Data needed (estimate **[verify]**) | 50–100 episodes of 30 s | 100+ episodes, several speakers |
| Baseline to beat | `Mimic` rule (lag ≤ 300 ms, F6) | `FaceTrack` + DoA bias (≤ 2 direction changes/s) |

Recipe:
1. Record episodes in LeRobot dataset format: camera frames plus the action stream that the motion loop actually sent.
2. Fine-tune **SmolVLA** (~450M parameters) on a GPU machine.
3. Serve it from that machine. The app sends frames and gets head targets back. The policy runs as a **behaviour**: its output is a base target that passes through the arbiter, smoothing, limits and the watchdog like any other.
4. Keep the rule-based behaviour as the fallback. If policy replies stop arriving for 500 ms, the arbiter falls back to the rule (the same "stale-is-absent" rule as WorldState).

Not worth it: training a VLA for U1/U2/U4. Those are search-and-answer tasks where a VLM plus a scan pattern is already reliable, and a policy would need far more data.

---

## 5. Build order

| Step | Ships | Depends on | Acceptance |
|---|---|---|---|
| 1 | U1 find + U2 object memory | detector in `perception/`, `Search` behaviour, sightings writer in `memory/` | "Where's my phone?" → looks at it and says where within 15 s (F5). Asked again after it is hidden → answers from memory with the time last seen |
| 2 | U4 look at this + U7 gestures | hand landmarker, `PointFollow` | Point + "what is this?" → frames the object and answers |
| 3 | U3 speaker gaze (rules) | DoA, `FaceTrack` | Two people taking turns → head follows the speaker, ≤ 2 direction changes/s |
| 4 | U5, U6, U8, U12 | steps 1–3 plus a scheduler | One scripted scenario each |
| 5 | U9 imitation as a trained policy | GPU box, LeRobot recording | Policy lag and smoothness beat the `Mimic` rule on the same clip |

Each step gets its usage doc in `docs/usage/<feature>.md` when it ships.

---

## 6. Privacy and safety

- The camera and microphones are in someone's home. Keep frames local by default. Send a frame to the cloud only for a tool call that needs it, downscaled, and never store it.
- U11 (face ID) stores biometric data. Owner opt-in only, per-person delete ("forget me"), never synced anywhere.
- U6 and U12 watch without being asked. Show it (antenna pose or LED) while a watcher runs, and let one voice command turn it off.
- A learned policy never bypasses the motion loop's limits, rate caps or watchdog (§1).

---

## 7. Open questions

- Is the HF realtime vision model good enough for `confirm`/`locate`, or does U1 need the OpenAI plan in [03 §4](../architecture/03-perception.md)?
- Real CPU cost of running the detector and hand landmarker together on the demo laptop (the Step 0 spike).
- Exact SDK joint ranges and antenna limits **[verify]**.
- Where the GPU box for §4 lives, and the Wi-Fi latency from it to the robot.
