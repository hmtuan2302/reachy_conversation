# 02 — Voice and audio (F2, F7, F9)

## 1. Pipeline

```
 robot mic (16 kHz mono, WebRTC)
        │ get_audio_sample()
        ▼
 audio_in thread ── 32 ms frames ──┬─► Silero VAD ──► SpeechStarted / UtteranceEnded(pcm)
                                   ├─► onset detector ─► (with DoA buffer) LoudSound
                                   └─► ring buffer (last 3 s, for pre-roll)
                                                     │
                     main loop: TurnManager ◄────────┘
                       │  1. speaker_gate.verify(pcm) → owner|guest     (executor)
                       │  2. resample 16k→24k, input_audio_buffer.append (chunks)
                       │  3. input_audio_buffer.commit
                       │  4. conversation.item.create  [speaker: owner|guest]
                       │  5. response.create
                       ▼
                 OpenAI Realtime WS ── response.output_audio.delta ──► playback (24k→16k) ─► robot speaker
                                    ── function_call events ─────────► agent tool executor
```

**Server turn detection is off** **[req]**: the session is configured with no turn detection, so the app decides when a turn ends. The audio of an utterance is appended only after it ends **[arch]**. That costs ~0 latency in practice: the buffer append of 1–5 s of audio over WSS takes tens of ms. It also guarantees guest/owner tagging comes *before* the model sees the audio.

Exact Realtime event names follow the current GA protocol and are isolated in `voice/realtime.py` **[verify]** against the docs at build time.

## 2. Turn manager state machine

```
          ┌──────────── SpeechStarted ─────────────┐
          ▼                                        │
   ┌────────────┐ UtteranceEnded  ┌──────────┐ commit+response.create ┌────────────┐
   │ LISTENING  │───────────────►│ GATING   │───────────────────────►│ RESPONDING │
   └────────────┘   (dur ≥0.4 s)  └──────────┘                        └────────────┘
          ▲   short/noise: drop        │ gate error → tag "unknown"=guest   │  response.done
          │                            ▼                                    │
          └──────────────────────────────────────────────────────────────── ┘
   RESPONDING + SpeechStarted (barge-in) → playback.flush(); response.cancel;
       conversation.item.truncate(audio_end_ms = ms actually played) → LISTENING
```

Rules:
- Utterances < 0.4 s are dropped as noise **[arch]**. Utterances 0.4–1.0 s are sent, but the gate result counts as `guest` unless score ≥ the strict threshold (short audio gives weak embeddings), so short commands from the owner may need repeating (R13).
- Only one response is in flight. A new utterance that arrives during RESPONDING (which is also a barge-in) cancels it first.
- During a tool call the robot says a filler line ("Let me look…"). The model is prompted to do this, and a tool that runs > 1 s also triggers `moves.thinking()` (an antenna motion) as a visual cue.

## 3. Speaker gate (F9)

- **Model** **[arch]**: a sherpa-onnx speaker-embedding model (e.g. 3D-Speaker ERes2Net or WeSpeaker ResNet34, ONNX, ~25 MB, CPU ~30–80 ms per 3 s clip **[verify]**).
- **Enrollment**: `python -m reachy_demo enroll`. It records 5 phrases × ~5 s through the *robot's* mic in the venue, stores the mean L2-normalised embedding plus per-clip embeddings in `data/owner_voiceprint.npz`, and computes the owner's own score spread.
- **Verify**: cosine(mean, utt). `owner` if ≥ `GATE_THRESHOLD` (default from enrollment: min owner self-score − margin). Calibrated in the on-site step with 10 owner + 10 guest phrases.
- **Output** goes into the turn (speaker tag) and into `WorldState.speaking`. Permission enforcement happens in the tool layer ([05](05-agent-tools-memory.md) §2), not here. The gate never drops audio, because guests can chat **[req]**.
- **Operator override**: key `o` forces the next utterance to `owner` (R13).

## 4. Sequence — "Find my phone" (owner)

```
Owner      audio_in   TurnMgr    Gate     Realtime     Agent      Arbiter/Search    Vision
  │ speaks ─►│ VAD end ─►│ verify ──►│                                                  
  │          │           │◄── owner ─┤                                                  
  │          │           │ append+commit+[speaker: owner]+response.create ─►│          
  │          │           │           │       │◄─ "Let me look" audio + call find_object(query="my phone")
  │◄─────────────────────────────── playback "Let me look…"                           
  │          │           │           │       │ tool ─────────►│ check owner_only ✔ 
  │          │           │           │       │                │ start Search ──►│ scan poses, detect
  │          │           │           │       │                │                 │ candidates ──────►│ confirm
  │          │           │           │       │                │◄── best: (pose, box, "on desk, left")
  │          │           │           │       │◄─ function_call_output {found, where} + response.create
  │◄────────────────────────── "It's on the desk, to your left" (head pointing at it)
  │          │           │           │       │  memory.sighting("phone", where, t)        
```

## 5. Playback, barge-in and echo

- `playback.py` keeps a ≤ 200 ms queue and pushes 20 ms chunks paced on a monotonic clock. `flush()` empties the queue immediately and records `played_ms` for truncation.
- **Barge-in**: `SpeechStarted` while `robot_speaking` → flush within one frame (≤ 50 ms) → F2 target ≤ 0.5 s is met with margin.
- **Echo (R7)**: the speaker sits close to the mics. Mitigations, in order **[arch]**:
  1. While the robot speaks, raise the VAD onset threshold (e.g. 0.5 → 0.8) and require ≥ 150 ms of continuous speech to call barge-in.
  2. Correlate mic energy with the playback envelope over the last 200 ms. If the mic tracks the playback, ignore it.
  3. If the robot's mic array provides hardware echo cancellation (ReSpeaker AEC), enable it **[verify]**.

  The acceptance test counts self-interruptions per 10 min (target 0).

## 6. Sound events (F7)

> **Decision 2026-11-17 [arch]:** F7 ships first **on request**, not as the reactive `Investigate`. The app stores the latest loud sound and the latest voice direction ([audio/sound_events.py](../../src/reachy_demo/audio/sound_events.py)), and the agent tool `look_toward_sound` turns when the user asks ("what happened?", "turn to me"). Head yaw stays ±45° from the body and body yaw ±120°. Past that the robot says where the sound was and asks the user to check. The automatic `Investigate` below waits for the arbiter. Usage: [sound-direction.md](../usage/sound-direction.md).

- **Onset detector**: short-term RMS (32 ms) vs. a slow ambient estimate (EMA over 5 s). Trigger when `rms_db − ambient_db ≥ ONSET_DB` (default 15 dB, calibrated on site), rise time < 50 ms, and not during robot speech. Refractory period 3 s.
- **Direction**: median of DoA readings within ±150 ms of the onset from the 10 Hz `doa_poller` buffer (`GET /api/state/doa` or `media.get_DoA()`). Map 0 = left, π/2 = front, π = right → target yaw `= (π/2 − angle)` clamped to ±60° head yaw plus body yaw for the rest **[verify sign convention]**.
- **Confidence**: low if the DoA spread > 30°. Then Investigate does a short camera sweep instead of a single turn.
- Speech onsets are excluded: if VAD probability > 0.5 at the onset, it is a voice, not a "something falls" event.
- Emits `LoudSound` → the arbiter ([04](04-motion-and-attention.md)).

## 7. Realtime session config (summary)

| Setting | Value |
|---|---|
| Model | `REALTIME_MODEL` (config) |
| Voice | `REALTIME_VOICE` (config) |
| Input/output audio | PCM16 24 kHz mono |
| Turn detection | none (manual) |
| Input transcription | on (for logs, memory extraction and the operator display) |
| Tools | from registry, all tools (permission is enforced at call time) |
| Instructions | from `agent/prompts.py` (persona + rules + memory summary) |

**Reconnect**: on WS close or error → exponential backoff (0.5, 1, 2, 4 s, max 8 s). Re-send session config + the last N=10 transcript lines as context. Say "one moment" locally from a pre-rendered WAV so there is no dead air (R6).
