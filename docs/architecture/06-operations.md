# 06 — Operations: config, deployment, resilience, observability, testing, security

## 1. Configuration

`config.py` loads `.env` (python-dotenv) into a frozen dataclass and validates at startup. Fail fast with a readable message.

| Key | Default | Notes |
|---|---|---|
| `ROBOT_HOST` | — (required) | IP of the Wireless. Use the IP, not mDNS, under WSL2. |
| `OPENAI_API_KEY` | — (required) | Never logged. |
| `REALTIME_MODEL`, `REALTIME_VOICE`, `VISION_MODEL` | set in `.env.example` | |
| `GATE_THRESHOLD` | from enrollment | overridden by the on-site calibration |
| `VAD_THRESHOLD`, `VAD_HANGOVER_MS` | 0.5, 450 | |
| `ONSET_DB` | 15 | calibration |
| `PERCEPTION_MAX_HZ` | 10 | |
| `LOG_DIR` | `data/logs` | |
| `RECORD_AUDIO` | false | rehearsals only (see §7) |

## 2. Deployment on the demo laptop (WSL2) **[req]**

- Python 3.12 venv managed with `uv`, a lockfile committed. Models are downloaded by `python -m reachy_demo fetch-models` with SHA-256 checks into `models/`.
- **Networking**: enable WSL2 **mirrored networking** (`.wslconfig`: `networkingMode=mirrored`) so WebRTC/UDP from the robot reaches WSL directly. Connect with `ReachyMini(host=ROBOT_HOST, connection_mode="network")`. Fallback documented in Step 0: run the same code on native Windows Python **[verify]**.
- **Venue network**: a dedicated router/hotspot with the robot and laptop only, 5 GHz.
- **GStreamer/WebRTC** deps come from the SDK's install instructions. `python -m reachy_demo doctor` checks they're present.

## 3. Startup sequence and commands

```
python -m reachy_demo doctor       # deps, models, robot reachable, camera/mic/speaker/DoA, OpenAI reachable, latency report
python -m reachy_demo enroll       # owner voiceprint
python -m reachy_demo calibrate    # ≤10 min on site: ambient noise, onset dB, gate threshold (10 owner + 10 guest phrases)
python -m reachy_demo run          # the show
```

`run` startup order: config → logging → robot connect (retry 3×) → motion thread (neutral pose) → audio threads → perception → memory → Realtime connect → ready chime + idle. Readiness is shown on the console status line.

## 4. Fault handling

| Failure | Detection | Response |
|---|---|---|
| Worker thread exception | supervisor wraps `run()` | Log, mark unhealthy, restart with backoff (1, 2, 5 s, max 5 restarts per 5 min, then leave disabled). The dependent feature reports "unavailable". |
| Perception model fails to load | at startup | Feature disabled; the tool returns `{"error":"unavailable"}`; the robot says so. |
| Realtime WS drop | close/error event | Reconnect ([02](02-voice-and-audio.md) §7); local "one moment" WAV. |
| Vision API timeout | `wait_for` | Spoken fallback, no retry storm. |
| Robot connection lost | SDK errors / no frames > 2 s | Pause behaviours, keep the conversation, retry connect every 2 s, announce "I lost my body for a moment". |
| Tool hangs | per-tool timeout | Cancel the task, cancel the behaviour, return an error. |
| Fatal crash in main | top-level `try/finally` | Neutral pose → disable motors → flush logs. |

Behaviours and tools never call the SDK directly. Every SDK call is in `robot/adapter.py`, with timeouts and a typed `RobotUnavailable` error.

## 5. Observability

- **Logs**: JSONL, one line per event, to `data/logs/run-<ts>.jsonl` + pretty console. Each user turn gets a `trace_id` that follows gate → commit → tool calls → vision calls → response.
- **Metrics** (in-process, logged every 30 s and summarised at exit): `turn_latency_ms` (end of speech → first audio, p50/p90), `gate_score` histogram owner/guest, `tool_ms` per tool, `perception_hz` per model, `motion_overruns`, `barge_ins`, `self_interruptions`, `worker_restarts`. These map one-to-one to the acceptance criteria, so a rehearsal run produces its own scorecard.
- **Transcripts**: user/robot text with speaker tags (from input transcription) in the log, for debugging F9.

## 6. Operator console (stdin, backstage) **[req: operator keys]**

| Key | Action |
|---|---|
| `o` | Force the next utterance as owner |
| `m` | Mute/unmute the mic to the model |
| `s` | Stop the current behaviour → FaceTrack |
| `r` | Reconnect Realtime |
| `n` | Neutral pose |
| `q` | Graceful shutdown (sleep pose) |

The status line shows connection states, the behaviour stack, last gate score, the last turn latency and worker health.

## 7. Testing strategy

| Layer | What | How |
|---|---|---|
| Unit | arbiter (priorities, stack, hysteresis), permission check, turn state machine, onset detector, DoA→yaw mapping, memory queries, motion limits | `pytest`, pure Python with fakes; no robot, no network |
| Contract | `realtime.py` against recorded WS event transcripts | replay JSON fixtures, assert the emitted client events |
| Replay | audio pipeline + gate on recorded WAVs (owner, 3 guests, bangs, robot echo); perception on recorded video clips | `tests/replay/`, deterministic, measures accuracy like the acceptance tests |
| Simulation | motion + behaviours with the SDK simulator (`use_sim=True` / sim daemon) **[verify]** | smoke run of the full app minus audio |
| Rehearsal | the §7 demo script ×10 in the room | metrics scorecard from the logs = acceptance report |

`RECORD_AUDIO=true` during rehearsals saves utterances (with consent) to grow the replay fixtures for gate tuning.

## 8. Security and privacy

- The API key lives only in `.env` (git-ignored) and is redacted from logs.
- The owner voiceprint (`data/owner_voiceprint.npz`) and memory DB stay local; no raw audio is stored unless `RECORD_AUDIO=true`.
- Guest data: no audio is stored, and names are not persisted.
- Images go to OpenAI only on cloud-vision tool calls (not streamed continuously).
- The owner-only gate is a demo-grade control. A voice recording of the owner could pass it, which is acceptable for a demo and documented as a known limit.

## 9. Build milestones (map to the requirements' order)

| M | Delivers | Exit check |
|---|---|---|
| 0 | Step 0 spike: adapter + `doctor` on WSL2 | A/V/DoA working, latency numbers recorded |
| 1 | Motion loop, Idle, Realtime + turn manager (gate pass-through), playback, barge-in, `introduce_self` | F1, F2 criteria |
| 2 | Faces + expression + `express` + context notes | F3 |
| 3 | Arbiter + FaceTrack + objects + `look_at` | F4, oscillation criterion |
| 4 | `find_object` + vision client + sightings | F5 |
| 5 | Mimic | F6 |
| 6 | Onset + DoA + Investigate | F7 |
| 7 | Memory store + prompts summary + recall/forget | F8 |
| 8 | Gate on: enroll/calibrate + permission enforcement | F9 |
| 9 | Hands + `look_at_pointed` | F10 |
| 10 | Hardening: supervisor drills, 10 rehearsals | Whole-show criteria |
