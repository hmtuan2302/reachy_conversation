# 01 — Overview

## 1. Architecture principles

1. **One process, one owner per resource.** One component owns each hardware resource: mic, speaker, camera, head. Everyone else goes through it. This prevents head "fights" and USB/WebRTC contention.
2. **Latest-value world, not message soup.** Perception writes a single `WorldState`, and behaviours read snapshots. Events are used only for things that *happen* (utterance ended, loud bang, tool finished).
3. **Fail small.** Every worker runs under a supervisor. A crash in one model degrades only that feature, and the conversation keeps going (requirement: "Resilience").
4. **Policy in code, not in the prompt.** Owner-only permissions, joint limits and timeouts are enforced by code. The LLM is told about them but is never trusted to apply them.
5. **Pay for compute only when used.** CPU models run on demand: a behaviour or tool turns them on and off.
6. **Lean by default.** Stdlib + SDK + a few model runtimes. No web framework, no DI container, no message broker.

## 2. System context

```
                 ┌──────────────────────────── Demo laptop (WSL2, CPU-only) ───────────────────────────┐
  Owner / guests │                                                                                      │
  (voice, face,  │   reachy_demo  (single Python 3.12 process)                                         │
   hands)        │                                                                                      │
      │          │   ┌─────────┐  ┌───────────┐  ┌────────────┐  ┌───────────┐  ┌─────────────────┐    │
      ▼          │   │ Robot   │  │ Audio     │  │ Voice      │  │ Agent     │  │ Operator console│    │
 ┌──────────┐ Wi‑Fi │   │ adapter │◄►│ pipeline  │◄►│ session    │◄►│ (tools,   │  │ (keys, status)  │    │
 │ Reachy   │◄─────┼──►│ (SDK)   │  └───────────┘  └────────────┘  │ memory)   │  └─────────────────┘    │
 │ Mini     │ WebRTC│   │         │  ┌───────────┐  ┌────────────┐  └───────────┘                         │
 │ Wireless │ + REST│   │         │◄►│ Perception│─►│ WorldState │◄─ Attention arbiter ─► Motion loop    │
 └──────────┘      │   └─────────┘  └───────────┘  └────────────┘                                        │
                 └─────────────────────────────┬──────────────────────────────────────────────────────┘
                                               │ HTTPS / WSS
                                   ┌───────────┴───────────┐
                                   │ OpenAI Realtime API   │ (voice ↔ voice, tool calls)
                                   │ OpenAI vision model   │ (find / identify / read)
                                   └───────────────────────┘
```

External dependencies:
- **Reachy Mini daemon** on the robot: `reachy_mini` SDK ≥ 1.13, `ReachyMini(host=<ip>, connection_mode="network")`, WebRTC media, REST `GET /api/state/doa` **[verify]**.
- **OpenAI Realtime API** (WebSocket) **[req]**, and an **OpenAI vision model** via HTTPS **[req]**. Model names live in config.

## 3. Runtime model

The work is a mix of hard real-time motion, blocking SDK I/O, CPU-bound inference and network async. So:

| Worker | Kind | Rate | Owns | Notes |
|---|---|---|---|---|
| `main` asyncio loop | event loop | — | voice session, agent, tool execution, event bus | All decisions happen here. |
| `audio_in` | thread | 32 ms frames | mic | Pulls `get_audio_sample()` (16 kHz mono), runs VAD + onset detector inline (cheap). |
| `audio_out` | thread | 20 ms chunks | speaker | Paced `push_audio_sample()`, supports instant flush (barge-in). |
| `perception` | thread | 5–10 Hz | camera | `get_frame()` → enabled models → `WorldState`. |
| `motion` | thread | 50 Hz | head, antennas, body yaw | Only caller of `set_target()`. |
| `doa_poller` | thread | 10 Hz | DoA | Rolling buffer of `(t, angle, speech)`. |
| `speaker_id` | `ThreadPoolExecutor(1)` | per utterance | — | Embedding + cosine. Runs off the loop so it never blocks audio. |

Threads hand data to the loop via `loop.call_soon_threadsafe(queue.put_nowait, event)`. Threads read shared state through lock-protected snapshot copies. CPU-heavy ONNX calls release the GIL, so threads are enough; no multiprocessing **[arch]**. If profiling in Step 0 shows GIL pressure on audio, perception moves to a subprocess behind the same interface (one file change).

## 4. Module layout

```
reachy_demo/
  __main__.py            # parse CLI → app.run()
  app.py                 # composition root: build components, wire, start supervisor
  config.py              # dataclass Config from .env / env vars, validated at startup
  bus.py                 # tiny typed pub/sub on asyncio (≈50 lines)
  world.py               # WorldState dataclasses + lock + snapshot()
  supervisor.py          # start/restart workers, health flags
  robot/
    adapter.py           # the ONLY module importing reachy_mini; wraps media + motion + REST
  audio/
    capture.py           # audio_in thread: frames, VAD, onset → events
    vad.py               # Silero VAD (onnxruntime)
    sound_events.py      # loud-onset detector + DoA fusion → LoudSound event
    speaker_gate.py      # enrollment, embedding, verify → owner|guest
    playback.py          # audio_out thread, resample 24k→16k, flush
  voice/
    realtime.py          # OpenAI Realtime WS client (raw protocol, reconnect)
    turns.py             # TurnManager: utterance → gate → commit/response, barge-in
  perception/
    loop.py              # perception thread, model scheduling
    faces.py             # face landmarks, head pose, expression
    hands.py             # hand landmarks, pointing ray
    objects.py           # object detector
    vision_client.py     # cloud vision calls (async)
  attention/
    arbiter.py           # priority + preemption
    behaviors.py         # Idle, FaceTrack, Mimic, PointFollow, Investigate, Command(Search/LookAt)
    motion.py            # motion thread, smoothing, limits, layered offsets
    moves.py             # expressive moves (antenna wiggle, nod, shake, emotion poses)
  agent/
    tools.py             # Tool base + registry + permission check
    tools_*.py           # one file per group: motion, vision, memory, social
    prompts.py           # system prompt builder (persona, memory summary, rules)
  memory/
    store.py             # SQLite store: facts, object sightings, people
  ops/
    logging.py           # JSONL structured logs + per-turn trace id
    metrics.py           # latency histograms, counters → periodic log line
    console.py           # operator keys (stdin), status line
tests/
  unit/  replay/  fixtures/
models/                  # downloaded ONNX/TFLite files (git-ignored), checksums in models.lock
data/                    # owner voiceprint, memory.db, logs (git-ignored)
```

Dependency rule: `robot/adapter.py` is the only SDK touchpoint. `voice/realtime.py` is the only OpenAI Realtime touchpoint. `perception/vision_client.py` is the only OpenAI vision touchpoint. Everything else depends on plain Python types. That lets tests swap fakes in.

## 5. Key contracts

```python
# world.py
@dataclass(frozen=True)
class Face:      t: float; box: Box; center_px: tuple[int,int]; head_ypr_deg: tuple[float,float,float]
                 expression: str; expr_conf: float; track_id: int
@dataclass(frozen=True)
class Hand:      t: float; fingertip_px: tuple[int,int]; ray_dir: tuple[float,float]; pointing: bool
@dataclass(frozen=True)
class Detection: t: float; label: str; conf: float; box: Box; head_pose: Pose  # pose at capture time
@dataclass(frozen=True)
class WorldSnapshot:
    t: float; faces: tuple[Face,...]; hands: tuple[Hand,...]; objects: tuple[Detection,...]
    speaking: Speaker | None          # "owner" | "guest" | None (last utterance)
    robot_speaking: bool; head_pose: Pose; frame_jpeg: bytes | None  # latest, on request only

# bus.py events (frozen dataclasses)
SpeechStarted(t)                    # local VAD onset
UtteranceEnded(t, pcm16k, dur_s)    # local VAD end
SpeakerResolved(utt_id, speaker, score)
LoudSound(t, angle_rad, confidence, rms_db)
ToolStarted(name, call_id) / ToolFinished(name, call_id, ok, ms)
BehaviorChanged(old, new, reason)
WorkerHealth(name, ok, detail)

# attention
class Behavior(Protocol):
    name: str; priority: int
    def tick(self, w: WorldSnapshot, now: float) -> HeadTarget | None   # None = done/yield
    def on_enter(self) -> None; def on_exit(self) -> None

# agent
class Tool(Protocol):
    name: str; description: str; schema: dict; owner_only: bool; timeout_s: float
    async def __call__(self, ctx: ToolContext, **args) -> dict          # {"ok":..., ...} | {"error":...}
```

## 6. Main data flows

**Conversational turn (owner command).** VAD end → `UtteranceEnded` → gate (~80 ms) → `owner` → commit audio + speaker tag → `response.create` → the model calls `find_object` → the permission check passes → the tool starts the Search behaviour, which preempts face tracking → detections + cloud confirm → the tool returns a location → the model speaks the answer → the Search behaviour ends → face tracking resumes. Full sequence in [02](02-voice-and-audio.md) §4.

**Reactive event (no LLM in the loop).** Onset detector + DoA → `LoudSound` → the arbiter starts Investigate (priority 2) → turn toward the angle → capture a frame → `vision_client.describe()` → inject a text item "[event] loud sound on the left; camera sees: a book on the floor" and `response.create` → the robot comments.

**Latency budget (F2, p90 target ≤ 1.5 s from end of user speech to first robot audio):**

| Stage | Budget |
|---|---|
| VAD end-of-speech hangover | 450 ms |
| Speaker gate | 100 ms |
| Commit + Realtime first audio delta | 600 ms |
| Wi‑Fi + playback buffer | 200 ms |
| Margin | 150 ms |
