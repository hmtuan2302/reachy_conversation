# 04 — Motion and attention (F4, F5, F6, F7, F10)

## 1. Layers

```
 behaviours (one active) ──► Arbiter ──► base HeadTarget ─┐
 expressive moves (0..n) ──────────────► offsets ─────────┼─► Motion loop 50 Hz ─► robot.set_target(head, antennas, body_yaw)
 speech wobble (while speaking) ───────► offset ──────────┘        smoothing · limits · rate caps
```

- **Base target**: exactly one behaviour owns it **[req]**.
- **Offsets**: additive small poses (nod, tilt, shake, antenna moods, speech wobble). Each is time-bounded and amplitude-capped. Offsets never change where the robot is looking by more than ±10°.

## 2. Arbiter

Priority table **[req]**:

| Prio | Behaviour | Started by | Ends when |
|---|---|---|---|
| 1 | `Command` (`Search`, `LookAt`, `GoToPose`) | owner-only tool | finished / cancelled / timeout |
| 2 | `Investigate` | `LoudSound` event | camera check done (≤ 6 s) |
| 3 | `PointFollow` | owner points ≥ 300 ms and says "this/look", or the `look_at_pointed` tool | target reached + 3 s, or the hand disappears |
| 4 | `Mimic` | owner-only tool `mimic_me(enabled)` | `mimic_me(false)`, spoken stop, or 5 min cap (face lost > 1 s: look back at the user and wait) |
| 5 | `FaceTrack` | default when faces are present | no face for 3 s |
| 6 | `Idle` | always (fallback) | never |

Rules **[arch]**:
- Decisions run on the main loop at 10 Hz plus on every relevant event. The arbiter keeps a **stack**: a higher-priority behaviour pushes on top, and when it ends it pops, so the interrupted one resumes. `Command` behaviours that are preempted (e.g. a bang during a search) resume their own progress (search continues at the next pose).
- **Hysteresis**: a behaviour must want control for ≥ 200 ms before preempting (except prio 1–2), and a FaceTrack target switch needs the new face to be "better" for ≥ 500 ms. This prevents oscillation (acceptance: ≤ 2 direction changes/s).
- Any `Command` started while another `Command` runs cancels the old one (the owner changed their mind).
- `BehaviorChanged` events are logged; the operator console shows the current stack.

## 3. Behaviours

| Behaviour | Tick logic |
|---|---|
| `Idle` | Slow breathing (z ± 3 mm, 0.2 Hz), a random glance every 6–12 s, antennas relaxed. |
| `FaceTrack` | Target = the speaking/largest face centre → `look_at_image` pose (computed, not executed: `perform_movement=False`). The target is passed through a 1-euro filter. |
| `Mimic` | User head yaw/pitch (relative to their line of sight) and roll, from YuNet box + landmarker matrix on a crop; mirrored (yaw negated), scale 1.0 (user request 2026-11-17), plus lean as head x. Clamped to ±35° yaw / ±22° pitch from facing the user (camera view), ±25° roll, ±20 mm x, then the motion safety box. 1-euro filtered, critically damped spring in the motion loop. Lag budget ≤ 300 ms (F6). Built in `attention/mimic.py`, see [usage/mimic.md](../usage/mimic.md). |
| `PointFollow` | Target = pointed point (see [03](03-perception.md) §2) in the capture pose. |
| `Investigate` (deferred; F7 is on request for now, see [02 §6](02-voice-and-audio.md#6-sound-events-f7)) | Turn toward the DoA yaw (head first, then body yaw if > 45°), wait 500 ms, grab a frame → `describe()` → inject the event into the conversation. Low confidence → a 3-pose sweep. |
| `Search` | Drives the scan in [03](03-perception.md) §5. Reports progress to the tool via a future. |
| `LookAt(label)` | Look at the best detection of the label; re-acquires each tick. This is the "track object" of F4. |

## 4. Motion loop and safety

- The **single caller** of `set_target()`. It runs at 50 Hz in its own thread with `time.perf_counter` pacing. A missed deadline is logged, not compensated.
- **Smoothing**: critically damped second-order filter toward the target. Max angular speed 120°/s (head), 60°/s (body yaw).
- **Limits** **[arch]**: soft limits inside the SDK's safe ranges (yaw ±60°, pitch −30°/+25°, roll ±20°, body yaw ±160°), applied after offsets are summed. The SDK's own kinematic limits stay as a backstop **[verify exact SDK ranges]**.
- **Watchdog**: if no base target is updated for 1 s (an arbiter crash), ease to the neutral pose.
- **Startup/shutdown**: on start, `goto_target(neutral, duration=1.5)` and enable motors. On exit (including a crash in `main`), go to the neutral pose, then disable motors (the "go to sleep" pose). This is wired via `try/finally` and a SIGINT handler.
- `moves.py` defines expressive moves as small keyframe tracks (offset + antennas) with names the LLM can trigger: `happy`, `sad`, `curious`, `surprised`, `no`, `yes`, `thinking`, `greet`.
