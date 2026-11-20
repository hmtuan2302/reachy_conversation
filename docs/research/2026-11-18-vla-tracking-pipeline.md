# VLA object tracking on Reachy Mini: design, data, training, sim test, deployment

Date: 2026-11-18
Scope: the end-to-end plan for a language-conditioned object-tracking policy ("keep looking at the white mug") on Reachy Mini, from design through data collection, training, simulation testing and real deployment, in this repo (`reachy_demo`).
Related: use cases in [2026-11-18-vla-use-cases.md](2026-11-18-vla-use-cases.md) (U1, V4), perception in [03](../architecture/03-perception.md), motion in [04](../architecture/04-motion-and-attention.md).

Conventions: **[verify]** = an assumption to confirm in the Step 0 spikes (§5). Numbers without **[verify]** are design targets, not measurements.

---

## 0. TL;DR

1. **Task:** the user names an object by voice; the head keeps it centred while it moves. The policy sees the camera frame, the current head pose and the instruction, and outputs where to look next.
2. **Model:** SmolVLA (~450M parameters) fine-tuned with LeRobot from `lerobot/smolvla_base`. Train on a rented GPU or Colab; serve from a GPU machine on the same LAN. The demo laptop is CPU-only.
3. **Data:** mostly real recordings, labelled automatically offline (open-vocabulary detector + SAM 2 → geometry → actions). Extra data from warping real frames; simulation only for rare cases.
4. **Testing:** replay on held-out real episodes is the accuracy gate. MuJoCo simulation and unit tests verify safety; real images verify accuracy (§11).
5. **Deployment:** the policy is one more **behaviour** in the arbiter. Its output passes through the motion loop's limits, smoothing and watchdog. If the policy server is slow or down, a detector + PID baseline takes over.
6. **Rule:** the VLA ships only if it beats the baseline on the metrics in §4.

---

## 1. Goal and scope

**In scope**
- Gaze tracking (yaw, pitch) of one named object, open vocabulary ("the blue tape roll", not only COCO classes). The policy says where the camera should look; a fixed rule splits yaw between head and body (§2.5), so objects can be followed around the robot.
- Saying "I can't see it" when the object isn't in view, so the agent can start a search.
- Running live on the real robot at ≥ 10 Hz with ≤ 200 ms end-to-end latency.

**Out of scope (for v1)**
- Body yaw **as a policy output** (the body moves, but by the §2.5 rule, not by the policy). Head translation (x/y/z) and antennas.
- Tracking several objects at once.
- Searching for an object that isn't in view: that stays the `Search` behaviour ([03 §5](../architecture/03-perception.md)).
- On-robot inference.

---

## 2. System design

### 2.1 Components and where they run

```
 Reachy Mini (Wi-Fi)          Laptop (CPU, WSL2)                          GPU box (LAN)
 ───────────────────          ────────────────────────────────            ──────────────────────
 camera ──WebRTC──────────►   adapter.grab_jpeg()                         
                              │                                           
                              ▼                                           
                              TrackBehaviour (arbiter, prio 1) ──ws──►    policy_server.py
                              │   sends {jpeg, state, task, t}            SmolVLA checkpoint
                              │   ◄── {actions[], t, infer_ms} ──────     
                              ▼                                           
                              base gaze target                            
                              │                                           
 motors ◄── set_target ────── Motion loop 50 Hz (head/body split §2.5, limits, smoothing, watchdog)
                              ▲
                              BaselineTracker (detector + PID) ── fallback when the policy is stale
```

- **The motion loop stays the only caller of `set_target()`** ([04 §4](../architecture/04-motion-and-attention.md)). The policy never talks to the SDK.
- **`TrackBehaviour`** is a new `Command` behaviour (priority 1), started by a `track(object)` tool and stopped by `stop`, a new command, or a timeout.
- **`BaselineTracker`** is the detector + PID tracker. It's the fallback at runtime and the yardstick in evaluation.

### 2.2 New code

| Path | Purpose |
|---|---|
| `src/reachy_demo/vla/spec.py` | Observation/action names, shapes, rates, limits. One source of truth for recorder, converter, server and client |
| `src/reachy_demo/vla/geometry.py` | Calibration loading, undistortion, pixel ↔ angle conversion |
| `src/reachy_demo/vla/client.py` | WebSocket client: sends observations, returns the latest action chunk with its age |
| `src/reachy_demo/attention/track.py` | `TrackBehaviour` + `BaselineTracker` |
| `src/reachy_demo/agent/tools/track.py` | `track(object)` / `stop_tracking()` tools |
| `scripts/track_calibrate.py` | Camera intrinsics from checkerboard images |
| `scripts/track_record.py` | Robot → raw episode folders |
| `scripts/track_label.py` | Raw episodes → `boxes.jsonl` (GPU) |
| `scripts/track_to_lerobot.py` | Raw + boxes + calibration → LeRobot dataset |
| `scripts/track_eval.py` | Replay, sim and real evaluation; prints the §4 metrics |
| `scripts/policy_server.py` | Runs on the GPU box; loads a checkpoint, serves actions |

### 2.3 Policy server protocol

One WebSocket per tracking session. Messages are JSON; the frame is base64 JPEG.

```jsonc
// client → server, one per observation (10 Hz)
{"t": 1731900000.123, "jpeg": "<base64>", "state": [yaw, pitch, roll, yaw_rate, pitch_rate, prev_d_yaw, prev_d_pitch], "task": "track the white mug"}

// server → client
{"t": 1731900000.123, "actions": [[d_yaw, d_pitch, present], ...], "infer_ms": 42}
```

- `t` is echoed so the client can measure round-trip time and drop stale replies.
- The server answers only the newest observation. If a new one arrives while it's busy, it drops the older one (no queue build-up).
- A homemade server is used instead of LeRobot's async inference stack because that stack expects a LeRobot `Robot` class, which Reachy Mini doesn't have **[verify]**. Revisit if an official integration exists.

### 2.4 Runtime rules in `TrackBehaviour`

| Situation | Behaviour |
|---|---|
| Fresh chunk (age ≤ 300 ms) | Apply the next unused action, relative to the gaze **at the moment it is executed** (not at the observation): target gaze = current gaze + (d_yaw, d_pitch). This matches how labels are built (§3, "Action chunks") |
| `present < 0.5` for ≥ 1 s | Hold the pose, tell the agent "lost it". The agent may start `Search` |
| No reply for > 500 ms | Switch to `BaselineTracker` (if the object is a COCO class) or hold the pose; log `policy_stale` |
| Server reconnects | Switch back after 1 s of fresh replies (hysteresis, avoids flapping) |
| `stop`, new command, or 60 s timeout | End the behaviour; the arbiter pops back to the previous one |

### 2.5 Head/body split

The policy outputs a **gaze** change: where the camera should point in the room. Gaze yaw = body yaw + head yaw. A fixed rule in the motion loop decides which joint moves:

| | Head yaw | Body yaw |
|---|---|---|
| Soft range ([04 §4](../architecture/04-motion-and-attention.md)) | ±60° | ±160° |
| Speed cap | 120°/s | 60°/s |
| Role | fast, small corrections | follows objects that move around the robot |

```
each tick:
  target_gaze_yaw = current_gaze_yaw + d_yaw
  head takes the correction first (fast)
  if |head_yaw| > 35° for > 0.5 s:
      body turns toward target_gaze_yaw (≤ 60°/s)
      head turns the opposite way at the same rate → gaze unchanged, head recentres
  pitch: head only
```

Why the body isn't a policy output:
- Labels come from the object's position in the image, which doesn't depend on which joint moved the camera. No extra data or labels.
- The policy stays small: it only answers "where should the camera look?"
- Body limits, speed caps and the watchdog stay in fixed code.

The 35° dead zone and 0.5 s hold time stop the body from starting and stopping constantly; tune both on the robot.

Changes needed:
- `robot/adapter.py`: send body yaw. Today `set_pose` passes only head and antennas to `set_target`; the SDK's `set_target` probably takes `body_yaw` **[verify]**.
- `robot/motion.py`: add `body_yaw` to `Pose`, the split rule, body limits and speed cap.

The camera sits off the body's turning axis, so a body turn also shifts the camera slightly. The policy corrects for that on the next frame, but the warp augmentation (§8.1) must not simulate body turns. Body turns may also blur or shake frames: record some episodes with body motion (§6.4).

Body yaw as a **policy output** is only worth it for use cases where the movement style is the point (camera operator V3, animator V8 in the use-cases doc). That needs teleop data with body motion and a 4-value action, so it's a v2 decision.

---

## 3. Observation and action specification

Defined once in `vla/spec.py`.

| Key | Shape | Contents | Notes |
|---|---|---|---|
| `observation.images.front` | (256, 256, 3) | undistorted, resized camera frame | SmolVLA resizes internally; keep the full-res raw frame on disk |
| `observation.state` | (7,) | head yaw, pitch, roll (degrees, at frame capture time); yaw_rate, pitch_rate (°/s) of the **gaze**, i.e. how fast the camera turns: yaw_rate = d(head yaw + body yaw)/dt; prev_d_yaw, prev_d_pitch (the previous action) | rates and previous action give motion information, see below. Body yaw is not a separate state value in v1, but it is needed to compute yaw_rate, so it must be in the raw data |
| `action` | (3,) | `d_yaw`, `d_pitch` (degrees), `present` (0–1) | see below |
| `task` | text | instruction | 3–5 phrasings per object |

SmolVLA only uses these four. Everything else in the raw data (`det_score`, `mask_score`, `cmd_*`, scenario fields, `held_out`) is for review, splits, coverage and evaluation.

**Action meaning.** `d_yaw`, `d_pitch` = the gaze change that would centre the object (§2.5 splits it between head and body). It's the full correction, not a per-step increment: the motion loop's smoothing and rate caps turn it into a smooth movement. Clamped to ±20° per action. `present` = 1 if the object is visible, else 0; when 0, `d_yaw` and `d_pitch` are 0 (hold still). Labels aim at the object's **smoothed** position **one system latency ahead** (§7.2), so the head arrives on time instead of late.

**Why relative and not absolute:** the same image offset always means the same action, regardless of where the head points. That generalises better and keeps action values small.

**Why motion information in the state:** one frame shows where the object is, not where it's going, so a policy that sees only the image always lags a moving object. Head angular rates and the previous action are a cheap fix. If fast objects still lag, add the previous frame (t − 100 ms) as a second image key `observation.images.prev`, at the cost of inference time **[verify against §12.1]**.

**Action chunks.** SmolVLA predicts a chunk of future actions. For tracking, use only the first 1–3 actions of each chunk (`n_action_steps`), so the policy reacts to every new frame. Start with 2. Each label row is relative to **that row's** gaze, so at runtime each action is applied relative to the gaze when it's executed (§2.4). With longer chunks, a mismatch here causes overshoot.

**Preprocessing record.** Store the undistortion model, resize size, latency offset and smoothing window in the dataset card (§9.4). Inference must preprocess frames exactly the same way; a mismatch lowers accuracy without any error.

**Rate.** Observations at 10 Hz. Motion loop at 50 Hz (unchanged).

**Sign convention.** Positive yaw = head turns to the robot's left, positive pitch = head looks down **[verify against the SDK]**. Write a unit test that converts a known pixel offset to the angle and checks the sign on the real robot once.

---

## 4. Baseline and acceptance metrics

### 4.1 Baseline

`BaselineTracker`: a small COCO detector (same as [03 §2](../architecture/03-perception.md)) finds the object, a PI controller with a 1-euro filter drives the head. It only works for COCO classes. That's fine: comparisons use COCO objects, and open-vocabulary tracking is a capability the baseline doesn't have.

### 4.2 Metrics

Computed by `scripts/track_eval.py` from logged frames and boxes (the offline labeller, §7, provides ground truth boxes for evaluation runs too).

| Metric | Definition | Target for shipping |
|---|---|---|
| Centring error | median distance from box centre to frame centre, % of frame width | ≤ baseline, and ≤ 10% |
| Lost rate | % of frames where the object is in view of the wide camera but outside the centre 60% of the frame | ≤ baseline |
| False presence | % of frames where `present ≥ 0.5` but the object is absent | ≤ 5% |
| Smoothness | head direction changes per second | ≤ 2 ([04 §2](../architecture/04-motion-and-attention.md)) |
| Latency | capture time → `set_target` with the resulting target, p95 | ≤ 200 ms |
| Open-vocabulary success | lost rate on held-out non-COCO objects | ≤ 20% (no baseline exists) |

---

## 5. Step 0 spikes (do these before writing anything else)

| Spike | Question | Why it matters |
|---|---|---|
| Camera latency | Time from a physical event (LED flash, phone stopwatch filmed) to the frame arriving in `grab_jpeg()` | The camera streams over WebRTC; if this is > 150 ms the 200 ms budget is gone |
| Frame timestamps | Does the SDK give a capture time per frame, or only arrival time? | Labels need the head pose at capture time (§6.3) |
| Pose readback | `get_current_head_pose()` rate and delay | Needed for `observation.state` and for syncing |
| Calibration | Intrinsics + distortion from ~20 checkerboard images; reprojection error; standard vs fisheye lens model (§6.7) | Pixel → angle conversion, frame warping |
| Camera focus | Fixed focus, or autofocus that can be locked? | Autofocus shifts the calibration slightly |
| Body yaw | Does `set_target` take `body_yaw`? Real body speed and readback rate | The head/body split (§2.5) |
| Camera offset | Distance from the camera to the head's rotation centre | Limits the warp augmentation (§8.1) |
| GPU round trip | Laptop ↔ GPU box: send a 256×256 JPEG, run SmolVLA, return | Sets the server's share of the latency budget |
| Sim | Does the SDK's MuJoCo simulator run here, render the head camera, and accept the same client code? (`use_sim` / sim daemon, [06](../architecture/06-operations.md)) **[verify]** | Decides how much of §11 is possible |

Record the results in this doc, replacing the **[verify]** marks.

---

## 6. Data collection

### 6.1 Physical setup

- Reachy at a fixed, taped spot on the desk. Calibrate the camera once and check it every session (§6.7).
- Props: 10–15 training objects, 3 held-out objects (keep them in a separate box so they never get recorded by accident), a string, a toy car, a turntable, a lamp.
- Two or three backgrounds: the usual desk, a cluttered desk, another room.

### 6.2 Who moves the head while recording

The head doesn't need to track well during recording, because actions are relabelled afterwards (§7). Mix three modes, chosen per episode by the session script:

| Mode | Share | What it gives |
|---|---|---|
| `BaselineTracker` with random jolts (a ±10° kick every 2–5 s) | 50% | Realistic tracking plus recovery examples |
| Slow random sweeps | 30% | Wide range of object positions in the frame |
| Teleop (mouse) | 20% | Human-like anticipation; needed for non-COCO objects the baseline can't follow |

### 6.3 Recorder (`scripts/track_record.py`)

Threads:
- **Camera thread:** grabs frames as they arrive; stores frame index + capture time (or arrival time minus the measured latency from §5 if the SDK has no capture time).
- **Pose thread:** reads the head pose and body yaw at ≥ 50 Hz and logs them with timestamps. Also logs the target the motion loop sent. Body yaw is logged even though v1 doesn't train on it, so a v2 that needs it doesn't require re-recording.
- **Main thread:** runs the session script, keyboard/voice controls, and writes episodes.

Controls (built 2026-11-20, typed lines so it works in any terminal): Enter records / stops early, `r` other scenario, `d` delete the last episode, `n TEXT` note, `q` quit. Episodes stop by themselves after `--seconds`, so your hands stay on the object. Teleop is not built yet. `--target face` records the same layout for face tracking (people list, YuNet tracker, face-specific factors). Guidelines: [track-record-face.md](../usage/track-record-face.md), [track-record.md](../usage/track-record.md).

Raw layout, written as-is (no processing during recording):

```
data/track/raw/
  calib/
    calib_2026-11-20.json    # intrinsics, distortion, image size, date (§6.7)
  session_2026-11-20_a/
    ep_0001/
      frames.mp4             # full resolution, as received
      frames.jsonl           # {i, t_capture}
      poses.jsonl            # {t, yaw, pitch, roll, body_yaw, cmd_yaw, cmd_pitch, cmd_roll, cmd_body_yaw}
      meta.json              # see §6.5
      boxes.jsonl            # written later by the labeller (§7.1)
```

Keep raw data forever. Every later step (labelling, action format, image size) can be rerun from it.

### 6.4 Scenario matrix

| Factor | Values |
|---|---|
| Object | 10–15 training, 3 held-out |
| Motion | still, slow slide, fast pass, circle, leaves and returns, picked up and carried |
| How it moves | hand, string, toy car, turntable |
| Distractor | none, similar object nearby, person walking behind |
| Lighting | daylight, lamp only, backlit window, dim |
| Background | clean desk, cluttered desk, other room |
| Start | centred, off-centre, half out of frame |
| Range | within head range (±60°), moves around the robot (beyond ±60°, so the body turns, §2.5) |
| Instruction | 3–5 phrasings per object, plus colour/position variants ("the mug on the left") |
| Negatives | ~10% of episodes: the named object isn't present → `present = 0` all episode |

You can't record every combination. The session script samples them so each factor value appears in at least 5% of episodes, and the coverage report (§9.3) shows the gaps.

### 6.5 Episode metadata (`meta.json`)

```json
{
  "object_id": "mug_white", "task_text": "track the white mug",
  "motion": "fast_pass", "mover": "string", "distractor": "similar_object",
  "lighting": "lamp", "background": "cluttered", "start": "off_centre",
  "head_mode": "baseline_jolt", "negative": false, "held_out": false,
  "session": "2026-11-20_a", "calib_id": "calib_2026-11-20", "notes": ""
}
```

Field meanings are in Appendix B.

### 6.6 Session protocol (about 1 hour of your time)

1. Check the tape mark and lighting; run the calibration quick check (§6.7) (10 min).
2. Record ~60 episodes of 20–30 s each, following the session script (45 min).
3. Upload the session to the GPU box and start labelling (unattended, ~30 min **[verify]**).
4. Review flagged episodes (10–15 min).

### 6.7 Camera calibration

**What it is.** Calibration measures how this particular camera turns the world into an image, so a pixel offset ("the mug is 180 px right of centre") can be converted into an angle ("turn 16.7° right"). Every label depends on that conversion. It measures three things:

| Number | Plain meaning |
|---|---|
| `fx`, `fy` | How zoomed in the camera is: pixels per unit of angle. Bigger = narrower view |
| `cx`, `cy` | The pixel the lens really points at (often a few pixels off the image middle) |
| `dist` | How much the lens bends the image. Reachy's wide lens curves straight lines near the edges; these numbers straighten it |
| `reproj_err_px` | How well the calibration fits: under ~0.5 px is good, above ~1 px redo it |

Skip it and every label is scaled wrong (the head consistently turns too far or not far enough, and the policy learns that), and angles near the image edges are wrong, which is exactly where tracking matters most.

**How (about 15 minutes, `scripts/track_calibrate.py`):**
1. Print a checkerboard (e.g. 9×6 inner corners), tape it to something flat and stiff, measure the square size.
2. Take ~20 photos with Reachy's camera: near, far, tilted, in the centre, corners and edges of the image.
3. OpenCV finds the corners and solves for the numbers. Try both the standard lens model (`calibrateCamera`, 5 distortion numbers) and the fisheye model (`fisheye.calibrate`, 4 numbers); keep the one with the lower error.
4. Save as `calib/calib_<date>.json`.

Calibrate at the exact resolution and stream (WebRTC) that recording and inference use; a frame saved on the robot may not be the same image as one received over the network.

**How often: once per camera, checked every session.** Calibration belongs to the camera, not the scene: moving the robot, lighting and head pose don't change it.

| Situation | Recalibrate? |
|---|---|
| New session, same robot | No, quick check only |
| Robot moved to another desk or room; different lighting | No |
| Different resolution or crop | Yes (or rescale the numbers, if it's a pure resize of the same image) |
| Stream or SDK media mode changes how the image is cropped | Yes |
| Camera knocked, replaced or repaired; SDK/firmware update touching the camera | Yes |
| A second Reachy unit | Yes, each camera is slightly different |
| Quick check fails | Yes |

**Quick check (2 minutes, every session):** take 3 checkerboard photos (centre, corner, tilted) and measure the error with the **saved** calibration, without recomputing. Under ~1 px → keep it; otherwise recalibrate.

**Field check (optional):** put an object near the image edge, turn the gaze by the computed angle; it should land in the centre.

**Versioning:** each calibration has an ID (`calib_2026-11-20`), and every episode's `meta.json` stores the `calib_id` it was recorded with. Older sessions keep pointing to the calibration valid at the time, so their labels stay correct after a recalibration.

---

## 7. Labelling

### 7.1 Automatic pipeline (`scripts/track_label.py`, GPU)

**Faces (built 2026-11-20):** actions now also cover frames where the face is not visible: hindsight targets (where the face is next seen within 3 s), so the policy learns "body in view: tilt up" and "left fast to the right: turn right"; `d_yaw`/`d_pitch` can be non-zero while `visible = 0`. State gains `last_d_yaw`, `last_d_pitch`, `since_seen_s` (memory of the last sighting). The recorder's tracker uses body and chase rules (`attention/hints.py`) so the camera does find the face again. SCRFD on every frame + ArcFace identity from an automatic per-person gallery, instead of OWLv2 + SAM 2. No new packages (onnxruntime), CPU. Bad episodes are deleted, not fixed. See [track-label.md](../usage/track-label.md). The steps below remain the plan for objects.

Per episode:
1. **Find:** run OWLv2 or Grounding DINO with the object name on the first frames where it's expected. Take the most confident box. If confidence is low, add the episode to the "click needed" list (one click on the object in the first frame).
2. **Follow:** give the box (or click) to SAM 2 video tracking for the whole episode. It handles brief occlusion and returns a mask per frame.
3. **Re-check:** every 10th frame, rerun the detector; if the SAM 2 mask and the detector disagree strongly, flag the episode.
4. **Absent frames:** when SAM 2 has no mask (object left the frame or fully hidden), mark `present = 0`.
5. Write `boxes.jsonl`: `{i, box_xyxy, mask_score, det_score, present}`.

### 7.2 Actions from boxes (`vla/geometry.py`)

For each frame t with the object present:

```
1. smooth:   box centres over time with a centred moving average (3–5 frames; fine offline)
             do not smooth `present` changes
2. look ahead: take the smoothed centre at frame t + L, where L = measured system latency
             (§12.1, start with 200 ms); if the object is absent at t + L, use frame t
3. undistort (u, v) with the episode's calibration (calib_id in meta.json)
4. d_yaw   = -atan((u - cx) / fx)     # degrees; sign per §3, checked by unit test
   d_pitch =  atan((v - cy) / fy)
5. action  = [clamp(d_yaw, ±20), clamp(d_pitch, ±20), 1.0]
```

Absent frames: `action = [0, 0, 0]`.

- **Smoothing** removes the few-pixel wobble of SAM 2 boxes. Without it, the wobble becomes noisy labels and the policy learns to twitch.
- **Looking ahead** compensates for latency: the robot acts about L after the frame is taken, so labels aim where the object *will* be. This is only possible offline, where future frames are known, and it's the biggest win for fast objects. A wrong L makes the head consistently lead or lag, so set it from the measured latency.

The executed head motion is **not** used as the label. Only the object's position is, so messy recordings still give correct labels (this is why the recording modes in §6.2 can be imperfect).

### 7.3 Human review

A small viewer (OpenCV window) plays each flagged episode with the box and action arrow drawn on it. Keys: `o` OK, `f` fix (click the object on the bad frame; SAM 2 reruns from there), `x` delete. Use CVAT or Label Studio instead only if you don't want to write this viewer. Expect 5–10% of episodes flagged **[verify]**.

### 7.4 Label quality checks

- **Timing test (once per session):** turn the head sharply while a still object is in view. The box must jump in the same frame as the pose change. A lag means the timestamps (§6.3) are wrong.
- **Spot check:** watch 10 random unflagged episodes with arrows drawn. The arrows must point at the object.
- **Sanity numbers:** fraction of `present = 0` frames per episode; action value histogram (no spikes at the ±20° clamp).

---

## 8. Extra data

### 8.1 Real-frame warping

A small head rotation is close to a pure camera rotation, which can be reproduced exactly by warping the image (a homography from the calibration). From each real frame, generate 3–5 virtual frames as if the head pointed a few degrees elsewhere; the new action is the original action minus the virtual rotation.

- Undistort first; warp; crop inside the valid image only (no invented pixels).
- Limit to ±10° (closer objects + the camera offset from the rotation centre cause parallax the warp doesn't model; tighten the limit if §5 shows a large offset).
- Target share: ~20% of training frames.

### 8.2 Simulation data (optional, ~10%)

Use MuJoCo only for cases real data lacks: very fast motion, full occlusion, objects leaving and returning. Randomise textures, lighting, background (real desk photos as backdrops), camera noise and motion blur, or the data barely helps. Scanned real objects (real2sim) can be used as the moving objects. Don't use sim-only training.

---

## 9. Dataset build

### 9.1 Conversion (`scripts/track_to_lerobot.py`)

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset.create(
    repo_id="<you>/reachy_track_v1", fps=10, robot_type="reachy_mini",
    features={
        "observation.images.front": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": ["yaw", "pitch", "roll", "yaw_rate", "pitch_rate", "prev_d_yaw", "prev_d_pitch"]},
        "action": {"dtype": "float32", "shape": (3,), "names": ["d_yaw", "d_pitch", "present"]},
    },
)
for ep in episodes:
    for frame in ep.frames_at_10hz():          # resample: nearest pose to each frame's capture time
        ds.add_frame({"observation.images.front": frame.image, "observation.state": frame.state,
                      "action": frame.action, "task": ep.meta["task_text"]})
    ds.save_episode()
ds.finalize()  # required: closes parquet writers, otherwise the dataset is corrupt
```

Check the `add_frame` format against the installed LeRobot version **[verify]**. A worked example of one episode, raw and converted, is in Appendix A. Keep the dataset private on the Hub, or local, since it contains footage of your home.

### 9.2 Splits

Split by **object and background**, never by random frames (neighbouring frames are near-identical, so random splits inflate scores).

| Split | Contents |
|---|---|
| train | training objects, training backgrounds |
| val | training objects, one held-out background |
| test | held-out objects, all backgrounds (touched only for final reports) |

### 9.3 Coverage report

Count episodes per factor value from `meta.json` (§6.4). Any value below 5% → record more of it next session.

### 9.4 Versioning

Name datasets `reachy_track_v<N>`. Record in the dataset card: sessions included, labeller version, calibration date, warp share, sim share. A trained checkpoint records the dataset version it used.

---

## 10. Training

### 10.1 Compute

The laptop is CPU-only. Options: the LeRobot SmolVLA Colab notebook (first runs), a rented GPU, or your own GPU box. LeRobot's docs give ~4 h for 20k steps on one A100.

### 10.2 Command

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<you>/reachy_track_v1 \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/reachy_track_v1 \
  --job_name=reachy_track_v1 \
  --policy.device=cuda \
  --wandb.enable=true
```

### 10.3 Settings to tune

| Setting | Start | Why |
|---|---|---|
| `n_action_steps` | 2 | React to every new frame |
| Chunk size | default, then shorter if it lags | Long chunks plan too far ahead for a moving object |
| Steps | 20k, check val loss every 2k | Small datasets overfit; keep the best val checkpoint |
| Image augmentation | colour jitter, small blur | Lighting robustness |
| Normalisation | per-dimension mean/std from train split | `present` is 0/1; check it isn't squashed |

### 10.4 Training order

1. **Pipeline run:** 1 session (~60 episodes), 2k steps. Goal: prove that recorder → labeller → dataset → training → server → robot works end to end. Accuracy doesn't matter yet.
2. **First real model:** 3–4 sessions, 20k steps.
3. **Add warp data**, then **sim data** only if specific failure cases need it. Compare against the previous checkpoint on the same val/test sets every time.

### 10.5 Offline evaluation (replay)

Run the checkpoint on val/test episodes frame by frame and compare to the labels: mean angle error of `d_yaw`/`d_pitch`, and accuracy of `present`. This is open loop (the policy's actions don't move the camera), so it can't measure smoothness or recovery, but it's the fastest accuracy gate and needs no robot.

---

## 11. Simulation testing

Split of responsibilities (decided 2026-11-20):
- **Sim and unit tests verify safety logic**: no command outside the limits, no NaN, no jumps, correct fallback and stop.
- **Real tests verify accuracy** (§10.5 replay, §11.4 real-image crop, §12.4 live), plus the physical risks sim can't model (§11.3).

Simulator images don't look like your desk, so **sim results are not accuracy claims**.

### 11.0 Status and known gaps (2026-11-20)

- `scripts/face_sim_example.py` ([usage](../usage/face-sim.md)) is a **tracking demo, not a safety test**. It shows that MuJoCo renders a face board, the head-camera loop works, and the true error comes free from the sim. But it uses a 2-hinge stand-in instead of the Reachy Mini model, clamps angles with its own `np.clip` instead of `robot/motion.py`, and runs none of the tests below.
- **Bug: `clamp()` lets NaN through.** `clamp({"yaw": nan, "body_yaw": nan})` returns NaN for both (`inf` is capped). Python's `min`/`max` don't reject NaN, so a NaN action would reach `set_target`.
- **No rate cap in the motion layer.** `goto(..., duration_s=0.001)` jumps in one tick. Callers like `attention/search.py` slow their own moves; a policy output would not be slowed.
- **§12.2 doesn't match the code.** §12.2 says yaw ±60°, pitch −30°/+25°, body ±160°, rate caps 120°/s and 60°/s. `motion.LIMITS` has yaw ±45° (relative to body), pitch ±30°, roll ±25°, body ±120°, and no rate cap. Pick one source of truth before writing tests.

### 11.1 Layer 1: unit tests on the motion layer (no MuJoCo)

Every command goes through `Motion` → `set_pose`, so a fake `set_pose` that records each sent pose, plus a fake clock, checks the rules in milliseconds in CI. File: `tests/unit/test_motion_safety.py`. Each test feeds bad or random input and asserts the rule on **every** recorded pose.

| # | Test | Input | Must hold for every sent pose |
|---|---|---|---|
| 1 | Position limits | Random poses, extremes, ±1e9 | Every axis within `LIMITS` |
| 2 | Bad numbers | NaN, inf, missing keys | All values finite; head holds last good pose (**fails today**) |
| 3 | Head vs body | Body at ±limit, head pushed the other way | \|yaw − body_yaw\| ≤ head yaw limit |
| 4 | Rate cap | `goto` with 0.001 s; policy jumping ±90° per tick | Change per tick ≤ cap × tick (**fails today**: no cap) |
| 5 | No shaking | Policy flipping left/right every tick | ≤ N direction changes per second |
| 6 | Stop | `stop()` mid-move | No new target after one tick; eased return to neutral, no jump |
| 7 | Watchdog | Policy silent, or replies 400 ms late | Head holds within 500 ms; stale replies dropped; no jump on resume |
| 8 | Crash in a move | `pose_at` raises | No more commands; error reported |
| 9 | Soak | 30 simulated minutes of random input | All rules hold; queue and memory bounded |

Tests 6–7 depend on how `TrackBehaviour` feeds `Motion`; adjust once it exists.

### 11.2 Layer 2: MuJoCo with the SDK's Reachy Mini model

Blocked on the §5 "Sim" spike (does the SDK sim run and render the head camera?). Setup:
- The SDK's Reachy Mini MJCF, not a stand-in.
- The app's adapter pointed at the sim, so the real arbiter → `TrackBehaviour` → `Motion` → adapter path drives it.
- Scene: the face board from `face_sim_example.py` (photo on a moving plane), scripted motions, an optional second face as a distractor.
- Fault injection: NaN or ±90° outputs, 400 ms delay, server killed mid-track.
- File: `tests/sim/test_sim_safety.py` (pytest, skipped when MuJoCo or the model is missing).

| # | Test | Why it needs the real model | Pass condition |
|---|---|---|---|
| 10 | Reach sweep over every corner of the `LIMITS` box (x/y/z/roll/pitch/yaw combined) | Each axis can be in range while the combination is unreachable for the Stewart platform | Inverse kinematics solves; no self-collision |
| 11 | Motor-level limits | Checks the 6 motor angles and their speeds, not just the head pose | Within the model's joint ranges and a speed limit |
| 12 | Body turn | Target circles the robot | Body stays within its limit; ≤ 2 head/body direction changes per second |
| 13 | End-to-end with faults | Real app code, real model | Rules 1–7 hold on sim motor positions; fallback within 500 ms; `policy_stale` logged |

`face_sim_example.py` stays as the accuracy and behaviour demo. Its one planned change is to call `motion.clamp` instead of `np.clip`, so the demo goes through the app's limits.

### 11.3 Layer 3: real robot only

Sim can't model these, so the first real runs (§12.3 "Supervised") cover them:
- motor temperature and current during a 10-minute soak;
- cable strain from body turns;
- real motor lag and overshoot;
- bumping into nearby objects;
- power or USB dropouts.

Start at half limits with someone ready to cut power.

Optional last guard: run the Layer 1 rule checks inside `set_pose` on the real robot too. Log and refuse any command that breaks a rule.

### 11.4 Real-image replay simulator (accuracy, optional)

For closed-loop tests on real images: record wide, still-camera videos of moving objects or faces. The "virtual head" is a crop that the policy moves around (using the warp from §8.1). This tests recovery and smoothness on real images without the robot. Ground truth comes from a detector on the full frame, so scores are approximate. Limited to the wide camera's field of view.

### 11.5 Order of work

1. Fix NaN in `clamp`; add a per-tick rate cap in `Motion._play`; reconcile limits with §12.2.
2. Write Layer 1 tests 1–9.
3. Switch `face_sim_example.py` to `motion.clamp`.
4. Run the §5 "Sim" spike, then write Layer 2 tests 10–13.

---

## 12. Real deployment

### 12.1 Latency budget (target ≤ 200 ms p95)

| Stage | Budget | Measured |
|---|---|---|
| Camera → laptop (WebRTC) | 80 ms | **[verify]** |
| Laptop → GPU box → laptop (network) | 20 ms | **[verify]** |
| SmolVLA inference | 50 ms | **[verify]** |
| Behaviour + motion loop pickup | 20 ms | |
| Slack | 30 ms | |

If the camera alone takes most of the budget, reduce the frame size, or accept a slower but smoother tracker (longer chunks).

### 12.2 Safety

Applied in the motion loop, so they hold whatever the policy outputs. **Not yet true of the code**: see §11.0 (NaN passes `clamp`, no rate cap, limits differ). Tested by §11.1–11.3.
- Soft limits (head yaw ±60°, pitch −30°/+25°, roll ±20°, body yaw ±160°) and rate caps (head 120°/s, body 60°/s) from [04 §4](../architecture/04-motion-and-attention.md).
- Action clamp ±20° and NaN rejection in `TrackBehaviour`.
- Watchdog: no base target for 1 s → ease to neutral.
- Stop: the voice command "stop" and a keyboard key end tracking at once.

### 12.3 Rollout stages

| Stage | What runs | Exit condition |
|---|---|---|
| **Shadow** | Baseline drives the head; the policy runs alongside and its actions are only logged | Policy's logged actions agree with the baseline on COCO objects; latency within budget |
| **Supervised** | Policy drives; you watch with a hand on the stop key; fixed test script (§12.4) | §4 metrics met on the script |
| **Normal** | `track()` tool available to the voice agent | — |

### 12.4 Real test script

Run the same script for the baseline and each checkpoint, and keep the logs:
1. Still object, centred and off-centre, for each of 3 training and 3 held-out objects.
2. Slow slide left-right, fast pass, circle.
3. Object leaves the frame and returns.
4. Similar distractor next to the target.
5. Lamp only, then backlit.
6. Named object absent (negative).

### 12.5 Logging

Per tick: observation time, reply time, action, `present`, which tracker drove the head (policy or baseline), head target. Save the frames of evaluation runs so `track_eval.py` can label them and compute the §4 metrics.

---

## 13. Improving after deployment

Where the policy fails (loses fast objects, locks onto a distractor), take over with teleop during the run. The recorder saves those segments as new episodes; the labeller labels them like any other. Retrain with them added (DAgger-style correction). This gives the largest improvement per minute of recording.

---

## 14. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Step 0 spikes | §5 results written into this doc |
| M1 | Baseline tracker | `BaselineTracker` runs on the robot; §4 metrics measured for it |
| M2 | Recorder + calibration | One session recorded; timing test passes (§7.4) |
| M3 | Labeller + converter | Session labelled and converted; spot check passes |
| M4 | Pipeline run | 2k-step model drives the head in sim and on the robot (any accuracy) |
| M5 | Sim test suite | All §11.2 tests pass |
| M6 | First real model | 3–4 sessions, 20k steps; replay results on val/test |
| M7 | Shadow + supervised | §12.3 stages passed; §4 targets met, or a written reason why not |
| M8 | Ship | `track()` tool live; usage doc in `docs/usage/track.md` |

---

## 15. Risks and open questions

| Risk | Effect | Mitigation |
|---|---|---|
| WebRTC camera latency too high | Tracking lags fast objects | Measure first (§5); smaller frames; accept a slower tracker |
| No capture timestamps from the SDK | Labels misaligned in time | Estimate from measured latency; timing test every session |
| GPU box not on the LAN | Cloud round trip adds 50–150 ms | Measure; keep the baseline as default if the budget breaks |
| Labeller errors copied by the policy | Policy inherits teacher mistakes | Re-check step (§7.1), review, test on held-out objects |
| Small dataset overfits to one desk | Fails in new rooms | Background variety, split by background, warp + colour augmentation |
| VLA doesn't beat the baseline | Wasted effort | Accept it: ship the baseline for COCO objects, VLA only for open vocabulary |
| Privacy | Home footage in the dataset | Keep it local or private; no people's faces in held-out sharing; delete raw sessions on request |

Open questions:
- Does an official LeRobot integration for Reachy Mini exist, so its recorder or async inference can be reused?
- Exact SDK sign conventions and joint limits.
- Whether open-vocabulary tracking with a VLM-seeded SAM 2 tracker (no training) is fast enough on the GPU box. If so, it's a strong second baseline and should be added to §4.

---

## Appendix A. Example episode

One episode, "keep looking at the white mug", fast pass under lamp light, shown as the recorder writes it (A.1), how one label is computed (A.2), and after conversion (A.3). Values are illustrative but consistent with each other. The calibration numbers are placeholders until §5 is done **[verify]**.

### A.1 Raw recording (`data/track/raw/session_2026-11-20_a/ep_0007/`)

**`meta.json`**
```json
{
  "object_id": "mug_white",
  "task_text": "keep looking at the white mug",
  "motion": "fast_pass", "mover": "hand", "distractor": "none",
  "lighting": "lamp", "background": "cluttered", "start": "off_centre",
  "head_mode": "baseline_jolt", "negative": false, "held_out": false,
  "session": "2026-11-20_a", "calib_id": "calib_2026-11-20", "duration_s": 25.0, "notes": ""
}
```

**`calib/calib_2026-11-20.json`** (one per calibration, shared by every session that uses it, §6.7)
```json
{"width": 1280, "height": 720, "fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 360.0,
 "dist": [-0.28, 0.07, 0.0, 0.0, 0.0], "reproj_err_px": 0.41, "date": "2026-11-20"}
```

**`frames.jsonl`** (one line per camera frame; the camera runs faster than 10 Hz)
```jsonl
{"i": 0,  "t_capture": 1763600000.000}
{"i": 1,  "t_capture": 1763600000.033}
{"i": 2,  "t_capture": 1763600000.067}
```

**`poses.jsonl`** (50 Hz: measured head pose, plus `cmd_*` = the target the motion loop sent)
```jsonl
{"t": 1763600000.000, "yaw": 0.0, "pitch": 5.0, "roll": 0.0, "body_yaw": 0.0, "cmd_yaw": 0.0, "cmd_pitch": 5.0, "cmd_roll": 0.0, "cmd_body_yaw": 0.0}
{"t": 1763600000.020, "yaw": 0.4, "pitch": 5.0, "roll": 0.0, "body_yaw": 0.0, "cmd_yaw": 1.2, "cmd_pitch": 5.0, "cmd_roll": 0.0, "cmd_body_yaw": 0.0}
```

**`boxes.jsonl`** (written by the labeller, §7.1)
```jsonl
{"i": 0,   "box_xyxy": [760, 250, 880, 350], "det_score": 0.81, "mask_score": 0.93, "present": true}
{"i": 3,   "box_xyxy": [744, 252, 864, 352], "det_score": null, "mask_score": 0.92, "present": true}
{"i": 540, "box_xyxy": null, "det_score": null, "mask_score": 0.0, "present": false}
```

`det_score` is `null` on frames where the detector didn't run (it re-checks every 10th frame); SAM 2 covers every frame.

### A.2 One label, step by step (frame 0, §7.2)

```
box centre          u = (760+880)/2 = 820,  v = (250+350)/2 = 300
offset from centre  u - cx = 180 px,  v - cy = -60 px
d_yaw   = -atan(180 / 600) = -16.7°     object right of centre → turn right
d_pitch =  atan(-60 / 600) =  -5.7°     object above centre → look up
action  = [-16.7, -5.7, 1.0]
```

This shows only the angle step. Real code first smooths the box centres and takes the centre one latency ahead (§7.2), then undistorts it with `dist`.

### A.3 After conversion: LeRobot dataset at 10 Hz

| frame_index | timestamp | task | observation.state (yaw, pitch, roll) | action (d_yaw, d_pitch, present) | Note |
|---|---|---|---|---|---|
| 0 | 0.0 | keep looking at the white mug | [0.0, 5.0, 0.0] | [-16.7, -5.7, 1.0] | starts off-centre |
| 1 | 0.1 | ″ | [-6.1, 3.9, 0.0] | [-10.9, -3.6, 1.0] | head catching up |
| 2 | 0.2 | ″ | [-12.8, 2.0, 0.0] | [-4.8, -1.4, 1.0] | |
| 3 | 0.3 | ″ | [-16.2, 0.9, 0.0] | [-1.1, -0.3, 1.0] | nearly centred |
| 92 | 9.2 | ″ | [-14.0, 0.5, 0.0] | [9.0, 0.2, 1.0] | random jolt pushed the head away |
| 93 | 9.3 | ″ | [-8.3, 0.6, 0.0] | [3.5, 0.1, 1.0] | recovering |
| 180 | 18.0 | ″ | [22.5, 1.0, 0.0] | [0.0, 0.0, 0.0] | mug left the frame |
| 181 | 18.1 | ″ | [22.5, 1.0, 0.0] | [0.0, 0.0, 0.0] | hold still |
| 196 | 19.6 | ″ | [22.5, 1.0, 0.0] | [18.4, -2.0, 1.0] | mug came back on the left |
| 249 | 24.9 | ″ | [3.1, 4.7, 0.0] | [0.4, 0.1, 1.0] | end of episode |

For readability the state column shows only yaw, pitch, roll; the full state also has yaw_rate, pitch_rate and the previous action (§3). Each row also has its 256×256 frame (`observation.images.front`), stored in the video files, not the table. A **negative episode** ("keep looking at the red ball" with no ball present) has `action = [0, 0, 0]` on every row.

### A.4 Folder layout (LeRobot v3 **[verify against the installed version]**)

```
reachy_track_v1/
  meta/
    info.json                              # fps=10, features, robot_type, counts
    stats.json                             # mean/std per feature (normalisation)
    tasks.parquet                          # task text → task_index
    episodes/chunk-000/file-000.parquet    # per-episode index, length, task
  data/chunk-000/file-000.parquet          # the rows in A.3
  videos/observation.images.front/chunk-000/file-000.mp4
```

### A.5 Size of a full v1 dataset

| | Value |
|---|---|
| Sessions | 4 |
| Episodes | ~240 (~10% negatives) |
| Frames at 10 Hz | ~60,000 (~1.7 h) |
| Objects | 12 training + 3 held-out |
| Tasks (phrasings) | ~50 |
| Split (§9.2) | train: training objects, 2 backgrounds · val: training objects, 3rd background · test: held-out objects |

---

## Appendix B. What each file means

Each raw file answers one question. `.json` holds one object; `.jsonl` (JSON Lines) holds one object per line, so the recorder can append without rewriting the file, and a crash loses only the last line.

| File | Question it answers | Written by | Read by | How often |
|---|---|---|---|---|
| `calib/calib_<date>.json` | How does this camera map pixels to angles? | `track_calibrate.py` | labeller, converter, warp augmentation | once per calibration (§6.7) |
| `meta.json` | What was this episode about? | recorder | converter, coverage report, splits | once per episode |
| `frames.jsonl` | When was each frame taken? | recorder | converter (time sync) | one line per frame |
| `poses.jsonl` | Where were head and body pointing, and where were they told to point? | recorder | converter (time sync, state) | one line every 20 ms |
| `boxes.jsonl` | Where is the object in each frame? | labeller | converter (actions), review | one line per frame |

### B.1 `calib/calib_<date>.json`

| Field | Meaning | Why it matters |
|---|---|---|
| `width`, `height` | Image size the calibration was done at | The other numbers are only valid at this size; if frames are resized, scale `fx, fy, cx, cy` by the same factor |
| `fx`, `fy` | Focal length in pixels: how zoomed in the camera is | `angle = atan(pixels / fx)`; with `fx = 600`, 180 px off centre is 16.7° |
| `cx`, `cy` | Optical centre: the pixel the camera axis passes through | The point the gaze tries to put the object on |
| `dist` | Lens distortion coefficients (standard model: k1, k2, p1, p2, k3; fisheye model: 4 numbers) | Straightens the wide lens's bending before the angle formula, which assumes a perfect lens |
| `reproj_err_px` | Average error re-predicting the checkerboard corners | Calibration quality: < 0.5 px good, > 1 px redo |
| `date` | When it was measured | Older sessions keep their own calibration via `calib_id` |

### B.2 `meta.json`

| Field | Meaning | Used for |
|---|---|---|
| `object_id` | Which physical object (a stable ID, not the wording) | Splits by object; coverage |
| `task_text` | The instruction the policy sees | The `task` column in training |
| `motion`, `mover`, `distractor`, `lighting`, `background`, `start` | Scenario factors (§6.4) | Coverage report; finding which conditions fail |
| `head_mode` | How the head was moved while recording | Checking the mix; training ignores it, since labels are recomputed |
| `negative` | `true` if the named object was deliberately absent | Sanity check: every frame must be `present = false` |
| `held_out` | `true` if the object belongs to the test set | Keeps test objects out of training |
| `session`, `calib_id` | Recording session; calibration used | Matching the right calibration; dropping a bad session |
| `duration_s`, `notes` | Length; free text ("mug fell over at 12 s") | Review |

### B.3 `frames.jsonl`

| Field | Meaning |
|---|---|
| `i` | Frame number inside `frames.mp4` |
| `t_capture` | When the frame was taken (Unix seconds): the camera's capture time, or arrival time minus the measured camera delay if the SDK gives only arrival time (§5) |

A video file stores frame order, not real times. `t_capture` lets the converter find the head pose at the moment each frame was taken, and pick frames for 10 Hz.

### B.4 `poses.jsonl`

| Field | Meaning |
|---|---|
| `t` | Time of the reading (same clock as `t_capture`) |
| `yaw`, `pitch`, `roll`, `body_yaw` | **Measured** angles in degrees, read back from the SDK |
| `cmd_yaw`, `cmd_pitch`, `cmd_roll`, `cmd_body_yaw` | The **target** the motion loop sent at that moment |

Measured angles become `observation.state` (and its rates). Commanded angles are not labels; the gap between commanded and measured shows how far behind the motors are, which helps latency debugging. Logged at 50 Hz, faster than the camera, so there is always a reading close to each frame.

### B.5 `boxes.jsonl`

| Field | Meaning |
|---|---|
| `i` | Same frame number as in `frames.jsonl` |
| `box_xyxy` | Object box in pixels: left, top, right, bottom; `null` when absent |
| `det_score` | Text-prompted detector confidence (0–1); `null` on frames where it didn't run (every 10th frame only) |
| `mask_score` | SAM 2's confidence that its mask is still on the object (0–1) |
| `present` | Whether the object is visible; becomes the third action value |

Box centres + calibration give `d_yaw`, `d_pitch` (§7.2). Low scores, or detector and SAM 2 disagreeing, send the episode to review.

### B.6 LeRobot files

Created by LeRobot during conversion, never by hand. They're a build output; the raw files above are the permanent source.

| File | Meaning |
|---|---|
| `meta/info.json` | Dataset description: fps, features with names and shapes, robot type, counts, file layout |
| `meta/stats.json` | Mean, std, min, max per feature; used to scale inputs and outputs during training |
| `meta/tasks.parquet` | Instruction texts, each with a number that rows refer to |
| `meta/episodes/…parquet` | One row per episode: length, task, where its rows and video segment are stored |
| `data/…parquet` | The training rows (A.3) |
| `videos/…mp4` | The frames, stored as video |
