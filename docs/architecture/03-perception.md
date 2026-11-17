# 03 — Perception (F3, F4, F5, F6, F10)

## 1. Perception loop

One thread owns the camera. Each cycle:

```
frame = robot.get_frame()                 # BGR ndarray, plus head pose at capture time
for model in scheduler.enabled():         # only models some behaviour/tool asked for
    model.run(frame) → results
world.update(faces=…, hands=…, objects=…, t=capture_t, head_pose=…)
```

- **Rate**: target 10 Hz, capped by CPU. The scheduler skips a model's run if its last run is younger than its period.
- **Model leases** **[arch]**: behaviours/tools call `perception.require("faces", by="FaceTrack")` and `release(...)`. A model runs while ≥ 1 lease holds it. Faces are always leased at a low rate (3 Hz) for emotion context and the greeting.
- **Capture pose**: each result stores the head pose at capture time. Boxes are only meaningful in that pose's camera frame (needed for F5 search across poses and for look-at).
- **Downscale**: models run on 640×360 (or smaller per model). Results are mapped back to full-res pixel coordinates for `look_at_image(u, v)`.

## 2. Local CPU models **[arch]**

| Feature | Model (CPU) | Output | Period when leased | Est. cost **[verify]** |
|---|---|---|---|---|
| Faces, head pose (F4, F6) | MediaPipe Face Landmarker (TFLite, with transformation matrix + blendshapes) | box, 478 landmarks, head yaw/pitch/roll | 100 ms | 15–25 ms |
| Expression (F3) | FER+ ONNX (emotion-ferplus, 64×64 grayscale crop), smoothed over 1 s | {neutral, happy, sad, surprise, anger} + conf | 300 ms | 3–5 ms/face |
| Hands / pointing (F10) | MediaPipe Hand Landmarker | 21 landmarks, `pointing` flag, ray | 100 ms | 15–25 ms |
| Objects (F4, F5) | YOLOX-nano (Apache-2.0), ONNX via onnxruntime at 416 px, + OpenCV CSRT tracker between detections (F4, built 2026-11-17 in `perception/objects.py`) | label, conf, box | detector 1 s, tracker every frame (only while tracking) | detector ~16 ms, CSRT ~16 ms |

Budget check: the worst case runs all four (pointing during a find) at ~150 ms/cycle → ~6 Hz. That's acceptable. The Step 0 spike measures the real numbers on this laptop. Detector licence: prefer an Apache/MIT-licensed export (e.g. YOLOX-nano, or RT-DETR-lite); note AGPL if an Ultralytics model is chosen **[arch]**.

Derived signals:
- **Pointing**: `pointing = index extended ∧ middle/ring/pinky curled`, held ≥ 300 ms. Ray = index MCP → tip, extended in image space.
- **Pointed target** (F10): walk along the ray from the fingertip. Pick the first object detection the ray intersects; otherwise take a 25%-of-width crop centred 1.5× finger-length beyond the tip. That point drives `look_at_image`. After the head moves, a fresh frame crop is used for the cloud call.
- **Who's speaking** (for FaceTrack): the largest face, biased toward the DoA angle while VAD is active.
- **Face track IDs**: IoU-based tracking across frames. No identity recognition (face ID is future work).

## 3. WorldState

- Written only by perception, the audio/DoA threads and the motion thread (head pose). Read by everyone via `world.snapshot()` (lock + shallow copy of immutable tuples, < 0.1 ms).
- Each field carries `t`. Readers treat data older than 500 ms as missing ("stale-is-absent"), so a crashed perception thread turns into "I can't see right now", not stale tracking.
- `latest_jpeg()` encodes on demand, only for cloud calls.

## 4. Cloud vision (`vision_client.py`)

**[changed 2026-11-16]** The first vision feature, the `camera` tool ("what is this?"), uses the **Reachy Hugging Face realtime server** instead (the same server as speech-to-text and as the reference app's camera tool): `perception/vision.py` `ask_image(jpeg, question)` sends the frame as an `input_image` item and reads the text answer. Spike: about 1.5 s per answer plus 0.9 s to open a session, and it runs alongside a speech-to-text session on the same token. No OpenAI key is needed. **[added 2026-11-16]** `VISION_PROVIDER=deepseek` switches it to DeepSeek chat with `image_url` content (`ask_image_chat`, OpenAI-compatible): `deepseek-chat` answered a test image in 1.1 s (2.7 s for the whole agent round trip vs 5.0 s with `hf`). `deepseek-v4-pro` can't see images. The OpenAI design below stays the plan for `confirm`/`locate` (F5) unless the HF model proves good enough there.

Original plan: async HTTPS to an OpenAI vision-capable model (`VISION_MODEL` in config), JPEG ≤ 1024 px long side, quality 80. Three calls with structured JSON output:

| Call | Used by | Input | Output |
|---|---|---|---|
| `confirm(query, crops[])` | F5 find | the user's phrase ("my phone") + up to 6 numbered candidate crops + one wide frame | `{best_index, confidence, description}` |
| `locate(query, frame)` | F5 fallback when the detector lacks the class | a frame per scan pose | `{found, box_norm, confidence}` |
| `describe(frame, crop?, mode)` | F7 investigate, F10 "what is this" / "read this" | frame or crop, `mode ∈ {describe, read_text}` | `{text}` |

- Timeout 6 s. One retry only if > 3 s remain within the tool's budget. On failure → `{"error": "vision_unavailable"}` → the robot says it can't see that right now.
- Concurrency: max 2 in flight. Calls carry the tool's trace id for logs.

## 5. Search strategy (F5)

1. Scan the poses in order (default 7: centre, left 30°, left 60°, right 30°, right 60°, down-left, down-right), each held 600 ms to settle, with a detector lease.
2. At each pose, collect detections whose label maps to the query (a small synonym map; the LLM gives a `coco_hint` argument, e.g. "cell phone").
3. If any candidates: `confirm()` on the crops. Stop early if confidence ≥ 0.7.
4. Else: `locate()` on the best two pose frames (semantic fallback for non-COCO things like "my keys").
5. Result → look at it (`look_at_image` in the capture pose) and describe its location relative to the robot and the speaker ("on the desk, to your left"), using pose yaw + image position.
6. Write the sighting to memory. Total budget 15 s (acceptance F5).
