# Object tracking: "track my iPhone", "follow the cup"

Reachy follows an object with its head and body until you say stop: a phone in your hand, a cup, a bottle, a book. It finds the object once, then follows it every frame.

## Setup (once)

```bash
uv sync --inexact --extra perception      # OpenCV (keeps your other extras)
```

The model downloads by itself the first time it is used (into `models/yolox/`, from the [YOLOX release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)). The first track or find then waits for the download: 3.6 MB for nano, up to about 400 MB for `x`.

### Another model size

Set `OBJECT_MODEL` in `.env` to a size name: `nano` (default), `tiny`, `s`, `m`, `l` or `x`. A missing model downloads by itself on first use. You can also give the path of any YOLOX COCO ONNX; a path that isn't a YOLOX release file is never downloaded. The input size is read from the model.

| Model | Input | Size | Notes |
|---|---|---|---|
| `yolox_nano.onnx` (default) | 416 | 3.6 MB | About 16 ms per frame on the laptop |
| `yolox_tiny.onnx` | 416 | 20 MB | More accurate, about 2x slower |
| `yolox_s.onnx` | 640 | 36 MB | Better on small objects (a phone far away), about 5x slower |

```bash
# .env
OBJECT_MODEL=tiny        # same as models/yolox/yolox_tiny.onnx
```

Only YOLOX models work. Ultralytics YOLOv8/YOLO11 exports have a different output format and are refused at load: `... is not a YOLOX COCO model`.

### Camera delay (once per robot and Wi-Fi)

Camera frames reach the laptop late, so tracking needs to know how late. With the agent app stopped, run:

```bash
uv run python scripts/camera_lag.py
```

Point the robot at a scene with texture and keep still. The head steps 8° left and right 6 times, and the script prints `CAMERA_LAG_S=0.xx`. Put that line in `.env`. The default is 0.3 s. A value more than about 0.1 s off makes the head overshoot.

## Run

```bash
uv run reachy-demo agent --mic --robot
```

It needs `--robot` and the robot's camera. Without a camera, the tool is not offered.

## What you can say

| You say | What Reachy does |
|---|---|
| "Track my iPhone", "follow the cup", "theo dõi điện thoại của tôi" | Finds it in view and follows it. Says so |
| "Stop", "stop tracking", "thôi", "dừng lại" | Stops, and the head eases back to the front |

If the object isn't in view, tracking starts anyway. Reachy says it will follow the object as soon as it sees it, and asks you to hold it up. Hold it at about 0.5–1.5 m. It waits up to 60 s for the object to show up.

## What it can track

The 80 everyday object classes the detector knows (COCO), for example: cell phone, cup, bottle, wine glass, book, remote, laptop, keyboard, mouse, scissors, teddy bear, banana, apple, sports ball, backpack, umbrella, clock, vase.

Not supported yet: keys, pens, AirPods, glasses, wallets, or any other object outside those classes. The LLM picks the class for what you said (a mug is `cup`, an iPhone is `cell phone`). When no class is that kind of object, it picks `none` instead of a look-alike (so keys are never tracked as a `remote`), and Reachy says it can only track everyday objects.

## How it works

1. **Find (once, under 1 s; 1–3 s with several candidates).** YOLOX-nano detects every object of that class in one frame. If there are several (your phone and another on the table), the frame is sent to the vision model (`VISION_PROVIDER`) with numbered boxes: "which box is my iPhone?" If that fails, the most confident box wins. If none is in view, it waits. The first one that shows up is followed (the largest, if several show up at once; the vision model doesn't pick then).
2. **Follow (about 15–20 Hz).** It tracks the object with the robot's face-tracker algorithm (reachy_mini `vision/face_tracking.py`), with YOLOX in place of YuNet. The detector runs on every frame, and the box nearest the last one is kept. A box more than a quarter of the frame away counts as a miss. The centre is smoothed: slowly while it jitters, faster when it moves. The box centre steers the head: the head turns alone up to 25° from the body, then the body turns too.
3. **Detector misses a frame** (a hand over the phone, the phone edge-on). An OpenCV CSRT tracker takes over from the last frame where the detector saw the object, for up to 20 frames (about 1 s). Those frames run at about 10–13 Hz. CSRT runs only on missed frames.
4. **Lost.** After 20 frames without a detection, the object is dropped. Reachy then takes the object of that class nearest where it was last seen. The face tracker takes the largest face, but that could be someone else's phone.
4. **Give up.** If the object is unseen for 10 s (60 s if it was never seen), Reachy stops tracking, prints `(lost the object, stopped tracking)` and eases back to the front.

## Debug view

```bash
uv run reachy-demo agent --mic --robot --debug-view
```

Open http://127.0.0.1:8090/ (see [face-tracking.md § Debug view](face-tracking.md#debug-view)). While an object is tracked, the page also shows:

- **blue box**: YOLOX found the object on this read. The label is its class and detector confidence (0–1), for example `cell phone 0.62`.
- **yellow box**, label ending in `csrt`: YOLOX missed the object on this read, and CSRT covers the gap. The confidence shown is from YOLOX's last detection.
- **cross of the same colour**: the smoothed aim point the head steers by.

No box means it is not seen right now. A box older than 0.5 s is not drawn. Many yellow boxes mean YOLOX often misses the object: try a bigger model (`OBJECT_MODEL`) or bring it closer.

## Limits

- One object at a time. While it tracks, other head moves, dances, mimic and face tracking are refused. Say stop first.
- A hand covering most of the phone, or the phone held edge-on, for more than about 1 s drops it. It is found again when it shows up.
- Once it is dropped, the nearest phone wins. With two phones close together, that can be the other one. The vision model only picks at the start.
- When the object leaves the view, CSRT can follow the background for up to 1 s before it gives up.
- In owner mode, only the owner can start tracking. Anyone can stop it.
- When the object is lost for 10 s, Reachy prints it but doesn't say it aloud.

## Errors

| You hear / see | Meaning and fix |
|---|---|
| `tracking_unavailable: object model missing: ...` | `OBJECT_MODEL` is a path to a file that isn't there: fix it, or use a size name |
| `tracking_unavailable: object model ... download failed: ...` | No internet, or GitHub unreachable: retry, or download the file by hand into `models/yolox/` |
| `tracking_unavailable: ... is not a YOLOX COCO model` | `OBJECT_MODEL` points to a non-YOLOX model |
| `unsupported_object: ...` | That object is not one of the 80 classes |
| `target read failed (skipping)` in the log | One camera frame failed. Tracking keeps going |
| The box is on the object in the debug view, but the head swings past it or turns the wrong way | `CAMERA_LAG_S` doesn't match the real delay. Measure it with `scripts/camera_lag.py` |
| `config error: CAMERA_LAG_S must be 0-2 seconds` | Fix `CAMERA_LAG_S` in `.env` |

## For developers

- `perception/objects.py`: `ObjectDetector` (YOLOX-nano, Apache-2.0, onnxruntime, 2 threads, about 16 ms per 1280×720 frame on the laptop) and `ObjectSource` (pick, then detect on every half-size frame, about 27 ms). `BoxTracker` and `CenterFilter` port the SDK's `Tracker` and `_AdaptiveCenterFilter` with the same constants. Two changes: after a loss `BoxTracker` re-picks the box nearest the last sighting, not the largest, and `bridge()` accepts CSRT boxes on missed frames (start ~19 ms + update ~29 ms). The daemon only aims from its own YuNet detector and has no API for an outside target, so this runs on the laptop. `attention/follow.py` steers the head from it. Each reading carries the time its frame shows (grab time minus `CAMERA_LAG_S`), and the offset is applied to the head pose of that moment. Without that time, a 0.2 s delay made the head swing 34° around a still object, and a 0.3 s delay turned it the wrong way (simulated with the real `Follow` code, 2026-11-18).
- `attention/follow.py`: `Follow` is shared with face tracking. It runs one long move in the motion loop, so stop and the safety limits work. `give_up_s` ends it when the target stays unseen.
- The detector is loaded on the first request, and both models are freed when tracking ends.
