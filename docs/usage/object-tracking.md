# Object tracking: "track my iPhone", "follow the cup"

Reachy follows an object with its head and body until you say stop: a phone in your hand, a cup, a bottle, a book. It finds the object once, then follows it every frame.

## Setup (once)

```bash
uv sync --inexact --extra perception      # OpenCV (keeps your other extras)
mkdir -p models/yolox
curl -L -o models/yolox/yolox_nano.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
```

### Another model size

Set `OBJECT_MODEL` in `.env` to any YOLOX COCO ONNX from the [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0). The input size is read from the model.

| Model | Input | Size | Notes |
|---|---|---|---|
| `yolox_nano.onnx` (default) | 416 | 3.6 MB | About 16 ms per frame on the laptop |
| `yolox_tiny.onnx` | 416 | 20 MB | More accurate, about 2x slower |
| `yolox_s.onnx` | 640 | 36 MB | Better on small objects (a phone far away), about 5x slower |

```bash
curl -L -o models/yolox/yolox_tiny.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
# .env
OBJECT_MODEL=models/yolox/yolox_tiny.onnx
```

Only YOLOX models work. Ultralytics YOLOv8/YOLO11 exports have a different output format and are refused at load: `... is not a YOLOX COCO model`.

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

If the object isn't in view, Reachy asks you to hold it in front of it. Hold it up at about 0.5–1.5 m.

## What it can track

The 80 everyday object classes the detector knows (COCO), for example: cell phone, cup, bottle, wine glass, book, remote, laptop, keyboard, mouse, scissors, teddy bear, banana, apple, sports ball, backpack, umbrella, clock, vase.

Not supported yet: keys, pens, AirPods, glasses, wallets, or any other object outside those classes. The LLM picks the class for what you said (a mug is `cup`, an iPhone is `cell phone`). When no class is that kind of object, it picks `none` instead of a look-alike (so keys are never tracked as a `remote`), and Reachy says it can only track everyday objects.

## How it works

1. **Find (once, under 1 s; 1–3 s with several candidates).** YOLOX-nano detects every object of that class in one frame. If there are several (your phone and another on the table), the frame is sent to the vision model (`VISION_PROVIDER`) with numbered boxes: "which box is my iPhone?" If that fails, the most confident box wins.
2. **Follow (about 15–25 Hz).** An OpenCV CSRT tracker follows that box in every frame. The box centre steers the head exactly like face tracking: the head turns alone up to 25° from the body, then the body turns too.
3. **Check (every 1 s).** The detector runs again and corrects the tracker's drift (phones are smooth and shiny). If the tracker loses the object, the detector looks for it near where it was last seen.
4. **Give up.** If the object is unseen for 10 s, Reachy stops tracking, prints `(lost the object, stopped tracking)` and eases back to the front.

## Limits

- One object at a time. While it tracks, other head moves, dances, mimic and face tracking are refused. Say stop first.
- A hand covering most of the phone, or the phone held edge-on, can lose it for a moment. The 1 s check usually finds it again.
- In owner mode, only the owner can start tracking. Anyone can stop it.
- When the object is lost for 10 s, Reachy prints it but doesn't say it aloud.

## Errors

| You hear / see | Meaning and fix |
|---|---|
| `tracking_unavailable: object model missing: ...` | Download the model (see Setup), or fix `OBJECT_MODEL` |
| `tracking_unavailable: ... is not a YOLOX COCO model` | `OBJECT_MODEL` points to a non-YOLOX model |
| `not_visible: no cell phone in view` | Hold the object in front of the robot and ask again |
| `unsupported_object: ...` | That object is not one of the 80 classes |
| `target read failed (skipping)` in the log | One camera frame failed. Tracking keeps going |

## For developers

- `perception/objects.py`: `ObjectDetector` (YOLOX-nano, Apache-2.0, onnxruntime, 2 threads, about 16 ms per 1280×720 frame on the laptop) and `ObjectSource` (pick, then CSRT about 16 ms on a half-size frame, check every second).
- `attention/follow.py`: `Follow` is shared with face tracking. It runs one long move in the motion loop, so stop and the safety limits work. `give_up_s` ends it when the target stays unseen.
- The detector is loaded on the first request, and both models are freed when tracking ends.
