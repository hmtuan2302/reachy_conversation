# Mimic: "mimic me", "bắt chước tôi"

Reachy copies your head moves like a mirror until you say stop: turning left and right, nodding up and down, tilting toward a shoulder, and leaning toward or away from it.

## Setup (once)

```bash
uv sync --inexact --extra perception      # MediaPipe + OpenCV (keeps your other extras)
mkdir -p models/mediapipe
curl -L -o models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The first time mimic starts, the SDK also downloads its face detector (YuNet, about 0.2 MB) from Hugging Face.

## Run

```bash
uv run reachy-demo agent --mic --robot
```

It needs `--robot` and the robot's camera. Without a camera, the tool is not offered.

## What you can say

| You say | What Reachy does |
|---|---|
| "Mimic me", "copy me", "bắt chước tôi" | Copies your head moves, and says it does so until you say stop |
| "Stop", "stop copying me", "thôi", "dừng lại" | Stops, and the head eases back to the front |

## How it copies you

| You | Reachy |
|---|---|
| Turn your head to your left by 20° | Turns 20° toward the same side of the room (its right), like your reflection |
| Look down 10° | Looks down 10° |
| Tilt your head toward your left shoulder | Tilts toward the same side (its right shoulder) |
| Lean 5 cm toward it | Moves its head 10 mm forward |

Angles are copied about 1:1, until the limits below. Reachy keeps facing you: if you stand to one side, it turns toward you first, and your head moves are added on top.

## Limits

| Axis | Limit | Why |
|---|---|---|
| Turn (yaw) | ±35° from facing you | The camera is in the head and sees about ±44°. A bigger turn would lose your face |
| Nod (pitch) | ±22° from facing you, never past ±30° | Same reason (camera sees about ±28° up/down). 30° is the head's safety limit |
| Tilt (roll) | ±25° | The head's safety limit |
| Lean (x) | ±20 mm | The head's safety limit |
| Body | ±120°, head at most 45° from the body | Safety limits. The body turns only if you stand far to the side |

You asked for 50–70°. That is not possible here. With the camera in the head, a 50° turn takes your face out of view, and Reachy could no longer follow you back. The head itself is limited to 30° pitch and 25° roll. Those limits cover every combination with the SDK's inverse kinematics (checked 2026-11-17).

## Smoothness

- Each measured angle goes through a one-euro filter: steady when you hold still, quick when you move.
- The motion loop moves each axis with a critically damped spring at 50 Hz. Position and speed change smoothly, and speed is capped at 150°/s. Settling takes about 0.3 s.
- If your face is lost for 1 s, Reachy slowly looks back to where you were and waits.
- Mimic stops on its own after 5 minutes.

## Other limits

- While mimicking, other head moves, dances and tracking are refused. Say stop first.
- One person at a time: it keeps copying the face it started with (the largest one at first).
- In owner mode, only the owner can start mimic. Anyone can stop it.

## Errors

| You hear / see | Meaning and fix |
|---|---|
| `mimic_unavailable: face model missing: ...` | Download the model (see Setup) |
| `mimic_unavailable: No module named 'mediapipe'` | `uv sync --inexact --extra perception` |
| `mimic frame failed` in the log | A camera frame failed once. Mimic keeps going |

## How it works (for developers)

- `perception/head_pose.py` finds your face box with the SDK's YuNet detector on a half-size frame. The box gives where you are and how far (pinhole camera model). MediaPipe's face landmarker runs on a square crop around the box: its own detector misses faces more than about 0.5 m away in a full 1280×720 frame. The head angles are measured relative to your line of sight to the camera, so they don't change when the robot's head (and camera) turns. That prevents feedback.
- `attention/mimic.py` runs as one long move in the motion loop (like a dance), so stop works and every pose is clamped.
- The model is loaded when mimic starts and freed when it ends.
- Add-on CPU: about 12 ms (YuNet) + 5 ms (landmarker) per frame.
