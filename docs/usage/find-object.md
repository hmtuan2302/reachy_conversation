# Find an object: "find my phone", "tìm cái cốc"

Reachy turns its head and body to look around the room for an object. When it finds it, it turns to look at it and tells you where it is.

## Setup

Same as object tracking: `uv sync --inexact --extra perception` and the YOLOX model (see [object-tracking.md § Setup](object-tracking.md#setup-once)). `OBJECT_MODEL` picks the model for both.

## Run

```bash
uv run reachy-demo agent --mic --robot
```

It needs `--robot` and the robot's camera.

## What you can say

| You say | What Reachy does |
|---|---|
| "Find my phone", "where is my cup?", "tìm điện thoại của tôi" | Says it will look, sweeps the room, stops at the object and looks at it. Describes it and asks: "I see a white mug on the desk, about 60 degrees to my left, low. Is that the one?" |
| "Yes", "đúng rồi" | Thanks you. Nothing else: the head stays on the object |
| "No", "không phải" | Goes on sweeping from where it stopped, skipping that object |
| "Stop" during the sweep | Stops and faces the front |

A search makes **at most 2 full sweeps**, including after a "no". Then Reachy says it searched all around and couldn't find it. Say "no" within 60 s: after that, or for a different kind of object, it starts a new search. For an object it can't recognise (keys, pen, AirPods), it says it can only find everyday objects. The objects are the same 80 classes as [object tracking](object-tracking.md#what-it-can-track).

Example (checked with DeepSeek 2026-11-17):

```
you: find my coffee cup
bot: Let me look around.
bot: I see a white ceramic mug on a wooden desk next to a laptop, about 60 degrees to my left and low, below my eye level. Is that the one?
you: không phải
bot: Để tôi tìm tiếp.        (sweeps on)
```

## How it searches

1. **One smooth sweep.** The head and body turn together from the front to 160° left, then across to 160° right, and back to the front. It's like Pollen's `sweep_look`, but it looks for the object while it moves. The path is a sine: it starts and ends at rest and slows down at both sides by itself (fastest about 75°/s crossing the front). The body takes 75% of the turn, so the head stays within 40° of the body. Pitch stays at 12° down, so the camera covers eye level and a table in one pass. Only about 30° straight behind is never seen.
2. **Detect while moving.** YOLOX runs on frame after frame: the whole frame plus 9 overlapping tiles, about 130 ms, so the head moves at most about 10° between checks. The tiles find smaller, farther objects.
3. **Brake and check.** On a candidate the sweep brakes smoothly (no jolt), turns to the candidate (at most 90°/s) and takes a sharp still frame. A single confident hit is accepted. In these cases the vision model checks it first: several candidates, "**my** phone", or a weak score.
4. **Report or go on.** Accepted: Reachy centres the object, the vision model describes it in one sentence (1–3 s), and Reachy reports it with the direction from its own point of view and the height (low, eye level, high), then asks if it's the one. Rejected by the vision model (a look-alike, like a remote for a phone), or you say "no": that direction is skipped, and the sweep goes on where it braked.
5. **Second sweep.** If the first sweep ends without a match, it sweeps once more (blurred frames can miss something). After 2 sweeps it gives up.

Timing: about 2–5 s when the object is in front (with the description), up to about 40 s for 2 full sweeps, plus 1–3 s for each vision check.

**Sweep length:** `SEARCH_SWEEP_S` in `.env` sets how long one full sweep takes: 8–120 s, default 20. Longer is slower and gives sharper frames, so it finds smaller objects during the sweep. The peak speed is about 75°/s × 20 / length, for example 50°/s at 30 s. A value outside that range is a config error at startup.

## Limits

- No memory yet: each new search sweeps again, and it doesn't remember where it saw things.
- The yes/no answer is up to the LLM. It doesn't call a tool after "yes" (checked with DeepSeek), but the code doesn't enforce that.
- Directions are from Reachy's point of view ("to my left"), not yours.
- Small or far objects: YOLOX-nano needs the object to be about 50×85 px or more in a 1280×720 frame. A phone is roughly that size at 1.5–2 m; farther away, it may be missed. A bigger model (`OBJECT_MODEL=yolox_s.onnx`) sees farther but is slower.
- Something blocking the view, or an object behind the robot (that ~30°), is never seen.
- While it dances, follows you or mimics, find is refused. Say stop first.
- In owner mode, only the owner can start a search.

## Errors

| You hear / see | Meaning and fix |
|---|---|
| `find_unavailable: object model missing: ...` | Download the model, or fix `OBJECT_MODEL` |
| `unsupported_object: ...` | That object isn't one of the 80 classes |
| `busy: dancing or following the user, say stop first` | Another long move holds the head |

## For developers

- `attention/search.py`: `Sweep` (the sine path as a long move, with braking and resuming) and `Search` (detect while sweeping, check a candidate, centre it). Turns to a candidate are short moves; `Motion.stop(neutral=False)` ends the braked sweep where it is. A spoken stop cancels the running answer, and so the scan. The head then eases back to the front, and the detector is freed.
- `perception/objects.py`: `detect_tiled` (frame + 3×3 tiles, merged with NMS) and `choose` (the vision model's numbered-box pick, shared with object tracking).
- `agent/tools/find.py`: the `find_object` tool.
