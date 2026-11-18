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
| "Find my phone", "where is my cup?", "tìm điện thoại của tôi" | Says it will look, then turns to look around the room, stopping every 40°. When it sees the object, it turns to it. Describes it and asks: "I see a white mug on the desk, about 60 degrees to my left, low. Is that the one?" |
| "Yes", "đúng rồi" | Thanks you and turns back to where the head was before the search (`find_object` with `confirmed: true`) |
| "No", "không phải" | Goes on looking from the stop where it found it, skipping that object |
| "Stop" during the search | Stops and turns back to where the head was before the search |

A search makes **at most 2 passes** over all stops, including after a "no". Then Reachy says it searched all around and couldn't find it. Say "no" within 60 s: after that, or for a different kind of object, it starts a new search. For an object it can't recognise (keys, pen, AirPods), it says it can only find everyday objects. The objects are the same 80 classes as [object tracking](object-tracking.md#what-it-can-track).

Example (checked with DeepSeek 2026-11-17):

```
you: find my coffee cup
bot: Let me look around.
bot: I see a white ceramic mug on a wooden desk next to a laptop, about 60 degrees to my left and low, below my eye level. Is that the one?
you: không phải
bot: Để tôi tìm tiếp.        (looks on)
```

## How it searches

1. **Stop and look.** The head and body turn to 9 directions one by one: front, then 40°, 80°, 120° and 160° to the left, then 40° to 160° to the right. The camera sees about 88° across, so neighbouring views overlap. The body takes most of each turn, so the head stays within 40° of the body. The head stays level (pitch 0°), so the camera sees about 28° above and below eye level. A low table close by can fall below the view. Only about 30° straight behind is never seen.
2. **Look on a still frame.** At each stop it waits for the head to settle plus the camera delay (`CAMERA_LAG_S`, about 0.65 s in all), then runs YOLOX on the whole frame plus 9 overlapping tiles. The tiles find smaller, farther objects. Frames taken while moving were blurred, and over Wi-Fi the video broke up into blocks, so a phone in plain view was missed. A still frame has neither problem. That's why it no longer looks while turning.
3. **Check.** A single confident hit is accepted. In these cases the vision model checks the same frame first: several candidates, "**my** phone", or a weak score. The log shows `vision pick for ...` with the vision model's answer. The question tells it to ignore "my" (a photo can't show who owns something) but use colour, brand and other details. If the vision model can't be reached (for example HF rate limit `429`), the detector's box is trusted instead of being thrown away.
4. **Report or go on.** Accepted: Reachy centres the object, the vision model describes it in one sentence (1–3 s), and Reachy reports it with the direction from its own point of view and the height (low, eye level, high), then asks if it's the one. It keeps looking at the object, with the magenta box in the debug view, until you answer. "Yes" turns it back to where it was before the search. With no answer for 60 s it turns back by itself, unless something else has moved the head since. Rejected by the vision model (a look-alike, like a remote for a phone; log: `vision said no to N boxes`), or you say "no": that direction is skipped, and it goes on to the next stop.
5. **Second pass.** If no stop shows it, it goes round once more (you may have moved it into view). After 2 passes it gives up.

Timing: about 1.5 s per stop (turn, settle, detect), so about 3–5 s when the object is in front (with the description) and about 20 s per pass, plus 1–3 s for each vision check. A bigger `OBJECT_MODEL` adds detection time at every stop.

**Camera delay:** `CAMERA_LAG_S` (see [object tracking](object-tracking.md)). If it misses objects that are in view, or centres them badly, measure it with `uv run python scripts/camera_lag.py`.

**Debug view:** with `--debug-view` (see [face-tracking.md](face-tracking.md)), a hit at a stop shows as a **blue** box with the class and score. Once found and centred, the object gets a **magenta** box while it is described. The box stays until you say yes, or 60 s pass. The camera has one reader in `robot/adapter.py` that all callers share, so the view no longer stutters while the search runs.

## Limits

- No memory yet: each new search looks all around again, and it doesn't remember where it saw things.
- The yes/no answer is up to the LLM. It doesn't call a tool after "yes" (checked with DeepSeek), but the code doesn't enforce that.
- Directions are from Reachy's point of view ("to my left"), not yours.
- Small or far objects: YOLOX-nano needs the object to be about 50×85 px or more in a 1280×720 frame. A phone is roughly that size at 1.5–2 m; farther away, it may be missed. A bigger model (`OBJECT_MODEL=yolox_s.onnx`) sees farther but is slower.
- Something blocking the view, or an object behind the robot (that ~30°), is never seen.
- While it dances, follows you or mimics, find is refused. Say stop first.
- In owner mode, only the owner can start a search.

## Errors

| You hear / see | Meaning and fix |
|---|---|
| `find_unavailable: object model missing: ...` | Fix `OBJECT_MODEL`, or use a size name (`nano`, `tiny`, `s`...), which downloads by itself |
| `find_unavailable: object model ... download failed: ...` | No internet: retry, or download it by hand (see [object-tracking.md § Setup](object-tracking.md#setup-once)) |
| `unsupported_object: ...` | That object isn't one of the 80 classes |
| `busy: dancing or following the user, say stop first` | Another long move holds the head |

## For developers

- `attention/search.py`: `Search` turns to each of `STOPS` with a short move, waits for a still frame, detects, checks a hit, and centres it. A resume goes on from `_Run.stop`. A spoken stop cancels the running answer, and so the search. The head then turns back to where it was before the search, and the detector is freed.
- `perception/objects.py`: `detect_tiled` (frame + 3×3 tiles, merged with NMS) and `choose` (the vision model's numbered-box pick, shared with object tracking).
- `agent/tools/find.py`: the `find_object` tool.
