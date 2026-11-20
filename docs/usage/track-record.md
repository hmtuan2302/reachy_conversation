# Recording the VLA object-tracking dataset

For faces, see [track-record-face.md](track-record-face.md) (`--target face`, the default).

`scripts/track_record.py` records raw episodes for the "keep looking at the white mug" policy (plan: [2026-11-18-vla-tracking-pipeline.md](../research/2026-11-18-vla-tracking-pipeline.md) §6). In each episode you move one object around while the head moves in one of two ways. The recorder saves the camera video, the head pose at 50 Hz and the scenario. It saves no labels: they are made offline later from the object's position in each frame (§7). So the head does not have to track well while you record.

Target: **4 sessions × ~60 episodes** (~240 episodes, about 1 hour of your time per session). Start with a single session and run it through the whole pipeline before you record more.

## Setup (once)

```bash
uv sync --inexact --extra robot --extra perception   # SDK + OpenCV + YOLOX
mkdir -p data/track
cp scripts/track_objects.example.json data/track/objects.json   # then edit it for your props
```

`.env` needs `ROBOT_HOST`. It uses the same `CAMERA_LAG_S` (run `scripts/camera_lag.py` first) and `OBJECT_MODEL` as object tracking ([object-tracking.md](object-tracking.md)).

### The prop list (`data/track/objects.json`)

```json
{"id": "mug_white", "coco": "cup", "phrasings": ["track the white mug", "keep looking at the mug", "follow the white cup"]}
```

| Field | Meaning |
|---|---|
| `id` | Stable name, never reused for another object |
| `coco` | Its COCO class if YOLOX knows it (`cup`, `bottle`, `cell phone`, ...), else `null`. Only COCO objects can use the tracker head mode |
| `phrasings` | 3–5 ways to ask for it. Add colour/position variants ("the mug on the left") for objects that have a look-alike |
| `specific` | Optional: phrasings that tell it apart from its look-alike ("the mug on the left"). Used instead of `phrasings` when the episode has a distractor |
| `held_out` | `true` for the 3 test objects. They are never picked in normal sessions, only with `--held-out` |

Pick 10–15 training objects and 3 held-out ones. Mix them: about half COCO, half not (tape roll, toy, stapler). Use objects of different sizes and colours, and include two that look alike (two mugs). **Keep the held-out objects in a separate box** so they never show up in a training episode by accident, not even in the background.

## Recording a session

1. Stop the agent app (only one program can own the head).
2. Put Reachy on its taped spot. Set the session's lighting and background, and keep them the same for the whole session.
3. Run:

```bash
uv run python scripts/track_record.py --target object --targets data/track/objects.json \
    --lighting daylight --background clean_desk --calib-id calib_2026-11-20
```

4. For each episode the script prints the scenario, for example:

```
next: mug_white "follow the white cup" | fast_pass by string, start off_centre | distractor similar_object | range head | head baseline_jolt
```

   Set it up (object in its start position, distractor placed), then press **Enter**. It records for 25 s (`--seconds`), and Enter again stops early. Act out the scenario the whole time.

| Key (then Enter) | Does |
|---|---|
| *(empty)* | Record this scenario |
| `r` | Draw another scenario (one you can't set up now, e.g. turntable not at hand) |
| `d` | Delete the episode just recorded |
| `n some text` | Note saved with the next episode ("sun came out", "mug fell") |
| `q` | Quit. The head goes to sleep and the motors turn off |

Ctrl+C is safe too. An episode cut off mid-recording has no `meta.json` and is ignored. Delete its folder when you see one.

### Session options

| Flag | Default | |
|---|---|---|
| `--lighting` | required | `daylight`, `lamp` (lamp only), `backlit` (window behind the object), `dim` |
| `--background` | required | `clean_desk`, `cluttered_desk`, `other_room` |
| `--seconds` | 25 | Episode length. Keep it 20–30 |
| `--session` | `session_<today>_<letter>` | Folder name under `data/track/raw/` |
| `--calib-id` | none | Camera calibration in use (§6.7). Can be left empty until calibration exists, since raw data keeps everything |
| `--held-out` | off | Evaluation session: held-out objects only |

Change lighting or background **between** sessions: across 4 sessions, cover every lighting value and every background at least once (e.g. `daylight+clean`, `lamp+cluttered`, `backlit+clean`, `dim+other_room`).

## What the scenario fields ask you to do

The sampler balances every field against **all sessions already on disk**, so values you haven't recorded much come up first. Don't skip hard ones with `r`: skipping is how datasets end up with gaps.

| Field | Values: what to do |
|---|---|
| `motion` | `still`: put it down and leave it. `slow_slide`: slide it slowly across the desk. `fast_pass`: sweep it through the view in about 1 s, a few times. `circle`: circles 20–40 cm wide. `leave_return`: take it out of view for 2–5 s, bring it back somewhere else. `carried`: pick it up and walk it around, at different heights |
| `mover` | Move it with your `hand`, on a `string`, on a `toy_car`, or put it on the `turntable` |
| `distractor` | `none`. `similar_object`: a look-alike nearby (the other mug). `person_behind`: someone walks behind the object |
| `start` | Where the object is when you press Enter: `centred`, `off_centre`, `half_out` (half outside the frame) |
| `range` | `head`: stay within about ±60° in front of Reachy. `around`: also go well to the side of the robot, so the body turns |
| `HIDE IT` | Negative episode (~1 in 10): the named object must **not** be anywhere in view for the whole episode. Move other objects around instead |

Head modes, chosen for you: **`baseline_jolt`** (COCO objects): the existing object tracker follows the object, but every 2–5 s it gets a fake ±10° kick, so the data contains recoveries. **`sweep`** (non-COCO objects and negatives, plus ~40% of COCO episodes): the head and, for `around`, the body sweep slowly whatever you do. Keep the object in view as best you can.

## Rules for good data

- **The named object must be the one you move.** If you move the wrong one, delete the episode (`d`).
- **Keep it in view most of the time,** except in `leave_return` and `half_out`, where leaving is the point.
- **Don't block the camera** with your arm for long. Move from the side or from behind the object.
- **Change how you move between episodes:** speed, distance (30 cm to 2 m), height. Ten identical slides teach less than three different ones.
- **No people's faces close to the camera** unless they agreed to be recorded (the plan's privacy rule: data stays local).
- **Watch the fps line** after each episode: `saved ep_0007: 680 frames (27.2 fps)`. Under 10 fps prints a warning, usually bad Wi-Fi. Delete those episodes and record them again.
- **Note anything odd** with `n ...` before the next episode, so you can find it during review.

## After the session

```bash
ls data/track/raw/                     # one folder per session
python - <<'EOF'                        # coverage: episodes per value, across all sessions
import json, glob, collections
m = [json.load(open(p)) for p in glob.glob("data/track/raw/*/ep_*/meta.json")]
for k in ("object_id", "motion", "mover", "distractor", "start", "range", "head_mode", "lighting", "background", "negative"):
    print(k, dict(collections.Counter(x[k] for x in m)))
EOF
```

- Watch 3–4 random `frames.mp4` files to check the image is sharp and the object is visible.
- Any value under 5% of episodes: record more of it next session (the sampler already favours it).
- Back up `data/track/raw/`. It is git-ignored, and every later step (labelling, dataset build) is rerun from it.

## Output (per episode)

```
data/track/raw/session_2026-11-20_a/ep_0001/
  frames.mp4     # camera frames as received (full resolution, mp4v)
  frames.jsonl   # {"i": frame index, "t_arrival": monotonic time it arrived on the laptop}
  poses.jsonl    # 50 Hz: {"t", "roll", "pitch", "yaw", "body_yaw", "cmd_yaw", "cmd_pitch", "cmd_roll", "cmd_body_yaw"} in degrees
  meta.json      # scenario + lighting, background, session, calib_id, camera_lag_s, t0, duration_s, frames, frame_size, notes
```

All times are on the laptop's monotonic clock. `t_arrival` is when the frame arrived, not when it was taken. Subtract `camera_lag_s` (stored in `meta.json`) to estimate the capture time. Measured `yaw`/`pitch`/`roll` is the SDK's last reading of the head (yaw in the world frame, as in the motion code). `cmd_*` is the last pose the motion loop sent.

## Limits and not built yet

- **No teleop (mouse) mode yet** (plan §6.2: 20% of episodes). Non-COCO objects only get sweeps.
- Frame times are arrival times polled every ~3 ms, not camera capture times. The SDK gives none (spike §5).
- Body yaw is read as head joint 0 **[verify on the robot]**: check that `body_yaw` in `poses.jsonl` follows `cmd_body_yaw` in an `around` episode.
- Video is mp4v (lossy). If the labeller struggles with compression, switch the writer to a lossless codec.
- Calibration (`track_calibrate.py`), labelling and the LeRobot conversion are separate steps, not built yet.
