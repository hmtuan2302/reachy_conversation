# Recording the VLA face-tracking dataset

`scripts/track_record.py --target face` records raw episodes for a policy that keeps a person's face centred ("look at me", "follow the person in blue"). It is the same recorder as the object version ([track-record.md](track-record.md)), with people instead of props. In each episode one person moves around while the head moves in one of two ways. The recorder saves the camera video, the head pose at 50 Hz and the scenario. It saves no labels: they are made offline later from where the face is in each frame (plan §7: [2026-11-18-vla-tracking-pipeline.md](../research/2026-11-18-vla-tracking-pipeline.md)). So the head does not have to track well while you record.

Target: **4 sessions × ~60 episodes** (~240 episodes, about 1 hour per session) with **at least 3–5 different people**. Record one session first and run it through the whole pipeline before you record more.

## Consent first

This dataset is video of people's faces. Before anyone is recorded:

- Ask them, and tell them where it is kept (`data/track_face/raw/`, local only, git-ignored).
- People who only walk through as distractors are recorded too. Ask them as well.
- Delete someone's episodes when they ask (`grep -l '"object_id": "person_b"' data/track_face/raw/*/ep_*/meta.json`).
- Don't record people who didn't agree, for example someone on a video call behind you.

## Setup (once)

```bash
uv sync --inexact --extra robot --extra perception   # SDK + OpenCV; YuNet comes with the SDK
mkdir -p data/track_face
cp scripts/track_people.example.json data/track_face/people.json   # then edit it
```

`.env` needs `ROBOT_HOST` and `CAMERA_LAG_S` (run `scripts/camera_lag.py` first), as for [face tracking](face-tracking.md).

### The people list (`data/track_face/people.json`)

```json
{"id": "person_a", "phrasings": ["look at me", "follow my face", "nhìn tôi đi"],
 "specific": ["look at the person in the blue shirt", "track person A"]}
```

| Field | Meaning |
|---|---|
| `id` | A code, not a real name (`person_a`). Keep a private list of who is who if you need one |
| `phrasings` | 3–5 general requests ("look at me", "follow my face"), in Vietnamese and English |
| `specific` | Requests that pick **this** person when someone else is in view ("the person in blue", "the man with glasses"). Used when the scenario has a distractor. Keep them true for what the person wears in that session: edit the file between sessions if needed |
| `held_out` | `true` for 1–2 people who appear only in evaluation sessions (`--held-out`), never in training ones |

## Recording a session

1. Stop the agent app (only one program can own the head).
2. Put Reachy on its taped spot, at about face height for a seated person if you can. Set the session's lighting and background, and keep them the same for the whole session.
3. Run:

```bash
uv run python scripts/track_record.py --target face --targets data/track_face/people.json \
    --lighting daylight --background plain_wall --calib-id calib_2026-11-20
```

4. For each episode the script prints the scenario:

```
next: person_a "look at the person in the blue shirt" | motion walk_across, distance mid, appearance glasses, distractor second_person, start off_centre, range head | head baseline_jolt
```

   The named person gets into the start position (and puts on glasses, the second person comes into view...). The operator presses **Enter**. It records 25 s (`--seconds`) and stops by itself. Enter again stops early.

| Key (then Enter) | Does |
|---|---|
| *(empty)* | Record this scenario |
| `r` | Draw another scenario (one you can't set up now, e.g. no hat at hand, nobody to be the second person) |
| `d` | Delete the episode just recorded |
| `n some text` | Note saved with the next episode ("B blinked out of frame", "lamp flickered") |
| `q` | Quit. The head goes to sleep and the motors turn off |

You can record alone: press Enter, then walk into place. `start` is where you are once you're in place, a second or two in.

### Session options

| Flag | Default | |
|---|---|---|
| `--lighting` | required | `daylight`, `lamp` (lamp only), `backlit` (window behind the person: the hard one), `dim` |
| `--background` | required | `plain_wall`, `cluttered_room`, `window`, `other_room` |
| `--seconds` | 25 | Episode length. Keep it 20–30 |
| `--session` | `session_<today>_<letter>` | Folder under `data/track_face/raw/` |
| `--calib-id` | none | Camera calibration in use (plan §6.7). Can stay empty until calibration exists |
| `--held-out` | off | Evaluation session: held-out people only |

Across 4 sessions, cover every lighting value and every background at least once, and change rooms at least once.

## What the scenario fields ask for

The sampler balances every field against **all sessions already on disk**, so the values recorded least come up first. Don't skip the hard ones with `r`: skipping is how datasets end up with gaps.

| Field | Values: what the named person does |
|---|---|
| `motion` | `still`: sit or stand still, looking around naturally. `small_moves`: talk, nod, shake your head, gesture, lean. `walk_across`: walk across the view, at different speeds. `approach_retreat`: walk towards Reachy and away again. `sit_stand`: sit down and stand up, more than once. `leave_return`: leave the view for 2–5 s, come back somewhere else. `turn_away`: turn to profile and your back, then face it again. `fast_exit`: leave the view quickly, to the side, up (stand up fast) or down (crouch), and stay out 2–3 s before coming back |
| `distance` | `near`: under 1 m. `mid`: 1–2 m. `far`: 2–4 m |
| `appearance` | `as_is`: as you are. `glasses`: put on glasses or sunglasses. `hat`: cap or hood. `hand_on_face`: cover part of your face now and then (hand, cup, phone) |
| `distractor` | `none`. `second_person`: someone else is in view the whole time, moving too. `face_picture`: a photo of a face (poster, phone or laptop screen) is in view |
| `start` | Where the face is when the episode starts: `centred`, `off_centre`, `half_out` (at the edge of the frame), `body_only` (stand close so only your chest/body is in view, head above the frame) |
| `range` | `head`: stay within about ±60° in front of Reachy. `around`: also walk to the side of the robot, so the body turns |
| `NEGATIVE` | ~1 in 10 episodes: the named person must stay **out of view** all episode. If the scenario has a distractor, the other person **stays in view**: these teach "that's not who I was asked to look at". Without a distractor: nobody in view |

Head modes, chosen for you: **`baseline_jolt`** (~60%): the face tracker follows the face, but every 2–5 s it gets a fake ±10° kick, so the data contains recoveries. It uses the same rules as the agent's "track me": with only a body in view it aims at the top of the body (tilts up for `body_only`), and when your face leaves by an edge or fast it keeps turning that way for 1.5 s (`fast_exit`). The camera then finds your face again, and the labeller uses that to teach the VLA where to look when the face isn't visible ([track-label.md](track-label.md#hindsight-where-to-look-when-the-face-isnt-visible)). **`sweep`** (~40%, and all negatives): the head and, for `around`, the body sweep slowly whatever happens. Keep moving as told, and the face will pass in and out of view.

## Rules for good data

- **Only the named person does the motion.** If the second person ends up closer or more central for most of the episode, delete it (`d`), unless that was the point. Keep the specific phrasing true: "the person in blue" must be the only one in blue.
- **Face the camera most of the time,** except in `turn_away` and the moments of `hand_on_face`.
- **Vary everything you're not told:** speed, height (sitting, standing, crouching), path, clothes between sessions. Ten identical walks teach less than three different ones.
- **Different people matter more than more episodes.** 5 people × 50 episodes generalises better than 1 person × 250. Vary skin tone, age, hair and glasses if you can.
- **Backlit and dim** are where face trackers fail. Make sure those sessions happen.
- **Watch the fps line** after each episode: `saved ep_0007: 680 frames (27.2 fps)`. Under 10 fps prints a warning, usually bad Wi-Fi. Delete those episodes and record them again.
- **Note anything odd** with `n ...` before the next episode.

## After the session

```bash
python - <<'EOF'   # coverage across all face sessions
import json, glob, collections
m = [json.load(open(p)) for p in glob.glob("data/track_face/raw/*/ep_*/meta.json")]
for k in ("object_id", "motion", "distance", "appearance", "distractor", "start", "range", "head_mode", "lighting", "background", "negative"):
    print(k, dict(collections.Counter(x[k] for x in m)))
EOF
```

- Watch 3–4 random `frames.mp4` to check the image is sharp and the face is visible.
- Any value under 5% of episodes: record more next session (the sampler already favours it).
- Back up `data/track_face/raw/`. It is git-ignored, and labelling and the dataset build are rerun from it.

## Output (per episode)

```
data/track_face/raw/session_2026-11-20_a/ep_0001/
  frames.mp4     # camera frames as received (full resolution, mp4v)
  frames.jsonl   # {"i": frame index, "t_arrival": monotonic time it arrived on the laptop}
  poses.jsonl    # 50 Hz: {"t", "roll", "pitch", "yaw", "body_yaw", "cmd_yaw", "cmd_pitch", "cmd_roll", "cmd_body_yaw"} in degrees
  meta.json      # target "face", person id (object_id), task_text, the scenario fields, lighting,
                 # background, session, calib_id, camera_lag_s, t0, duration_s, frames, frame_size, notes
```

`object_id` holds the person's id (the same field as for objects, so one labeller and converter handle both). Times are on the laptop's monotonic clock. `t_arrival` is when the frame arrived: subtract `camera_lag_s` to estimate the capture time.

## Limits and not built yet

- **No teleop (mouse) mode yet.** The head is driven by the face tracker with jolts, or by sweeps.
- For labelling, pick the named face out of several (`second_person`) with the `specific` phrasing, for example with an open-vocabulary detector or a VLM. YuNet alone can't tell people apart. This is a labeller task, not built yet.
- Frame times are arrival times, polled every ~3 ms. Body yaw is read as head joint 0 **[verify on the robot]**: in an `around` episode, `body_yaw` should follow `cmd_body_yaw`.
- Video is mp4v (lossy). Switch to a lossless codec if labelling struggles.
