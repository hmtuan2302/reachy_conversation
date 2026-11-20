# Auto-labelling the face-tracking dataset

`scripts/track_label.py` labels the episodes recorded with [track-record-face.md](track-record-face.md). For every frame it writes where the **named person's** face is (`boxes.jsonl`), and the **action** the VLA should learn (`actions.jsonl`): how far to turn to look at them, even when their face isn't visible. Converting these to a LeRobot dataset is the next step (not built yet).

## Run it

```bash
uv sync --inexact --extra perception          # OpenCV (onnxruntime is already installed)
uv run python scripts/track_label.py          # all face episodes under data/track_face/raw/
```

The first run downloads InsightFace's `buffalo_l` model pack (289 MB) and keeps two files in `models/insightface/`. After that it works offline.

It runs on the CPU and uses all cores: about **1–1.5 min per 25 s episode**, so ~1–1.5 h for a 60-episode session. Leave it running. Rerunning is quick: faces found in an episode are cached in `faces.npz`, so only new episodes are processed. After you edit `people.json` or delete episodes, just run it again.

| Flag | Default | |
|---|---|---|
| `--raw` | `data/track_face/raw` | Where the sessions are |
| `--preview` | `flagged` | Write `label_preview.mp4` with boxes drawn: `flagged`, `all`, or `none` |
| `--min-score` | 0.5 | Face detector confidence. Lower (0.3) finds more dark or far faces, but also more false ones |
| `--redetect` | off | Run the detector again on every episode (after changing `--min-score`) |

## How it works

1. **Find every face** in every frame with **SCRFD**. It's much stronger than YuNet on small, dark and side-on faces; too slow for live tracking, fine offline.
2. **Link faces into tracks** frame to frame by box overlap. A track survives short misses (up to 6 frames, ~0.2 s).
3. **Recognise who is who** with **ArcFace**, a face-recognition model. It runs every 10th frame and whenever the number of faces changes.
4. **Learn each person's face** from their own episodes with nobody else in view (distractor `none`, not negative). No manual enrolment needed.
5. **Pick the named person** in every episode: the track most like their face (similarity ≥ 0.3; the same person usually scores 0.4–0.9, others under 0.2). So with a `second_person` or a `face_picture` in view, the box lands on the right face.
6. **Fill short gaps** by interpolation. Longer absences are `present = 0`. Negative episodes are `present = 0` throughout.
7. **Flag** anything that needs a human look.

**Order matters for a new person:** they need at least one episode with nobody else in view (distractor `none`) before their other episodes can be labelled with identity. The recorder's sampler produces these naturally, about 1 in 3 episodes.

## Actions (`actions.jsonl`)

From the boxes and the head pose logged at 50 Hz, every face becomes a **direction in the room**: where the camera pointed when the frame was taken, plus where the face sits in the image. Then for every frame:

| Frame | `src` | Action (`d_yaw`, `d_pitch`) |
|---|---|---|
| Face visible | `seen` | Turn toward where the face will be 0.2 s later (the robot acts that late), smoothed over 5 frames |
| Face not visible, seen again within 3 s | `hindsight` | Turn toward where it is seen next |
| Not seen again within 3 s, or a negative episode | `none` | Hold (0, 0) |

Actions are clamped to ±20° (the robot repeats them each step). `visible` (0/1) is the third action value.

### Hindsight: where to look when the face isn't visible

This is how the VLA learns "only the body is in view, so tilt up" and "the face left fast to the right, so turn right". When the recorder's tracker tilts up to your head or chases you, your face shows up again a moment later. The labeller knows where it showed up, so the frames before it get "turn there" as their label. The camera didn't need to see your face in those frames: the label comes from the future of the recording. Only `fast_exit`, `body_only` and similar episodes where the head found you again produce these labels. Sweeps produce some too, when they happen to pass you.

### Memory

Each row also has `last_d_yaw`, `last_d_pitch` (where the face was last seen, relative to the current gaze) and `since_seen_s` (how long ago, `-1` if not yet). They are inputs for the policy, not labels: an empty frame alone can't tell which way you left, but "last seen 0.3 s ago, 40° right" can. At runtime the tracker keeps them from its own past outputs.

## After it runs: review the flagged episodes

The script prints a summary and writes the list to `data/track_face/raw/flagged.txt`:

```
labelled 60 episodes, 5 flagged
  mostly_absent: 3
  identity_unsure: 1
  negative_seen: 1
```

For each flagged episode, open its `label_preview.mp4`. The green box is the named person, yellow where a gap was filled, thin grey boxes are other faces, and "not present" shows when nobody is labelled.

| Flag | Means | What to do |
|---|---|---|
| `mostly_absent` | The named person is labelled in under 25% of frames | Normal for some `leave_return`/sweep episodes; check the preview. If your face is visible but not boxed (too dark, too far), try `--min-score 0.3 --redetect`, or delete the episode |
| `identity_unsure` | Two faces look similar to the named person, or there's no gallery and several faces | Check the green box follows the right person. If not, delete the episode |
| `near_threshold` | A face scored just under the identity threshold | Check whether the named person was missed (glasses, hat, side-on) |
| `no_gallery` | The person has no episode with nobody else in view yet | Record one (distractor `none`), then rerun |
| `negative_seen` | In a "stay out of view" episode, the named person was seen | Delete the episode: its labels would be wrong |
| `jump` | The box jumped across the frame in one step | Usually a switch to another face. Check, and delete if wrong |

Delete a bad episode with `rm -r data/track_face/raw/<session>/ep_XXXX`, then rerun the labeller.

**Spot check (plan §7.4):** also watch 3–5 random **unflagged** episodes with `--preview all`. The green box must stay on the named person's face.

## Output (added to each episode folder)

| File | Contents |
|---|---|
| `boxes.jsonl` | One line per frame: `{"i", "box_xyxy": [x1, y1, x2, y2] or null, "det_score", "sim", "present": 0/1, "src": "det" / "interp" / "none"}`. Pixels in the original frame |
| `actions.jsonl` | One line per frame: `{"i", "t", "gaze_yaw", "gaze_pitch", "d_yaw", "d_pitch", "visible", "src": "seen" / "hindsight" / "none", "last_d_yaw", "last_d_pitch", "since_seen_s"}`. Degrees: yaw + = left, pitch + = down. Needs `frames.jsonl`, `poses.jsonl` and `frame_size` (all written by the recorder) |
| `label.json` | `{"label_version", "present_frac", "tracks", "target_sim", "flags": [...], "seen", "hindsight", "none"}` (the last three: share of frames per action source) |
| `faces.npz` | Cache: every face found (box, score, ArcFace vector). Delete it to force detection again |
| `label_preview.mp4` | Only for flagged episodes (or all with `--preview all`) |

## Limits

- **CPU only** (this onnxruntime has no CUDA). 1–1.5 h per session. Run it overnight or while recording the next session.
- **A person's look changes a lot between sessions** (beard shaved, new glasses): identity still usually works, but check `near_threshold` flags.
- **Twins, or a photo of the named person** as `face_picture`, will confuse identity. Don't use the named person's own photo as the face picture.
- **Angles use the nominal field of view, no lens correction**, until camera calibration exists. Faces near the frame edges get slightly wrong angles. Rerun the labeller after calibration; it's cheap thanks to the cache.
- The plan's SAM 2 tracking and click-to-fix review viewer (§7.1, §7.3) are not used. For faces, detection on every frame plus identity is enough. Bad episodes are deleted rather than fixed.
- The face models are InsightFace's: free for non-commercial research only.
