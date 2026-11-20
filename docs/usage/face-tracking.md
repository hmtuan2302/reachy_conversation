# Face tracking: "track me", "follow me"

Reachy follows your face with its head and body until you say stop. It uses the same tracking as [object tracking](object-tracking.md), with a face detector in place of the object detector. Without a camera, it follows your voice instead (direction of arrival, DoA).

## Run

```bash
uv run reachy-demo agent --mic --robot      # robot at ROBOT_HOST in .env (use its IP)
```

It needs `--robot`. With `--type` there is no mic, so it follows faces only.

## What you can say

| You say | What Reachy does |
|---|---|
| "Track me", "follow me", "keep looking at me", "theo dõi tôi" | Turns toward your voice, then follows your face. Says it follows you until you say stop |
| "Stop", "stop following me", "thôi", "dừng lại" | Stops following, and the head eases back to the front |

Any spoken stop works, as for dances (see [agent.md § Interrupting](agent.md#interrupting)). While it follows you, each sentence is checked for a stop first.

## How it moves

- **Your face**: YuNet (the SDK's face detector, ~11 ms per frame) runs on the laptop on every camera frame. It takes the largest face, then keeps the face nearest the last one. The centre is smoothed, and a CSRT tracker covers frames where the detector misses you (a hand over your face, a turned head) for up to about 1 s. It is the same path as object tracking.
- **Camera delay**: each reading is matched to where the head was when the frame was taken (`CAMERA_LAG_S`, see [object-tracking.md § Camera delay](object-tracking.md#camera-delay-once-per-robot-and-wi-fi)). That keeps the head from swinging past you.
- **Head and body**: the head turns alone up to 25° from the body, then the body turns too.
- **Voice**: when you speak from somewhere it isn't looking, it turns toward your voice first, then finds your face there.
- **Only your body in view** (back turned, face covered, or so close that your head is above the frame): YOLOX looks for a `person` and the head aims at the top of your body, tilting up when your body is cut off at the top edge. It only runs while no face is seen, and takes the body nearest where your face was.
- **You left the frame**: if your face was lost at the edge, or while moving fast, the head keeps turning that way for 1.5 s to find you again. Lost in the middle of the frame (turned away, covered), it holds.
- **Lost you**: after that, the head holds where it last saw you. Your next sentence turns it toward your voice.
- **No camera**: if the face detector doesn't start, Reachy follows your voice only. After each sentence it turns toward your voice.
- It follows for at most 30 minutes, then stops on its own.

## Limits

- Tracking, object tracking and mimic always end when the app exits, and a new session starts with none of them on. Each turn the model gets a `[robot]` line with what is running right now, so it doesn't trust old chat or memories about it. Memory keeps no trace of it: no facts, no mention in conversation summaries, and turns about tracking or mimic are not reloaded on restart.

- While it follows you, other head moves and dances are refused ("busy: dancing or following the user, say stop first"). Say stop first.
- With several faces in view, it picks the largest and stays with it. Once it loses that face for about 1 s, it takes the largest face again. A sentence from more than 35° away switches it to that speaker.
- It uses the laptop's CPU (YuNet, plus CSRT on missed frames), not the robot's.
- The mic array can't tell front from back, so a voice behind Reachy can turn it toward the mirror spot in front.
- In owner mode, only the owner can start tracking. Anyone can stop it.
- Without a camera, or if the face detector doesn't start, it follows your voice only and says it can't see.

## Debug view

```bash
uv run reachy-demo agent --mic --robot --debug-view        # or --debug-view 9000
```

Open http://127.0.0.1:8090/ in a browser (from Windows too: WSL2 forwards localhost). It needs the camera and the `perception` extra. The page shows the live camera with:

- **green boxes**: YuNet run on the laptop on this frame. They never lag the image. The number above each box is its confidence (0–1). Boxes under `FACE_MIN_SCORE` (default 0.85) are dropped. This sets only this page's boxes. Tracking uses its own YuNet with the SDK's threshold, 0.6.
- **blue/yellow box labelled `face`**: the face tracking follows, blue when YuNet found it on this read, yellow when CSRT covers a miss. The cross of the same colour is the smoothed aim point. It shows only while tracking runs ("track me").
- **grey cross**: the image centre, where tracking tries to put your face.
- **blue/yellow box and cross**: the object being tracked ("track my phone"). See [object-tracking.md § Debug view](object-tracking.md#debug-view).
- **blue/magenta box**: find object ("find my phone"): blue for a hit at a stop, magenta for the found object (until you say yes, or 60 s). See [find-object.md](find-object.md).
- top line: frames per second, and the number of faces (green boxes).
- **chat panel** (right, or below on a narrow window): the conversation as `you:` / `bot:` lines, without context blocks or status lines. It keeps the last 200 lines and refreshes every second.

**Log**: each agent run writes `data/logs/run-<date>-<time>.jsonl` (path printed at start; `LOG_DIR`, `LOG_LEVEL` change it). It has every tool call with its result, and lines like `following the user (detector on)` or `stopped following the user`. The terminal shows only warnings.

**Recording**: the view also records what it shows to `data/recordings/view-<date>-<time>.mp4`. The video is saved when you exit the app. Slow frames are repeated, so the video plays at real speed, lag included. The chat goes to the `.txt` file with the same name, with a time on each line. Delete old recordings yourself: they are not cleaned up.

If the blue box is on you but the head swings past you, `CAMERA_LAG_S` is off: measure it. If the head only trails you, that comes from tracking's gain and speed limit (`attention/follow.py`). The view runs a second YuNet on the laptop, so turn it off for demos.

## Errors

| You see | Meaning |
|---|---|
| `face detector unavailable: ...` in the log | YuNet didn't load (no `robot` extra, or its first download failed). Reachy follows your voice only |
| `target read failed (skipping)` in the log | One camera frame failed. Tracking keeps going |

## How it works (for developers)

`attention/follow.py` runs tracking as one long move in the motion loop (`robot/motion.py`), like a dance, so stop ends it and other moves are refused. Face readings come from a `perception/objects.py` `ObjectSource` with the SDK's YuNet wrapped in `BoxesDetector` (`begin("face")`, wired in `agent/cli.py`). That is the same pipeline as object tracking. Each reading carries its frame time, and the offset is applied to the head pose of that moment (2 s history). The face reader is wrapped by `attention/hints.py`: `with_body` falls back to `PersonHeads` (YOLOX `person`, the top of the box; loaded on first use, freed when tracking stops), and `chasing` keeps turning toward the side the face left by for `CHASE_S`. Both feed `Follow` ordinary readings (past ±1 means beyond the frame edge), so `Follow` itself is unchanged.

Until 2026-11-18, face tracking used the daemon's own tracker (`steer_on_robot=True`, REST `/api/media/tracking/enable`), as the reference app does. That path is still in `Follow`, but nothing uses it now. It was rougher: its target only updates with each detection and has a +15° pitch trim. Object tracking on the laptop moved more smoothly.
