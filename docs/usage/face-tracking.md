# Face tracking: "track me", "follow me"

Reachy follows you with its head and body until you say stop. It keeps your face in the middle of the camera view. When it can't see you, or someone else starts talking from another spot, it turns toward the voice it heard (direction of arrival, DoA).

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

- **Your face**: the robot's own face detector (YuNet, running on the robot, not on the laptop) reports where your face is in the camera view, about 10 times a second. Reachy turns and tilts to keep it centred.
- **Head or body**: the head turns alone up to 25° from the body. Past that the body turns too (body up to ±120°, head up to 45° from the body). Turns are limited to 90°/s.
- **Your voice**: after each sentence, Reachy turns toward your voice in these cases:
  - no face has been seen for 1 s, for example on start or after you walked out of view;
  - the voice is more than 35° from where it looks, for example when someone else talks.

  For 4 s after that turn, faces more than 25° away from the voice are ignored, so it doesn't jump back to the person it looked at before.
- **Lost you, silent**: it holds still where it last saw you. Say something and it turns to your voice.
- It follows for at most 30 minutes, then stops on its own.

## Limits

- While it follows you, other head moves and dances are refused ("busy: dancing or following the user, say stop first"). Say stop first.
- With several faces in view, the robot's detector picks one, and your voice decides between people only when you speak from more than 35° away.
- The mic array can't tell front from back, so a voice behind Reachy can turn it toward the mirror spot in front.
- In owner mode, only the owner can start tracking. Anyone can stop it.
- Without a camera, or if the robot's detector doesn't start, it follows your voice only and says it can't see.

## Errors

| You see | Meaning |
|---|---|
| `face detector unavailable` in the log | The daemon's `/api/media/tracking/enable` failed. Reachy follows your voice only |
| `face read failed (voice only for now)` | One face read over Wi-Fi failed. Tracking keeps going |

## How it works (for developers)

`attention/follow.py` runs tracking as one long move in the motion loop (`robot/motion.py`), like a dance. So the motion loop stays the only sender of poses, every pose is clamped, and stop ends it. The daemon's detector is turned on with blend weight 0.01: weight 0 would pause the detector, and at 0.01 the daemon's own aim moves the head by less than a degree. The detector is turned off when tracking ends and on exit. Face readings and the on/off switch go over the daemon's REST API (`/api/media/tracking/face`, `/enable`, `/disable`), from `robot/adapter.py`.
