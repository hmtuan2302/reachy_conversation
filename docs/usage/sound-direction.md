# Sound direction: "what was that?" and "turn to me"

Reachy remembers where the last loud sound and your last sentence came from, using the robot's mic array (direction of arrival, DoA). When you ask, it turns toward that spot with its head and body. If the spot is out of reach, it tells you where the sound was and asks you to check it yourself.

## Run

```bash
uv run reachy-demo agent --mic --robot      # robot at ROBOT_HOST in .env (use its IP)
```

It needs `--robot`. With `--type` there is no mic, so nothing is stored and the tool answers "nothing heard recently".

## What you can say

| You say | What Reachy does |
|---|---|
| "Turn to me", "look at me", "where am I?" | Turns to where your voice came from in your last sentence |
| "What was that?", "what happened?", "did something fall?" | Turns to the last loud sound and says how long ago it was |

Example after a book falls on the left:

```
you: what happened?
bot: I heard something about 60 degrees to my left 4 seconds ago. Turning to look.
```

Out of reach (roughly straight behind):

```
bot: The sound came from behind me, but that's out of my reach. Can you check it?
```

## What is stored

Only the **latest** of each kind, in memory (lost on exit):

- **Voice**: one direction, averaged over your last sentence (local VAD start to end). Kept for **30 s**.
- **Loud sound**: one direction, averaged over the 0.3 s after the sound. Kept for **2 minutes**. A new loud sound replaces the old one, and many bangs never push out your voice.

What is skipped:

- **Not loud enough:** a mic chunk below `SOUND_RMS` (default 3000).
- **Reachy's own voice:** everything while Reachy's speaker plays, plus 0.5 s of echo after.
- **Repeats:** after a loud sound, the next 1 s (one bang = one event).

## Motion safety

All moves go through the motion loop, which clamps every pose before it is sent:

| Joint | Limit here | SDK hard limit |
|---|---|---|
| Body yaw | ±120° | ±160° |
| Head yaw, measured from the body | ±45° | ±65° |

- The head turns alone when it can. The body turns only as much as needed.
- Turns run at 60°/s at most (180° takes 3 s).
- While Reachy dances, the turn is refused (`busy: dancing, say stop first`).
- `turn_head` works from the robot's front, so "turn head center" also brings the body back to 0°.

## Settings (`.env`)

| Variable | Default | What it does |
|---|---|---|
| `SOUND_RMS` | 3000 | Mic loudness (PCM16 RMS) that counts as a loud sound. Run `uv run python scripts/mic_check.py`, drop something, and set it below that `rms` and above your speech. |

## Known limits

- **Front and back look the same.** The mic array reports a half circle (0 = left, 90° = front or back, 180° = right). A sound behind can turn Reachy to the mirror spot in front. Reachy knows this and may say so.
- **Out of reach is rare.** With the head facing front, every DoA reads within ±90°, which is always reachable. Out of reach only happens when the head was already turned far when the sound was heard.
- **Loudness comes from the laptop mic.** The direction comes from the robot's mics, but "is it loud?" uses the laptop mic. Keep the laptop near the robot.
- **Assumes the mic array is in the head** **[verify]**. The DoA angle is added to the head yaw at read time.
- **Shouting counts as a loud sound** too.
- **Very short clicks can be missed**, because DoA updates at 10 Hz.
- **If DoA is unreachable** (daemon down, no mic array), the app warns once, keeps running, and the tool answers "nothing heard recently".

## Check it on the robot

1. Clap at the robot's left, then front, then right. Ask "what was that?" each time: it should turn the right way.
2. Walk out of view to one side and say "turn to me".
3. Say something while Reachy talks: its own voice must not become a "loud sound".
4. Wait over 2 minutes after a clap, then ask "what was that?": it should say it heard nothing recently.
5. For out of reach: clap at its far left and ask "what was that?" (it turns about 90° left). Clap again at its new far left, which is behind where it started, and ask again. It should say the sound is out of its reach.

## Troubleshooting

| Problem | Fix |
|---|---|
| "nothing heard recently" right after a bang | Lower `SOUND_RMS` (see `scripts/mic_check.py`), and check `http://<ROBOT_HOST>:8000/api/state/doa` returns an angle |
| Talking counts as a loud sound | Raise `SOUND_RMS` above your speech `rms` |
| Turns the wrong way | Check the angle at the URL above: clap left, it should be near 0 |
| `DoA read failed` warning | Daemon not reachable at `ROBOT_HOST:8000`; the rest of the app still works |
