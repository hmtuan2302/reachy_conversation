# Startup greeting

Date: 2026-11-19. Status: approved.

## Intent

When `reachy-demo agent` has finished all startup (robot, camera, models, speech-to-text), Reachy says once: "Hi, I'm Reachy Mini, nice to meet you." It shows the audience that the robot is ready.

## Goals

- The greeting comes only after everything has loaded, including the local Whisper model, which today loads lazily on the first listen.
- It plays like an answer: `bot: Hi, I'm Reachy Mini, nice to meet you.` is printed, then spoken. A spoken stop cuts it off, and other speech heard during it is dropped, as with any answer.
- `--no-speak` prints it only. `--type` gets it too.

## Non-goals

- Antenna or head movement during the greeting.
- A `.env` setting for the text, or another language.
- Storing it in memory or chat history.
- A greeting when a face appears. The design doc already rejects that, and this is a different thing.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **A (picked)** | Load local STT up front in `run`. A `greeting` parameter on `converse` is played through the answer path once listening runs. | The mic hears the greeting while listening runs, but barge-in handles that as it does for every answer. |
| B | Speak the greeting before `converse` starts. | The mic buffers Reachy's own voice, and it is transcribed afterwards as the user's turn. |

## Design

- `agent/cli.run`: when `STT_PROVIDER=local` and input is audio, call `models_from_env(os.environ)` before the conversation and pass `models=` to `transcribe_local`. An `SttError` prints `error: <msg>` and exits 2.
- `converse(..., greeting: str | None = None)`: once the listener runs, play `greeting` through the answer path. There is no "you:" line and no `on_answer`. With `classify`, barge-in applies. `on_line("bot", greeting)` feeds the debug view.
- `GREETING = "Hi, I'm Reachy Mini, nice to meet you."` in `agent/cli.py`, passed in both the robot and the no-robot paths.
