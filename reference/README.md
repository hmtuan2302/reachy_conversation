# Reference: tools from `reachy_mini_conversation_app`

Verbatim copies, **not part of `reachy_demo`**: they aren't imported, packaged, linted, type-checked or tested. They're kept here to port from, one tool at a time, into `src/reachy_demo/agent/tools/` against our own registry (see `docs/superpowers/specs/2026-11-16-agent-tools-design.md`).

- Source: `reachy_mini/opensource/apps/reachy_mini_conversation_app/src/reachy_mini_conversation_app/tools/`
- Commit: `e90192559149d6281d782c1c4aed39bb7d84bdef` (2026-11-04)
- Copied: 2026-11-16, unmodified
- License: Apache-2.0, see [LICENSE.reachy_mini_conversation_app](LICENSE.reachy_mini_conversation_app). Copyright Pollen Robotics.

They won't run as-is: they import `core_tools.Tool/ToolDependencies`, the app's config, MCP client, tool spaces, profiles, `moves.MovementManager`, `dance_emotion_moves` and the `reachy_mini` SDK, none of which are copied.

| File | Tool(s) | Needs to run | Port phase |
|---|---|---|---|
| `remember.py`, `forget.py` | remember, forget | app `memory.py` (JSON facts) | 1 |
| `dance.py`, `stop_dance.py` | dance, stop_dance | robot, MovementManager, dance moves | 3 |
| `play_emotion.py`, `stop_emotion.py` | play_emotion, stop_emotion | robot, emotion moves | 3 |
| `move_head.py`, `sweep_look.py`, `head_tracking.py`, `idle_do_nothing.py` | head motion | robot, MovementManager | 3 |
| `camera.py` | camera | robot camera + vision model | 3 |
| `volume_control.py`, `robot_status.py`, `go_to_sleep.py` | robot controls | robot SDK / daemon | 3 |
| `task_status.py`, `task_cancel.py`, `background_tool_manager.py` | background tasks | app tool manager | later |
| `core_tools.py`, `tool_constants.py`, `__init__.py` | base classes, registry, constants | app config, MCP, profiles | design reference only |

Not files: the remote tools (`search_web`, `get_time`, `get_weather`, `get_info`) run on Pollen's Hugging Face Spaces over MCP and are only described in the app's `docs/TOOL_AUGMENTED_ASSISTANCE.md` §5.
