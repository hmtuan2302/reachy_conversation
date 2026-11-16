"""Head tools: turn (yaw), tilt (pitch), roll at fixed safe angles, dances, and stop.

Every tool returns at once; robot/motion.py plays the moves in the background, so Reachy can
talk while it moves. Ideas from reachy_mini_conversation_app `tools/move_head.py` and
`tools/dance.py` (Apache-2.0, Pollen Robotics); the dances are that tool's library moves
(robot/dances.py).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from typing import Any

from reachy_demo.robot.dances import Dance
from reachy_demo.robot.motion import MAX_LONG_S, Motion, dance, goto

log = logging.getLogger(__name__)

# tool -> (axis, direction -> fixed angle in degrees). Signs follow the SDK:
# yaw + = left, pitch + = down, roll + = left (ReachyMini.wake_up rolls +20 "to the left").
POSES: dict[str, tuple[str, dict[str, float]]] = {
    "turn_head": ("yaw", {"left": 20.0, "right": -20.0, "center": 0.0}),
    "tilt_head": ("pitch", {"up": -15.0, "down": 15.0, "center": 0.0}),
    "roll_head": ("roll", {"left": 15.0, "right": -15.0, "center": 0.0}),
}
DESCRIPTIONS = {
    "turn_head": "Turn your head left or right (yaw), or back to center.",
    "tilt_head": "Tilt your head to look up or down (pitch), or back to center.",
    "roll_head": "Lean your head toward your left or right shoulder (roll), or back to center.",
}
DANCE = "dance"
STOP = "stop_head"
MOVE_S = 1.0

PROMPT = (
    "You have a head you can move with turn_head (left/right), tilt_head (up/down) and "
    "roll_head (lean left/right), and you can dance. Each head tool changes one axis and "
    "keeps the others. For a combined request, such as 'look right and down', call one tool "
    "per axis. To look straight ahead, set every moved axis back to center. "
    "Call every tool the request needs in the same reply. To dance longer, use repeat; "
    "never call dance twice. "
    "If the user only asked you to move, the move is the answer: write no text. "
    "If the user also asked you to say or answer something (a joke, a question), "
    "set reply to true on the tool call, and you will answer while moving. "
    "While you dance, other moves are refused until the user says stop. "
    "If the user tells you to stop moving or dancing, call stop_head."
)
REPLY = {
    "type": "boolean",
    "description": "true if the user also asked you to say or answer something, "
    "so you speak while moving",
}


class HeadTools:
    """Turns tool calls into moves for the motion loop. Validation lives here."""

    def __init__(
        self, motion: Motion, dances: Mapping[str, Dance] | None = None, move_s: float = MOVE_S
    ) -> None:
        self._motion = motion
        self._dances = dict(dances or {})  # no dances: the dance tool isn't offered
        self._move_s = move_s

    @property
    def busy(self) -> bool:
        return self._motion.busy

    def stop(self) -> None:
        self._motion.stop()

    async def close(self) -> None:
        await self._motion.close()

    def specs(self) -> list[dict[str, Any]]:
        def fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> Any:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

        moves = [
            fn(
                name,
                DESCRIPTIONS[name],
                {"direction": {"type": "string", "enum": list(angles)}, "reply": REPLY},
                ["direction"],
            )
            for name, (_, angles) in POSES.items()
        ]
        stop_spec = fn(
            STOP,
            "Stop moving: end a dance now and cancel moves that have not started.",
            {},
            [],
        )
        if not self._dances:
            return [*moves, stop_spec]
        catalog = "\n".join(f"{n}: {d.description}" for n, d in self._dances.items())
        max_repeat = int(MAX_LONG_S // min(d.pass_s for d in self._dances.values()))
        dance_spec = fn(
            DANCE,
            "Play a dance move in the background (one pass is 2-5 s). "
            "Other moves are refused until it ends.",
            {
                "move": {
                    "type": "string",
                    "enum": list(self._dances),
                    "description": f"the move; omit for a random one. Moves:\n{catalog}",
                },
                "repeat": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_repeat,
                    "description": "how many passes (default 1)",
                },
                "reply": REPLY,
            },
            [],
        )
        return [*moves, dance_spec, stop_spec]

    def needs_response(self, name: str, arguments: str) -> bool:
        """Moves are silent unless the model flags that the user also asked for words."""
        return _args(arguments).get("reply") is True

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        """Queue a move and return at once."""
        args = _args(arguments)
        if name == STOP:
            self._motion.stop()
            return {"status": "stopped"}
        if name == DANCE and self._dances:
            move_name = args.get("move") or random.choice(list(self._dances))  # noqa: S311
            repeat = args.get("repeat", 1)
            if move_name not in self._dances:
                return {"error": f"bad_arguments: move must be one of {list(self._dances)}"}
            if not isinstance(repeat, int) or isinstance(repeat, bool):
                return {"error": "bad_arguments: repeat must be an integer"}
            d = self._dances[move_name]
            move = dance(move_name, d.pass_s, d.offset, repeat)
            label = f"{move.name} for {move.duration_s:.1f} s"
        elif name in POSES:
            axis, angles = POSES[name]
            direction = args.get("direction")
            if direction not in angles:
                return {"error": f"bad_arguments: direction must be one of {list(angles)}"}
            move = goto(name, {**self._motion.planned(), axis: angles[direction]}, self._move_s)
            label = f"{name} {direction}"
        else:
            return {"error": f"unknown_tool: {name}"}
        if error := self._motion.submit(move):
            return {"error": error}
        log.info("queued %s", label)
        return {"status": f"started {label}"}


def _args(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
