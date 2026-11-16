"""Dance moves from `reachy_mini_dances_library` (Pollen Robotics, Apache-2.0), as offsets.

Like reachy_mini_conversation_app `tools/dance.py`, the moves come from the library's
`AVAILABLE_MOVES`. Here each one becomes an offset function for robot/motion.py, which plays it
in our one motion loop (clamped, stoppable) instead of the SDK's own `play_move`.
The library imports the SDK, so it is the optional `robot` extra; without it there are no dances.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from reachy_demo.robot.motion import Offset, Pose

log = logging.getLogger(__name__)

BPM = 114.0  # the library's DanceMove.default_bpm


@dataclass(frozen=True)
class Dance:
    description: str
    pass_s: float  # one pass of the move
    offset: Offset


def _offset(fn: Callable[..., Any], params: dict[str, Any]) -> Offset:
    def offset(t: float) -> Pose:
        o = fn(t * BPM / 60.0, **params)  # the library counts time in beats
        x, y, z = map(float, o.position_offset)  # numpy -> plain floats
        roll, pitch, yaw = map(float, o.orientation_offset)
        a0, a1 = map(float, o.antennas_offset)
        return {
            "x": x * 1000.0,  # m -> mm
            "y": y * 1000.0,
            "z": z * 1000.0,
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
            "antenna_0": math.degrees(a0),
            "antenna_1": math.degrees(a1),
        }

    return offset


def load_dances() -> dict[str, Dance]:
    """All library moves by name; empty (with a warning) when the library isn't installed."""
    try:
        from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    except ImportError as e:
        log.warning("dance library not available, no dances: %s", e)
        return {}
    return {
        name: Dance(
            description=meta.get("description", ""),
            pass_s=meta.get("default_duration_beats", 1) * 60.0 / BPM,
            offset=_offset(fn, dict(params)),  # a copy: the library's dicts are shared
        )
        for name, (fn, params, meta) in AVAILABLE_MOVES.items()
    }
