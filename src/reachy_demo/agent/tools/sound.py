"""look_toward_sound: turn head and body toward the user's voice or the last loud sound.

The direction comes from audio/sound_events.py; the turn goes through robot/motion.py, which
clamps every pose (head 45° from the body, body 120°). Past that reach the tool does not move:
it returns where the sound was, and the agent tells the user to check it themselves.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from reachy_demo.audio.sound_events import (
    MAX_AGE_S,
    SoundEvents,
    describe,
    plan_turn,
    turn_seconds,
    wrap,
)
from reachy_demo.robot.motion import Motion, goto

log = logging.getLogger(__name__)

NAME = "look_toward_sound"
MIN_TURN_S = 1.0
# The chip reports a half circle, so a sound behind can read as its mirror in front.
MIRROR_NOTE = "The microphones cannot tell front from back, so it may be the mirror spot behind."

PROMPT = (
    "Your microphones remember where sounds came from. When the user asks you to turn or look "
    "toward them ('turn to me', 'look at me', 'where am I?'), call look_toward_sound with "
    "source speaker. When they ask about a sound ('what was that?', 'what happened?', 'did "
    "something fall?'), call it with source noise. If it returns out_of_reach, say where the "
    "sound came from, that it is out of your reach, and ask the user to check it themselves. "
    "If it returns no_recent_sound, say you didn't hear anything recently."
)


class SoundTools:
    def __init__(
        self, sound: SoundEvents, motion: Motion, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._sound = sound
        self._motion = motion
        self._clock = clock

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": NAME,
                    "description": "Turn head and body toward where a sound came from. "
                    "speaker: the user's voice in their last sentence. noise: the last loud "
                    "sound, such as something falling.",
                    "parameters": {
                        "type": "object",
                        "properties": {"source": {"type": "string", "enum": list(MAX_AGE_S)}},
                        "required": ["source"],
                    },
                },
            }
        ]

    def needs_response(self, name: str, arguments: str) -> bool:
        return True  # always say where it was, or that it is out of reach

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        if name != NAME:
            return {"error": f"unknown_tool: {name}"}
        try:
            source = json.loads(arguments or "{}").get("source")
        except (json.JSONDecodeError, AttributeError):
            source = None
        if source not in MAX_AGE_S:
            return {"error": "bad_arguments: source must be speaker or noise"}
        now = self._clock()
        event = self._sound.latest(source, now)
        if event is None:
            return {"status": "no_recent_sound", "source": source, "max_age_s": MAX_AGE_S[source]}
        result: dict[str, Any] = {
            "source": source,
            "where": describe(event.yaw),
            "seconds_ago": round(now - event.at),
            "note": MIRROR_NOTE,
        }
        planned = self._motion.planned()
        plan = plan_turn(event.yaw, planned["body_yaw"])
        if plan is None:
            log.info("sound %s out of reach", result["where"])
            return {**result, "status": "out_of_reach"}
        body, yaw = plan
        swing = max(abs(body - planned["body_yaw"]), abs(wrap(yaw - planned["yaw"])))
        move = goto(
            NAME, {**planned, "yaw": yaw, "body_yaw": body}, turn_seconds(swing, MIN_TURN_S)
        )
        if error := self._motion.submit(move):
            return {**result, "error": error}
        log.info("turning toward %s %s", source, result["where"])
        return {**result, "status": "turned"}
