"""mimic_me: Reachy copies the user's head moves like a mirror until they say stop (F6)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from reachy_demo.attention.mimic import Mimic
from reachy_demo.memory.store import OWNER, Speaker

log = logging.getLogger(__name__)

NAME = "mimic_me"
NOT_OWNER = {"error": "not_owner: only the owner can ask for that right now"}
PROMPT = (
    "You can copy the user's head moves like a mirror. When they ask you to mimic or copy "
    "them ('mimic me', 'copy me', 'bắt chước tôi'), call mimic_me with enabled true, then say "
    "one short sentence that you copy them until they say stop. If they ask you to stop "
    "copying, call mimic_me with enabled false."
)


class MimicTools:
    def __init__(self, mimic: Mimic, speaker: Callable[[], Speaker] = lambda: OWNER) -> None:
        """`speaker`: who said the current turn; only the owner may start mimic (owner mode)."""
        self._mimic = mimic
        self._speaker = speaker

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": NAME,
                    "description": "Start or stop copying the user's head moves like a mirror "
                    "(turn, nod, tilt, lean) with the camera. Runs until stopped.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {
                                "type": "boolean",
                                "description": "true to start copying, false to stop",
                            }
                        },
                        "required": ["enabled"],
                    },
                },
            }
        ]

    def needs_response(self, name: str, arguments: str) -> bool:
        return True  # confirm, and say how to stop

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        if name != NAME:
            return {"error": f"unknown_tool: {name}"}
        try:
            enabled = json.loads(arguments or "{}").get("enabled")
        except (json.JSONDecodeError, AttributeError):
            enabled = None
        if not isinstance(enabled, bool):
            return {"error": "bad_arguments: enabled must be true or false"}
        if not enabled:
            self._mimic.stop()
            return {"status": "stopped"}
        if self._speaker() != OWNER:
            return NOT_OWNER
        if error := await self._mimic.start():
            return {"error": error}
        return {"status": "mimicking"}
