"""track_face / track_object: follow the user, or a named object, with head and body until stop.

Idea from reachy_mini_conversation_app `tools/head_tracking.py` (Apache-2.0, Pollen Robotics).
There the daemon's tracker owns the head; here attention/face_track.py steers our motion loop
from the daemon's face readings and the voice direction, so stop and the safety limits still work.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from reachy_demo.attention.follow import Follow
from reachy_demo.memory.store import OWNER, Speaker
from reachy_demo.perception.objects import COCO, ObjectSource
from reachy_demo.robot.motion import BUSY_LONG

log = logging.getLogger(__name__)

NAME = "track_face"
OBJECT = "track_object"
NONE = "none"  # the model's way out when no class fits, so it never forces a look-alike
NOT_OWNER = {"error": "not_owner: only the owner can ask for that right now"}
PROMPT = (
    "You can follow the user with your head and body. When they ask you to track, follow or "
    "keep looking at them ('track me', 'follow me', 'theo dõi tôi'), call track_face with "
    "enabled true, then say one short sentence that you are following them until they say "
    "stop. If they ask you to stop following, call track_face with enabled false. If it "
    "returns voice_only, add that you can't see right now, so you follow their voice."
)
OBJECT_PROMPT = (
    "You can also follow an object with your head ('track my phone', 'follow the cup', "
    "'theo dõi điện thoại của tôi'): call track_object with the user's words as target and the "
    "coco_class that is the same kind of thing (a mug is a cup, an iPhone is a cell phone). "
    "If no class is that kind of thing (keys, a pen, earbuds), use none, never a look-alike. "
    "If it returns not_visible, ask them to hold it in front of you. If "
    "it returns unsupported_object, say you can only track everyday objects like a phone, "
    "cup, bottle, book or remote. To stop, call track_face with enabled false."
)


class TrackTools:
    def __init__(
        self,
        track: Follow,
        speaker: Callable[[], Speaker] = lambda: OWNER,
        objects: tuple[Follow, ObjectSource] | None = None,
    ) -> None:
        """`speaker`: who said the current turn; only the owner may start tracking (owner mode).
        `objects`: the object follower and its source; None (no camera): no track_object."""
        self._track = track
        self._speaker = speaker
        self._objects = objects

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": NAME,
                    "description": "Start or stop following the user: turn head and body to "
                    "keep looking at their face, and toward their voice when they speak from "
                    "elsewhere. Runs until stopped. false also stops following an object.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {
                                "type": "boolean",
                                "description": "true to start following, false to stop",
                            }
                        },
                        "required": ["enabled"],
                    },
                },
            }
        ] + ([] if self._objects is None else [self._object_spec()])

    @staticmethod
    def _object_spec() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": OBJECT,
                "description": "Follow an object in view with head and body until stopped, "
                "such as the user's phone or a cup.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "the object in the user's words, in English, e.g. "
                            "'my iPhone', 'the red cup'",
                        },
                        "coco_class": {
                            "type": "string",
                            "enum": [*COCO, NONE],
                            "description": "the detector class of the same kind as the "
                            "target; none if no class is that kind of object (keys, pen, "
                            "earbuds), never a look-alike",
                        },
                    },
                    "required": ["target", "coco_class"],
                },
            },
        }

    def needs_response(self, name: str, arguments: str) -> bool:
        return True  # confirm, and say how to stop

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        args = args if isinstance(args, dict) else {}
        if name == OBJECT and self._objects is not None:
            return await self._follow_object(args.get("target"), args.get("coco_class"))
        if name != NAME:
            return {"error": f"unknown_tool: {name}"}
        enabled = args.get("enabled")
        if not isinstance(enabled, bool):
            return {"error": "bad_arguments: enabled must be true or false"}
        if not enabled:
            self._track.stop()
            if self._objects is not None:
                self._objects[0].stop()
            return {"status": "stopped"}
        if self._speaker() != OWNER:
            return NOT_OWNER
        error, faces = await self._track.start()
        if error:
            return {"error": error}
        return {"status": "following" if faces else "voice_only"}

    async def _follow_object(self, target: object, label: object) -> dict[str, Any]:
        assert self._objects is not None
        follow, source = self._objects
        if not isinstance(target, str) or not target.strip() or not isinstance(label, str):
            return {"error": "bad_arguments: target and coco_class are required"}
        if label == NONE:
            return {"error": f"unsupported_object: {target.strip()} is not one I can recognise"}
        if self._speaker() != OWNER:
            return NOT_OWNER
        if follow.blocked:  # refuse before loading the detector (about 1 s)
            return {"error": BUSY_LONG}
        if error := await source.acquire(target.strip(), label):
            return {"error": error}
        follow.name = target.strip()
        error, _ = await follow.start()
        if error:
            source.release()
            return {"error": error}
        return {"status": "tracking", "object": label}
