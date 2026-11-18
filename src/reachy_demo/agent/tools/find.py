"""find_object: look around the room for a named object, then look at it and say where (F5)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from reachy_demo.agent.tools.track import NONE
from reachy_demo.attention.search import Search
from reachy_demo.memory.store import OWNER, Speaker
from reachy_demo.perception.objects import COCO

log = logging.getLogger(__name__)

NAME = "find_object"
NOT_OWNER = {"error": "not_owner: only the owner can ask for that right now"}
PROMPT = (
    "You can look around the room for an object. When the user asks you to find or look for "
    "something ('find my phone', 'where is my cup?', 'tìm điện thoại của tôi'), first say one "
    "short sentence that you'll look around, then call find_object with the user's words as "
    "target and the coco_class of the same kind (none if no class fits, never a look-alike). "
    "It takes up to 40 seconds. If found, say what you see from 'description' (if given) and "
    "where it is from your point of view ('where', 'height'), then ask if that is the one. "
    "You keep looking at it until they answer. If the user then says yes, call find_object "
    "with confirmed true (you then look back) and thank them briefly. If they say no, "
    "call find_object again with the same target and coco_class and resume true. If "
    "not_found, say you searched all around and couldn't find it. If unsupported_object, "
    "say you can only find everyday objects like a phone, cup, bottle, book or remote."
)


class FindTools:
    def __init__(self, search: Search, speaker: Callable[[], Speaker] = lambda: OWNER) -> None:
        """`speaker`: who said the current turn; only the owner may start a search."""
        self._search = search
        self._speaker = speaker

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": NAME,
                    "description": "Turn head and body to look around the room for an "
                    "object, then look at it. Returns where it is.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "the object in the user's words, in English, "
                                "e.g. 'my iPhone', 'the red cup'",
                            },
                            "resume": {
                                "type": "boolean",
                                "description": "true only when the user said the object you "
                                "just found is not the right one: keep searching from there",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "true only when the user said the object you "
                                "just found is the right one: look back to where you were",
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
        ]

    def needs_response(self, name: str, arguments: str) -> bool:
        return True  # say where it is

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        if name != NAME:
            return {"error": f"unknown_tool: {name}"}
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        args = args if isinstance(args, dict) else {}
        if args.get("confirmed") is True:
            return await self._search.confirm()
        target, label = args.get("target"), args.get("coco_class")
        if not isinstance(target, str) or not target.strip() or not isinstance(label, str):
            return {"error": "bad_arguments: target and coco_class are required"}
        if label == NONE:
            return {"error": f"unsupported_object: {target.strip()} is not one I can recognise"}
        if self._speaker() != OWNER:
            return NOT_OWNER
        resume = args.get("resume") is True
        try:
            return await self._search.find(target.strip(), label, resume=resume)
        except RuntimeError as e:  # a move refused mid-scan
            return {"error": str(e)}
