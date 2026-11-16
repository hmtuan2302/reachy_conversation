"""The camera tool: take a picture and ask the vision model about it.

Description from reachy_mini_conversation_app `tools/camera.py` (Apache-2.0, Pollen Robotics).
There the realtime model reads the image itself; here the tool asks the vision model
(perception/vision.py) and returns its text answer, which the agent then says.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from reachy_demo.perception.vision import VisionError

log = logging.getLogger(__name__)

NAME = "camera"
GrabJpeg = Callable[[], bytes | None]  # current camera frame as JPEG, None if unavailable
Ask = Callable[[bytes, str], Awaitable[str]]  # (jpeg, question) -> answer

PROMPT = (
    "You have a camera in your head. When the user asks what you see, what something is, "
    "what they are holding, or to look at something, call camera with their question, then "
    "answer from what it returns. Never guess what is in front of you without calling it."
)


class CameraTools:
    def __init__(self, grab: GrabJpeg, ask: Ask) -> None:
        self._grab = grab
        self._ask = ask

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": NAME,
                    "description": (
                        "Take a picture with the camera to see what is in front of the robot. "
                        "Use this when the user asks you to look at something, see what they "
                        "are holding, describe the scene, or asks what you can see. "
                        "The camera is live, each call captures the current moment. "
                        "If the user asks you to look without saying at what, do not ask for "
                        "clarification, call this tool and describe what you see."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "What to observe or ask about in the picture, in English. "
                                    "Examples: what is the user holding, what object is in "
                                    "front of you, describe the scene."
                                ),
                            }
                        },
                        "required": ["question"],
                    },
                },
            }
        ]

    def needs_response(self, name: str, arguments: str) -> bool:
        return True  # the answer is what the user asked for

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        if name != NAME:
            return {"error": f"unknown_tool: {name}"}
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        question = args.get("question") if isinstance(args, dict) else None
        if not isinstance(question, str) or not question.strip():
            return {"error": "bad_arguments: question must be a non-empty string"}
        jpeg = await asyncio.to_thread(self._grab)
        if not jpeg:
            return {"error": "no camera frame (camera off or not connected)"}
        try:
            answer = await self._ask(jpeg, question.strip())
        except VisionError as e:
            log.warning("camera: %s", e)
            return {"error": str(e)}
        return {"saw": answer}
