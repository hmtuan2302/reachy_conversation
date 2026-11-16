import json
from typing import Any

import pytest

from reachy_demo.agent.tools import Toolbox
from reachy_demo.agent.tools.camera import NAME, CameraTools
from reachy_demo.perception.vision import VisionError

JPEG = b"\xff\xd8jpeg"


def _tools(
    frame: bytes | None = JPEG, answer: str | Exception = "a red cup"
) -> tuple[CameraTools, list[tuple[bytes, str]]]:
    asked: list[tuple[bytes, str]] = []

    async def ask(jpeg: bytes, question: str) -> str:
        asked.append((jpeg, question))
        if isinstance(answer, Exception):
            raise answer
        return answer

    return CameraTools(lambda: frame, ask), asked


async def test_grabs_a_frame_and_returns_what_the_model_saw() -> None:
    tools, asked = _tools()
    result = await tools.call(NAME, json.dumps({"question": " what is the user holding? "}))
    assert result == {"saw": "a red cup"}
    assert asked == [(JPEG, "what is the user holding?")]


async def test_no_frame_is_an_error_and_asks_nothing() -> None:
    tools, asked = _tools(frame=None)
    result = await tools.call(NAME, json.dumps({"question": "what is this?"}))
    assert "no camera frame" in result["error"]
    assert asked == []


async def test_vision_failure_is_an_error() -> None:
    tools, _ = _tools(answer=VisionError("vision answer took over 15 s"))
    assert await tools.call(NAME, json.dumps({"question": "q"})) == {
        "error": "vision answer took over 15 s"
    }


@pytest.mark.parametrize("arguments", ["{}", '{"question": "  "}', "not json", "[1]"])
async def test_bad_arguments(arguments: str) -> None:
    tools, asked = _tools()
    assert (await tools.call(NAME, arguments))["error"].startswith("bad_arguments")
    assert asked == []


def test_camera_answers_are_always_spoken() -> None:
    assert _tools()[0].needs_response(NAME, "{}")


class FakeSet:
    def __init__(self, name: str, silent: bool = False) -> None:
        self.name = name
        self.silent = silent
        self.calls: list[str] = []

    def specs(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": self.name}}]

    def needs_response(self, name: str, arguments: str) -> bool:
        return not self.silent

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        self.calls.append(arguments)
        return {"status": name}


async def test_toolbox_routes_each_call_to_its_set() -> None:
    head, cam = FakeSet("turn_head", silent=True), FakeSet("camera")
    box = Toolbox(head, cam)
    assert [s["function"]["name"] for s in box.specs()] == ["turn_head", "camera"]
    assert await box.call("camera", "{}") == {"status": "camera"}
    assert cam.calls == ["{}"] and head.calls == []
    assert not box.needs_response("turn_head", "{}")
    assert box.needs_response("camera", "{}")
    assert await box.call("fly", "{}") == {"error": "unknown_tool: fly"}
    assert box.needs_response("fly", "{}")  # the error gets spoken


def test_toolbox_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Toolbox(FakeSet("camera"), FakeSet("camera"))
