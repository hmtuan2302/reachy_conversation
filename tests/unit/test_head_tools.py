import asyncio
import json

import pytest

from reachy_demo.agent.tools.head import DANCE, POSES, STOP, HeadTools
from reachy_demo.robot.dances import Dance
from reachy_demo.robot.motion import BUSY_LONG, Motion, Pose

DANCES = {
    "bob": Dance("nod to the beat", 4.0, lambda t: {"pitch": 5.0}),
    "sway": Dance("side to side", 2.0, lambda t: {"y": 10.0}),
}


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


def _tools(dances: dict[str, Dance] | None = None) -> tuple[HeadTools, list[tuple[float, ...]]]:
    poses: list[tuple[float, ...]] = []

    def send(p: Pose) -> None:
        poses.append((p["yaw"], p["pitch"], p["roll"]))

    motion = Motion(send, hz=1000, clock=FakeClock())
    return HeadTools(motion, DANCES if dances is None else dances, move_s=0.2), poses


async def _settle(tools: HeadTools) -> None:
    async with asyncio.timeout(2):
        while tools.busy:  # noqa: ASYNC110 - test polling, no event to await
            await asyncio.sleep(0.001)


def _args(**kw: object) -> str:
    return json.dumps(kw)


@pytest.mark.parametrize(
    ("tool", "direction", "pose"),
    [
        ("turn_head", "left", (20, 0, 0)),
        ("turn_head", "right", (-20, 0, 0)),
        ("tilt_head", "up", (0, -15, 0)),
        ("tilt_head", "down", (0, 15, 0)),
        ("roll_head", "left", (0, 0, 15)),
        ("roll_head", "right", (0, 0, -15)),
    ],
)
async def test_each_direction_goes_to_a_fixed_angle(
    tool: str, direction: str, pose: tuple[float, float, float]
) -> None:
    tools, poses = _tools()
    assert await tools.call(tool, _args(direction=direction)) == {
        "status": f"started {tool} {direction}"
    }
    await tools.call(tool, _args(direction=direction))  # repeating never goes further
    await _settle(tools)
    assert poses[-1] == pose
    assert all(abs(p[i]) <= abs(pose[i]) for p in poses for i in range(3))
    await tools.close()


async def test_axes_combine_from_the_planned_pose() -> None:
    tools, poses = _tools()
    await tools.call("turn_head", _args(direction="right"))
    await tools.call("tilt_head", _args(direction="down"))  # before the turn ended
    await _settle(tools)
    assert poses[-1] == (-20, 15, 0)
    await tools.close()


async def test_moves_refused_while_dancing_then_accepted_after_stop() -> None:
    tools, _ = _tools()
    assert await tools.call(DANCE, _args(move="bob", repeat=2)) == {
        "status": "started dance bob for 8.0 s"
    }
    assert await tools.call("turn_head", _args(direction="left")) == {"error": BUSY_LONG}
    assert await tools.call(DANCE, _args(move="sway")) == {"error": BUSY_LONG}
    assert await tools.call(STOP, "{}") == {"status": "stopped"}  # stop is never refused
    await _settle(tools)
    assert await tools.call("turn_head", _args(direction="left")) == {
        "status": "started turn_head left"
    }
    await tools.close()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("turn_head", '{"direction": "behind"}'),
        ("turn_head", "{}"),
        ("turn_head", "not json"),
        ("turn_head", "[1]"),
        (DANCE, '{"move": "moonwalk"}'),
        (DANCE, '{"move": "bob", "repeat": "lots"}'),
        (DANCE, '{"move": "bob", "repeat": true}'),
    ],
)
async def test_bad_arguments_do_not_move(name: str, arguments: str) -> None:
    tools, poses = _tools()
    result = await tools.call(name, arguments)
    assert result["error"].startswith("bad_arguments")
    assert not tools.busy and poses == []


async def test_unknown_tool() -> None:
    tools, _ = _tools()
    assert await tools.call("spin", "{}") == {"error": "unknown_tool: spin"}


def test_specs_offer_fixed_directions_dance_and_stop() -> None:
    specs = _tools()[0].specs()
    assert [s["function"]["name"] for s in specs] == [*POSES, DANCE, STOP]
    params = {s["function"]["name"]: s["function"]["parameters"] for s in specs}
    assert params["turn_head"]["properties"]["direction"]["enum"] == ["left", "right", "center"]
    move = params[DANCE]["properties"]["move"]
    assert move["enum"] == ["bob", "sway"]
    assert "sway: side to side" in move["description"]
    assert params[DANCE]["properties"]["repeat"]["maximum"] == 7  # 15 s // 2 s pass
    assert params[DANCE]["required"] == []


def test_no_dance_tool_without_dances() -> None:
    names = [s["function"]["name"] for s in _tools({})[0].specs()]
    assert names == [*POSES, STOP]


async def test_dance_without_a_move_picks_one() -> None:
    tools, _ = _tools({"bob": DANCES["bob"]})
    assert await tools.call(DANCE, "{}") == {"status": "started dance bob for 4.0 s"}
    await tools.close()


@pytest.mark.parametrize(
    ("arguments", "needed"),
    [
        ('{"direction": "left"}', False),
        ('{"direction": "left", "reply": false}', False),
        ('{"direction": "left", "reply": true}', True),
        ('{"move": "bob", "reply": true}', True),
        ('{"direction": "left", "reply": "yes"}', False),
        ("not json", False),
    ],
)
def test_moves_are_silent_unless_reply_flagged(arguments: str, needed: bool) -> None:
    assert _tools()[0].needs_response("turn_head", arguments) is needed


async def test_dance_refused_for_guest() -> None:
    from reachy_demo.agent.tools.head import NOT_OWNER
    from reachy_demo.agent.tools.owner import GUEST

    motion = Motion(lambda p: None, hz=1000, clock=FakeClock())
    tools = HeadTools(motion, DANCES, move_s=0.2, speaker=lambda: GUEST)
    assert await tools.call(DANCE, json.dumps({"move": "bob"})) == NOT_OWNER
    assert not tools.busy


async def test_stop_and_turn_allowed_for_guest() -> None:
    from reachy_demo.agent.tools.owner import GUEST

    motion = Motion(lambda p: None, hz=1000, clock=FakeClock())
    tools = HeadTools(motion, DANCES, move_s=0.2, speaker=lambda: GUEST)
    assert await tools.call(STOP, "{}") == {"status": "stopped"}
    assert "started" in (await tools.call("turn_head", json.dumps({"direction": "left"})))["status"]
    await tools.close()
