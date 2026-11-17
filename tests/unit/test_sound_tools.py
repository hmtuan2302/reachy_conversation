import asyncio
import json

import pytest

from reachy_demo.agent.tools.sound import NAME, SoundTools
from reachy_demo.audio.sound_events import SoundEvent, SoundEvents
from reachy_demo.robot.motion import Motion, Pose


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


class FixedSound(SoundEvents):
    def __init__(self, event: SoundEvent | None) -> None:
        super().__init__()
        self.event = event

    def latest(self, kind: str, now: float) -> SoundEvent | None:
        return self.event


def _tools(event: SoundEvent | None) -> tuple[SoundTools, Motion, list[Pose]]:
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    return SoundTools(FixedSound(event), motion, clock=lambda: 105.0), motion, poses


async def _settle(motion: Motion) -> None:
    async with asyncio.timeout(5):
        while motion.busy:  # noqa: ASYNC110 - test polling, no event to await
            await asyncio.sleep(0.001)


async def _call(tools: SoundTools, source: str) -> dict[str, object]:
    return await tools.call(NAME, json.dumps({"source": source}))


async def test_turns_body_and_head_toward_a_wide_sound() -> None:
    tools, motion, poses = _tools(SoundEvent("noise", 120.0, 100.0))
    result = await _call(tools, "noise")
    assert result["status"] == "turned"
    assert result["where"] == "about 120 degrees to my left, behind me"
    assert result["seconds_ago"] == 5
    await _settle(motion)
    assert poses[-1]["yaw"] == pytest.approx(120)
    assert poses[-1]["body_yaw"] == pytest.approx(75)
    # every pose sent kept the head within 45° of the body
    assert all(abs(p["yaw"] - p["body_yaw"]) <= 45 + 1e-6 for p in poses)


async def test_out_of_reach_reports_without_moving() -> None:
    tools, motion, poses = _tools(SoundEvent("noise", 179.0, 100.0))
    result = await _call(tools, "noise")
    assert result["status"] == "out_of_reach"
    assert result["where"] == "behind me"
    assert not motion.busy and not poses


async def test_no_recent_sound() -> None:
    tools, motion, _ = _tools(None)
    result = await _call(tools, "speaker")
    assert result["status"] == "no_recent_sound"
    assert not motion.busy


async def test_bad_source() -> None:
    tools, _, _ = _tools(None)
    assert "error" in await _call(tools, "cat")


async def test_speaks_the_result() -> None:
    tools, _, _ = _tools(None)
    assert tools.needs_response(NAME, "{}")
