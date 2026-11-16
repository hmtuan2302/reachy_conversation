import asyncio
import math
import threading
from collections.abc import Callable

from reachy_demo.robot.motion import (
    AXES,
    BUSY_FULL,
    BUSY_LONG,
    LIMITS,
    NEUTRAL,
    Motion,
    Move,
    Pose,
    dance,
    goto,
)

LEFT: Pose = {**NEUTRAL, "yaw": 20.0}


def _bob(repeat: int = 1) -> Move:
    """A 4 s test dance: nod (pitch) and wiggle the antennas."""
    return dance(
        "bob",
        4.0,
        lambda t: {"pitch": 10 * math.sin(t), "antenna_0": 20 * math.sin(t)},
        repeat,
    )


def _sway(repeat: int = 1) -> Move:
    return dance("sway", 4.0, lambda t: {"yaw": 12 * math.sin(t), "y": 30 * math.sin(t)}, repeat)


class FakeClock:
    """Advances `step` seconds per read, so a 4 s dance takes ~80 ticks, not 4 s."""

    def __init__(self, step: float = 0.05) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class FakeRobot:
    def __init__(self, fail_after: int | None = None) -> None:
        self.poses: list[tuple[float, float, float]] = []  # yaw, pitch, roll
        self.full: list[Pose] = []
        self.fail_after = fail_after
        self.busy = threading.Lock()

    def __call__(self, pose: Pose) -> None:
        assert self.busy.acquire(blocking=False), "two senders at once"
        try:
            if self.fail_after is not None and len(self.poses) >= self.fail_after:
                raise ConnectionError("robot unreachable")
            assert set(pose) == set(AXES)  # always a full pose
            self.full.append(pose)
            self.poses.append((pose["yaw"], pose["pitch"], pose["roll"]))
        finally:
            self.busy.release()


def _motion(robot: FakeRobot, **kw: object) -> Motion:
    return Motion(robot, hz=1000, clock=FakeClock(), **kw)  # type: ignore[arg-type]


async def _until(cond: Callable[[], bool], timeout_s: float = 2.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not cond():  # noqa: ASYNC110 - test polling, no event to await
            await asyncio.sleep(0.001)


async def test_short_moves_run_in_order_and_end_on_target() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    assert m.submit(goto("left", LEFT)) is None
    assert m.submit(goto("down", {**LEFT, "pitch": 15.0})) is None
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (20.0, 15.0, 0.0)
    assert (20.0, 0.0, 0.0) in robot.poses  # the first move ended on its target
    await m.close()


async def test_dance_blocks_other_moves_until_stopped() -> None:
    m = _motion(FakeRobot())
    assert m.submit(goto("left", LEFT)) is None
    assert m.submit(_bob()) is None  # a dance may follow short moves
    assert m.submit(goto("right", NEUTRAL)) == BUSY_LONG
    assert m.submit(_sway()) == BUSY_LONG
    assert m.dancing
    m.stop()
    assert not m.dancing
    assert m.submit(goto("left", LEFT)) is None
    await m.close()


async def test_queue_limit() -> None:
    m = _motion(FakeRobot(), max_pending=1)
    m.submit(goto("a", LEFT))
    await asyncio.sleep(0)  # the loop took it
    m.submit(goto("b", NEUTRAL))
    assert m.submit(goto("c", LEFT)) == BUSY_FULL
    await m.close()


# 1. a crash never leaves the loop busy
async def test_failure_reports_once_clears_busy_and_loop_restarts() -> None:
    errors: list[str] = []
    robot = FakeRobot(fail_after=3)
    m = _motion(robot, on_error=errors.append)
    m.submit(_bob())
    await _until(lambda: not m.busy)
    assert errors == ["robot unreachable"]
    assert not m.dancing  # not stuck: new moves are accepted
    robot.fail_after = None
    assert m.submit(goto("left", LEFT)) is None
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (20.0, 0.0, 0.0)
    await m.close()


# 3. after a stop mid-dance the head eases to neutral; the next move starts from the real pose
async def test_stop_mid_dance_eases_to_neutral() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    m.submit(_sway(3))
    await _until(lambda: len(robot.poses) > 30)  # mid-dance, head off-center
    assert m.pose != NEUTRAL
    m.stop()
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (0.0, 0.0, 0.0)
    assert len(robot.poses) < 3 * 80  # it did not play the whole 12 s dance
    m.submit(goto("left", LEFT))
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (20.0, 0.0, 0.0)
    await m.close()


async def test_stop_lets_a_short_move_finish_on_its_fixed_angle() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    m.submit(goto("left", LEFT))
    m.submit(goto("down", {**LEFT, "pitch": 15.0}))
    await _until(lambda: len(robot.poses) > 2)
    m.stop()
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (20.0, 0.0, 0.0)  # left finished, down dropped
    await m.close()


async def test_dance_eases_in_from_start_and_ends_at_neutral() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    m.submit(goto("left", LEFT))
    m.submit(_bob())
    await _until(lambda: not m.busy)
    i = robot.poses.index((20.0, 0.0, 0.0))
    assert abs(robot.poses[i + 1][0] - 20.0) < 2  # no jump when the dance starts
    assert robot.poses[-1] == (0.0, 0.0, 0.0)
    await m.close()


# 4. dances are capped; poses are clamped
def test_dance_length_capped_at_15_s() -> None:
    assert _bob(99).duration_s == 12.0
    assert _bob(0).duration_s == 4.0


async def test_every_pose_sent_is_clamped() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    wild = Move(
        "wild", False, 0.1, lambda t, s: {"yaw": 90.0, "pitch": -90.0, "roll": 90.0}, NEUTRAL
    )
    m.submit(wild)
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (LIMITS["yaw"], -LIMITS["pitch"], LIMITS["roll"])
    await m.close()


# 5. exit mid-dance does not wait for the dance: it ends it and goes to neutral
async def test_close_mid_dance_ends_at_neutral_quickly() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    m.submit(_sway(3))
    await _until(lambda: len(robot.poses) > 30)
    await m.close()
    assert robot.poses[-1] == (0.0, 0.0, 0.0)
    assert len(robot.poses) < 3 * 80
    assert not m.busy


async def test_close_when_idle_returns() -> None:
    await _motion(FakeRobot()).close()


async def test_dance_moves_antennas_and_position_then_ends_at_neutral() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    m.submit(_bob())
    await _until(lambda: not m.busy)
    assert max(p["antenna_0"] for p in robot.full) > NEUTRAL["antenna_0"] + 5
    assert robot.full[-1] == NEUTRAL
    m.submit(_sway())
    await _until(lambda: not m.busy)
    assert max(p["y"] for p in robot.full) > 5  # head position, not just orientation
    await m.close()


async def test_new_dance_accepted_right_after_stop() -> None:
    robot = FakeRobot()
    m = _motion(robot)
    m.submit(_bob(3))
    await _until(lambda: len(robot.poses) > 10)
    m.stop()
    assert m.submit(_sway()) is None  # same tick as the stop
    await _until(lambda: not m.busy)
    assert robot.poses[-1] == (0.0, 0.0, 0.0)
    assert len(robot.poses) < 3 * 80 + 80
    await m.close()
