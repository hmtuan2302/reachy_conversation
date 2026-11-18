import asyncio
import json
import math
from collections.abc import Callable

import numpy as np
import pytest

from reachy_demo.agent.tools.mimic import NAME, NOT_OWNER, MimicTools
from reachy_demo.attention.mimic import (
    LOST_S,
    MAX_PITCH,
    MAX_YAW,
    SPEED_DPS,
    Mimic,
    OneEuro,
)
from reachy_demo.memory.store import Speaker
from reachy_demo.perception.head_pose import HeadPose, pose_from_matrix
from reachy_demo.robot.motion import BUSY_LONG, LIMITS, NEUTRAL, Motion, Pose, dance

# --- pose_from_matrix (MediaPipe camera space: x right, y up, camera looks down -z) ---------


def _matrix(rot_x: float = 0, rot_y: float = 0, rot_z: float = 0, t=(0.0, 0.0, -60.0)):  # type: ignore[no-untyped-def]
    ax, ay, az = map(math.radians, (rot_x, rot_y, rot_z))
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx @ rz
    m[:3, 3] = t
    return m


def test_face_straight_ahead_is_all_zero() -> None:
    p = pose_from_matrix(_matrix())
    assert (p.yaw, p.pitch, p.roll, p.bearing_yaw, p.bearing_pitch) == pytest.approx((0,) * 5)
    assert p.distance_cm == pytest.approx(60)


def test_turning_toward_own_left_is_positive_yaw() -> None:
    assert pose_from_matrix(_matrix(rot_y=20)).yaw == pytest.approx(20)


def test_looking_down_is_positive_pitch() -> None:
    assert pose_from_matrix(_matrix(rot_x=15)).pitch == pytest.approx(15)


def test_counter_clockwise_tilt_is_positive_roll() -> None:
    assert pose_from_matrix(_matrix(rot_z=10)).roll == pytest.approx(10)


def test_off_centre_face_looking_at_camera_has_no_turn() -> None:
    # face 20 cm to the image right, turned to look straight at the camera
    look = -math.degrees(math.atan2(20, 60))
    p = pose_from_matrix(_matrix(rot_y=look, t=(20.0, 0.0, -60.0)))
    assert p.yaw == pytest.approx(0, abs=1e-6)
    assert p.bearing_yaw == pytest.approx(look)  # image right = robot's right = negative


def test_one_euro_holds_still_input_and_follows_a_step() -> None:
    f = OneEuro()
    assert f(10.0, 0.0) == 10.0
    assert f(10.0, 0.1) == pytest.approx(10.0)
    out = [f(30.0, 0.1 + 0.05 * i) for i in range(1, 40)]
    assert 10 < out[0] < 30 and out[-1] == pytest.approx(30, abs=0.5)


# --- the behaviour --------------------------------------------------------------------------


def _head(yaw: float = 0, pitch: float = 0, roll: float = 0, **kw: float) -> HeadPose:
    return HeadPose(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        bearing_yaw=kw.get("bearing_yaw", 0.0),
        bearing_pitch=kw.get("bearing_pitch", 0.0),
        distance_cm=kw.get("distance_cm", 60.0),
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


class FakeEstimator:
    def __init__(self) -> None:
        self.head: HeadPose | None = _head()
        self.camera: Callable[[], Pose] | None = None  # set: the user stands still at yaw 0
        self.closed = False

    def estimate(self, bgr: object, at_s: float) -> HeadPose | None:
        if self.head is None or self.camera is None:
            return self.head
        cam = self.camera()  # the camera turns with the head, so the user drifts in the image
        return _head(
            self.head.yaw, self.head.pitch, bearing_yaw=-cam["yaw"], bearing_pitch=-cam["pitch"]
        )

    def close(self) -> None:
        self.closed = True


def _mimic(
    make: Callable[[], FakeEstimator] | None = None,
) -> tuple[Mimic, Motion, list[Pose], FakeEstimator]:
    est = FakeEstimator()
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mimic = Mimic(motion, lambda: frame, make or (lambda: est), poll_s=0.001)
    return mimic, motion, poses, est


async def _until(cond: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not cond():  # noqa: ASYNC110 - test polling
            await asyncio.sleep(0.001)


async def _started(mimic: Mimic) -> None:
    assert await mimic.start() is None
    await _until(lambda: mimic._target is not None)


def _feed(mimic: Mimic, head: HeadPose | None, n: int = 40, seen: Pose | None = None) -> Pose:
    """Hold one reading long enough for the filters to settle; returns the target."""
    for i in range(n):
        mimic.step(head, seen or dict(NEUTRAL), 100.0 + 0.05 * i)
    assert mimic._target is not None
    return mimic._target


async def test_mirrors_a_turn_to_the_users_left_toward_the_robots_right() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    target = _feed(mimic, _head(yaw=20))
    assert target["yaw"] == pytest.approx(-20, abs=0.5)
    assert target["body_yaw"] == 0
    mimic.stop()


async def test_copies_nod_and_tilt() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    target = _feed(mimic, _head(pitch=10, roll=-12))
    assert target["pitch"] == pytest.approx(10, abs=0.5)
    assert target["roll"] == pytest.approx(-12, abs=0.5)
    mimic.stop()


async def test_big_moves_are_capped_to_keep_the_user_in_view_and_the_robot_safe() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    target = _feed(mimic, _head(yaw=70, pitch=60, roll=50))
    assert target["yaw"] == pytest.approx(-MAX_YAW, abs=0.5)
    assert target["pitch"] == pytest.approx(MAX_PITCH, abs=0.5)
    assert target["roll"] == pytest.approx(LIMITS["roll"], abs=0.5)
    mimic.stop()


async def test_faces_a_user_standing_to_the_side() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    target = _feed(mimic, _head(bearing_yaw=30), n=400)  # user 30° left; facing drifts slowly
    assert target["yaw"] == pytest.approx(30, abs=0.5)
    mimic.stop()


async def test_robot_roll_is_added_back_to_the_image_roll() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    target = _feed(mimic, _head(roll=0), seen={**NEUTRAL, "roll": -10})
    assert target["roll"] == pytest.approx(-10, abs=0.5)  # the camera rolled, not the user
    mimic.stop()


async def test_leaning_in_moves_the_head_forward() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    _feed(mimic, _head(distance_cm=60), n=5)  # resting distance
    target = _feed(mimic, _head(distance_cm=55), n=1)
    assert target["x"] == pytest.approx(10)
    target = _feed(mimic, _head(distance_cm=30), n=1)
    assert target["x"] == pytest.approx(LIMITS["x"])
    mimic.stop()


async def test_lost_face_looks_back_at_the_user() -> None:
    mimic, *_ = _mimic()
    await _started(mimic)
    _feed(mimic, _head(yaw=20, roll=10, bearing_yaw=10), n=400)  # facing drifts slowly
    mimic.step(None, dict(NEUTRAL), 120.0 + LOST_S + 0.1)
    assert mimic._target is not None
    assert mimic._target["yaw"] == pytest.approx(10, abs=0.5)
    assert mimic._target["roll"] == 0
    mimic.stop()


async def test_trajectory_is_smooth_capped_and_ends_at_neutral() -> None:
    mimic, motion, poses, est = _mimic()
    await _started(mimic)
    est.camera = lambda: motion.pose
    est.head = _head(yaw=30)  # sudden full turn
    await _until(lambda: len(poses) > 120)
    yaws = [p["yaw"] for p in poses]  # the mimic part, before stop eases back to neutral
    mimic.stop()
    await _until(lambda: not mimic.active and not motion.busy)
    assert est.closed
    assert poses[-1] == pytest.approx(NEUTRAL)
    steps = np.diff(yaws)
    assert max(abs(steps)) <= SPEED_DPS * 0.05 + 1e-6  # speed cap per 50 ms tick
    assert min(yaws) == pytest.approx(-30, abs=2)  # reached the mirrored turn


async def test_refused_while_dancing_without_loading_the_model() -> None:
    loads: list[int] = []

    def make() -> FakeEstimator:
        loads.append(1)
        return FakeEstimator()

    mimic, motion, *_ = _mimic(make)
    assert motion.submit(dance("d", 4.0, lambda t: {})) is None
    assert await mimic.start() == BUSY_LONG
    assert not loads
    motion.stop()


async def test_missing_model_is_reported() -> None:
    def make() -> FakeEstimator:
        raise FileNotFoundError("face model missing")

    mimic, *_ = _mimic(make)
    assert (await mimic.start() or "").startswith("mimic_unavailable")
    assert not mimic.active


# --- the tool -------------------------------------------------------------------------------


async def _call(tools: MimicTools, **args: object) -> dict[str, object]:
    return await tools.call(NAME, json.dumps(args))


async def test_tool_starts_and_stops() -> None:
    mimic, *_ = _mimic()
    tools = MimicTools(mimic)
    assert await _call(tools, enabled=True) == {"status": "mimicking"}
    assert mimic.active
    assert await _call(tools, enabled=False) == {"status": "stopped"}
    await _until(lambda: not mimic.active)


async def test_tool_refuses_guests_and_bad_arguments() -> None:
    mimic, *_ = _mimic()
    tools = MimicTools(mimic, speaker=lambda: Speaker(2, "guest"))
    assert await _call(tools, enabled=True) == NOT_OWNER
    assert not mimic.active
    assert str((await _call(tools, enabled=1))["error"]).startswith("bad_arguments")


async def test_stop_tracking_then_mimic_in_one_sentence() -> None:
    """'Stop tracking and mimic me': the stop ends tracking, then the rest starts mimic."""
    from reachy_demo.attention.follow import Follow

    mimic, motion, poses, _ = _mimic()
    follow = Follow(motion, lambda: (0.3, 0.0), lambda on: True, poll_s=0.001)
    assert await follow.start() == (None, True)
    await _until(lambda: len(poses) > 5)  # tracking
    assert await mimic.start() == BUSY_LONG  # "mimic me" alone is refused
    motion.stop()  # what barge-in's on_stop does
    assert await mimic.start() is None  # the rest of the sentence
    await _until(lambda: mimic._target is not None and not follow.active)
    assert mimic.active
    mimic.stop()


def _late_camera(lag: float) -> tuple[list[float], list[float]]:
    """Head yaw with frames `lag` s old: user still at 10° (2-4 s), then turns their head 20°."""
    m = Mimic(Motion(lambda p: None), lambda: None, lambda: None)  # type: ignore[arg-type,return-value]
    m._pose_at(0.0, dict(NEUTRAL))
    seen: list[tuple[float, float]] = []
    yaws: list[float] = []
    for i in range(500):  # 50 Hz ticks, ~16 Hz readings
        t = i * 0.02
        pose = m._pose_at(t, dict(NEUTRAL))
        seen.append((t, pose["yaw"]))
        if i % 3 == 0:
            cam = [y for s, y in seen if s <= t - lag][-1:] or [0.0]
            turn = 20.0 if t >= 5 else 0.0
            m.step(HeadPose(turn, 0.0, 0.0, 10.0 - cam[0], 0.0, 80.0), dict(pose), t)
        yaws.append(pose["yaw"])
    return yaws[100:250], yaws[400:]


def test_late_frames_do_not_swing_the_head() -> None:
    still, turned = _late_camera(0.4)
    assert all(abs(y - 10) < 3 for y in still)  # facing the user, steady
    assert all(abs(y + 10) < 3 for y in turned)  # mirrored 20° turn, steady
