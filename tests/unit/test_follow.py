import asyncio
import json
from collections.abc import Callable

import pytest

from reachy_demo.agent.tools.track import NAME, NOT_OWNER, TrackTools
from reachy_demo.attention.follow import HEAD_REACH, Follow
from reachy_demo.audio.sound_events import SoundEvent, SoundEvents
from reachy_demo.memory.store import Speaker
from reachy_demo.robot.motion import BUSY_LONG, NEUTRAL, Motion, Pose, dance


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


class Voice(SoundEvents):
    def __init__(self) -> None:
        super().__init__()
        self.event: SoundEvent | None = None

    def latest(self, kind: str, now: float) -> SoundEvent | None:
        return self.event


class Detector:
    def __init__(self, ok: bool = True, fail: bool = False) -> None:
        self.calls: list[bool] = []
        self.ok, self.fail = ok, fail
        self.face: tuple[float, float] | None = None

    def switch(self, on: bool) -> bool:
        self.calls.append(on)
        if self.fail:
            raise OSError("no daemon")
        return self.ok

    def read(self) -> tuple[float, float] | None:
        return self.face


def _track(det: Detector | None = None) -> tuple[Follow, Motion, list[Pose], Voice, Detector]:
    det = det or Detector()
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    voice = Voice()
    track = Follow(motion, det.read, det.switch, voice, poll_s=0.001)
    return track, motion, poses, voice, det


async def _until(cond: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not cond():  # noqa: ASYNC110 - test polling
            await asyncio.sleep(0.001)


async def _started(track: Follow, motion: Motion) -> None:
    assert await track.start() == (None, True)
    await _until(lambda: track._target is not None)


def _settle_target(track: Follow) -> None:
    """Jump the chased pose onto the target, as if the head had arrived."""
    assert track._target is not None
    track._pose = dict(track._target)


async def test_face_right_of_centre_turns_head_right_and_body_stays() -> None:
    track, motion, *_ = _track()
    await _started(track, motion)
    track.step((0.5, 0.0), now=10.0)
    assert track._target is not None
    assert -HEAD_REACH < track._target["yaw"] < 0
    assert track._target["body_yaw"] == 0
    track.stop()


async def test_face_low_in_frame_tilts_down() -> None:
    track, motion, *_ = _track()
    await _started(track, motion)
    track.step((0.0, 0.5), now=10.0)
    assert track._target is not None and track._target["pitch"] > 0  # + = down
    track.stop()


async def test_no_face_turns_head_and_body_toward_the_voice() -> None:
    track, motion, _, voice, _ = _track()
    await _started(track, motion)
    voice.event = SoundEvent("speaker", 90.0, 5.0)
    track.step(None, now=10.0)
    assert track._target is not None
    assert track._target["yaw"] == pytest.approx(90)
    assert track._target["body_yaw"] == pytest.approx(90 - HEAD_REACH)
    track.stop()


async def test_same_sentence_is_acted_on_once() -> None:
    track, motion, _, voice, _ = _track()
    await _started(track, motion)
    voice.event = SoundEvent("speaker", 90.0, 5.0)
    track.step(None, now=10.0)
    assert track._target is not None
    track._target["yaw"] = 0.0
    track.step(None, now=11.0)
    assert track._target["yaw"] == 0.0
    track.stop()


async def test_voice_near_a_seen_face_keeps_the_face() -> None:
    track, motion, _, voice, _ = _track()
    await _started(track, motion)
    track.step((0.0, 0.0), now=10.0)  # face straight ahead
    voice.event = SoundEvent("speaker", 20.0, 10.1)  # DoA noise, same person
    track.step((0.0, 0.0), now=10.2)
    assert track._target is not None and track._target["yaw"] == pytest.approx(0)
    track.stop()


async def test_new_speaker_far_off_wins_and_old_face_is_ignored_for_a_while() -> None:
    track, motion, _, voice, _ = _track()
    await _started(track, motion)
    track.step((0.0, 0.0), now=10.0)
    voice.event = SoundEvent("speaker", 60.0, 10.1)
    track.step((0.0, 0.0), now=10.2)
    assert track._target is not None and track._target["yaw"] == pytest.approx(60)
    _settle_target(track)
    track.step((0.9, 0.0), now=10.5)  # the old face, far right in frame: ignored
    assert track._target["yaw"] == pytest.approx(60)
    track.step((0.1, 0.0), now=10.6)  # a face near the voice: followed
    assert track._target["yaw"] < 60
    track.stop()


async def test_out_of_reach_voice_turns_as_far_as_it_can() -> None:
    track, motion, _, voice, _ = _track()
    await _started(track, motion)
    voice.event = SoundEvent("speaker", 179.0, 5.0)
    track.step(None, now=10.0)
    assert track._target is not None
    assert track._target["body_yaw"] == pytest.approx(120)
    assert track._target["yaw"] == pytest.approx(120 + HEAD_REACH)
    track.stop()


async def test_stop_ends_the_loop_turns_detector_off_and_goes_neutral() -> None:
    track, motion, poses, voice, det = _track()
    det.face = (0.3, 0.0)
    await _started(track, motion)
    voice.event = SoundEvent("speaker", 50.0, 0.0)
    await _until(lambda: abs(poses[-1]["yaw"]) > 5)  # moving under tracking
    track.stop()
    await _until(lambda: not track.active and not motion.busy)
    assert det.calls == [True, False]
    assert poses[-1] == pytest.approx(NEUTRAL)
    assert all(abs(p["yaw"] - p["body_yaw"]) <= 45 + 1e-6 for p in poses)


async def test_refused_while_dancing_and_detector_left_off() -> None:
    track, motion, _, _, det = _track()
    assert motion.submit(dance("d", 4.0, lambda t: {})) is None
    error, _ = await track.start()
    assert error == BUSY_LONG
    assert det.calls == [True, False]
    assert not track.active
    motion.stop()


async def test_no_detector_follows_voice_only() -> None:
    track, motion, *_ = _track(Detector(fail=True))
    assert await track.start() == (None, False)
    assert track.active
    track.stop()
    await _until(lambda: not track.active)


async def test_other_moves_are_refused_while_tracking() -> None:
    track, motion, *_ = _track()
    await _started(track, motion)
    assert motion.submit(dance("d", 4.0, lambda t: {})) == BUSY_LONG
    track.stop()


# --- the tool -----------------------------------------------------------------------------


async def _call(tools: TrackTools, **args: object) -> dict[str, object]:
    return await tools.call(NAME, json.dumps(args))


async def test_tool_starts_and_stops() -> None:
    track, motion, *_ = _track()
    tools = TrackTools(track)
    assert await _call(tools, enabled=True) == {"status": "following"}
    assert track.active
    assert await _call(tools, enabled=False) == {"status": "stopped"}
    await _until(lambda: not track.active)


async def test_tool_reports_voice_only_without_a_camera() -> None:
    track, *_ = _track(Detector(ok=False))
    tools = TrackTools(track)
    assert await _call(tools, enabled=True) == {"status": "voice_only"}
    track.stop()


async def test_tool_refuses_guests_and_bad_arguments() -> None:
    track, *_ = _track()
    guest = Speaker(2, "guest")
    tools = TrackTools(track, speaker=lambda: guest)
    assert await _call(tools, enabled=True) == NOT_OWNER
    assert not track.active
    assert str((await _call(tools, enabled="yes"))["error"]).startswith("bad_arguments")


async def test_stop_dance_then_track_in_one_sentence() -> None:
    """'Stop dancing and track my face': the stop ends the dance, then the rest starts tracking."""
    track, motion, poses, _, _ = _track()
    assert motion.submit(dance("d", 4.0, lambda t: {"yaw": 10.0})) is None
    await _until(lambda: len(poses) > 5)  # dancing
    motion.stop()  # what barge-in's on_stop does
    assert await track.start() == (None, True)  # the rest of the sentence, same reply
    await _until(lambda: track._target is not None)  # the neutral ease ran, tracking runs
    assert track.active
    track.stop()


def _still_face_with_lag(lag: float, stamped: bool) -> list[float]:
    """Head yaw over 8 s chasing a still face at 20° left, readings `lag` s old (10 Hz)."""
    import math

    from reachy_demo.perception.head_pose import TAN_HALF_H

    t = [0.0]
    track = Follow(Motion(lambda p: None), lambda: None, lambda on: True, clock=lambda: t[0])
    seen: list[tuple[float, float]] = []
    yaws: list[float] = []
    for i in range(400):  # 50 Hz motion ticks
        t[0] = i * 0.02
        yaw = track._pose_at(t[0], dict(NEUTRAL))["yaw"]
        seen.append((t[0], yaw))
        if i % 5 == 0:
            at = t[0] - lag
            cam = [y for s, y in seen if s <= at][-1:] or [0.0]
            x = math.tan(math.radians(cam[0] - 20.0)) / TAN_HALF_H
            track.step((x, 0.0, at) if stamped else (x, 0.0), t[0])
        yaws.append(yaw)
    return yaws


def test_lagged_readings_without_a_time_wander_around_a_still_face() -> None:
    late = _still_face_with_lag(0.4, stamped=False)[-150:]
    assert max(late) - min(late) > 20  # the bug this guards: old offsets applied to a new pose


def test_lagged_readings_with_their_time_settle_on_a_still_face() -> None:
    yaws = _still_face_with_lag(0.4, stamped=True)
    assert max(yaws) < 22
    assert all(abs(y - 20) < 1 for y in yaws[-150:])


def test_a_repeated_reading_is_applied_once() -> None:
    t = [10.0]
    track = Follow(Motion(lambda p: None), lambda: None, lambda on: True, clock=lambda: t[0])
    track._pose_at(0.0, dict(NEUTRAL))
    track.step((0.5, 0.0, 9.9), now=10.0)
    assert track._target is not None
    first = track._target["yaw"]
    t[0] = 10.5
    track._pose_at(0.5, dict(NEUTRAL))  # the head moved toward it
    track.step((0.5, 0.0, 9.9), now=10.5)  # same frame again: no new information
    assert track._target["yaw"] == first


async def test_robot_steered_tracking_ignores_readings_and_turns_detector_off() -> None:
    det = Detector()
    det.face = (0.9, 0.0)  # would turn the head far right if steered from here
    poses: list[Pose] = []
    motion = Motion(poses.append, hz=1000, clock=FakeClock())
    track = Follow(motion, det.read, det.switch, Voice(), poll_s=0.001, steer_on_robot=True)
    await _started(track, motion)
    await asyncio.sleep(0.05)
    assert track._target is not None and track._target["yaw"] == pytest.approx(0)
    assert det.calls == [True]
    track.stop()
    await _until(lambda: not track.active)
    assert det.calls == [True, False]


async def test_robot_steered_without_a_detector_still_follows_the_voice() -> None:
    det = Detector(ok=False)
    motion = Motion(lambda p: None, hz=1000, clock=FakeClock())
    voice = Voice()
    track = Follow(motion, det.read, det.switch, voice, poll_s=0.001, steer_on_robot=True)
    assert await track.start() == (None, False)
    await _until(lambda: track._target is not None)
    voice.event = SoundEvent("speaker", 40.0, 1.0)
    await _until(lambda: track._target is not None and track._target["yaw"] > 30)
    track.stop()
