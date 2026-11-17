import asyncio
import math

import numpy as np
import pytest

from reachy_demo.audio.sound_events import (
    ECHO_TAIL_S,
    SETTLE_S,
    SoundEvents,
    describe,
    doa_to_yaw,
    plan_turn,
)
from reachy_demo.robot.motion import LIMITS

LOUD = np.full(4800, 8000, dtype="<i2").tobytes()  # 200 ms at 24 kHz
QUIET = np.full(4800, 300, dtype="<i2").tobytes()
LEFT, FRONT, RIGHT = 0.0, math.pi / 2, math.pi  # chip angles


def _bang(s: SoundEvents, t: float, angle: float = LEFT) -> None:
    s.add_reading((angle, False), 0.0, t)
    s.feed(LOUD, False, t)
    s.add_reading((angle, False), 0.0, t + SETTLE_S)


def test_doa_maps_left_front_right() -> None:
    assert doa_to_yaw(LEFT) == pytest.approx(90)
    assert doa_to_yaw(FRONT) == pytest.approx(0)
    assert doa_to_yaw(RIGHT) == pytest.approx(-90)


def test_loud_sound_stored_after_settling() -> None:
    s = SoundEvents()
    s.add_reading((LEFT, False), 0.0, 10.0)
    s.feed(LOUD, False, 10.0)
    s.add_reading((LEFT, False), 0.0, 10.1)
    assert s.latest("noise", 10.1) is None  # DoA not settled yet
    s.add_reading((LEFT, False), 0.0, 10.0 + SETTLE_S)
    event = s.latest("noise", 11.0)
    assert event is not None
    assert event.yaw == pytest.approx(90)
    assert event.at == 10.0


def test_quiet_sound_not_stored() -> None:
    s = SoundEvents()
    s.add_reading((LEFT, False), 0.0, 10.0)
    s.feed(QUIET, False, 10.0)
    s.add_reading((LEFT, False), 0.0, 11.0)
    assert s.latest("noise", 11.0) is None


def test_own_voice_and_its_echo_are_skipped() -> None:
    s = SoundEvents()
    s.mark_own_voice(10.0)
    _bang(s, 10.0 + ECHO_TAIL_S / 2)
    assert s.latest("noise", 12.0) is None
    _bang(s, 10.0 + ECHO_TAIL_S + 0.1)  # after the tail: heard
    assert s.latest("noise", 12.0) is not None


def test_old_events_expire() -> None:
    s = SoundEvents()
    _bang(s, 10.0)
    assert s.latest("noise", 100.0) is not None
    assert s.latest("noise", 200.0) is None  # > 2 min


def test_speaker_direction_from_vad_window_adds_head_yaw() -> None:
    """The mic angle is relative to the head: a head turned 30° left shifts the world yaw."""
    s = SoundEvents()
    s.feed(QUIET, True, 10.0)  # VAD rises: speech starts
    s.add_reading((RIGHT, True), 30.0, 10.1)
    s.add_reading((RIGHT, True), 30.0, 10.2)
    s.feed(QUIET, False, 10.3)  # VAD falls: speech ends
    event = s.latest("speaker", 10.3)
    assert event is not None and event.yaw == pytest.approx(-60)
    assert s.latest("speaker", 10.3 + 31) is None  # voice expires after 30 s


def test_only_latest_noise_kept_and_voice_survives_many_noises() -> None:
    s = SoundEvents()
    s.feed(QUIET, True, 0.0)
    s.add_reading((RIGHT, True), 0.0, 0.1)
    s.feed(QUIET, False, 0.2)
    for t in range(1, 15):
        _bang(s, float(t))
    noise = s.latest("noise", 15.0)
    assert noise is not None and noise.at == 14.0
    assert s.latest("speaker", 15.0) is not None


def test_plan_head_only_when_it_reaches() -> None:
    assert plan_turn(30, 0) == pytest.approx((0, 30))


def test_plan_turns_body_just_enough_within_limits() -> None:
    plan = plan_turn(150, 0)
    assert plan is not None
    body, head = plan
    assert body == pytest.approx(150 - LIMITS["yaw"])
    assert abs(body) <= LIMITS["body_yaw"] and abs(head - body) <= LIMITS["yaw"] + 1e-6


def test_plan_refuses_out_of_reach() -> None:
    assert plan_turn(180, 0) is None  # needs > body 120 + head 45
    assert plan_turn(-170, 0) is None


def test_describe() -> None:
    assert describe(0) == "in front of me"
    assert describe(40) == "about 40 degrees to my left"
    assert describe(-120) == "about 120 degrees to my right, behind me"
    assert describe(180) == "behind me"


async def test_poller_survives_failed_reads() -> None:
    """A network error skips one reading; later readings still land."""
    s = SoundEvents()
    reads: list[Exception | tuple[float, bool]] = [OSError("down"), (LEFT, True)]

    def read() -> tuple[float, bool] | None:
        r = reads.pop(0) if reads else (LEFT, True)
        if isinstance(r, Exception):
            raise r
        return r

    s.feed(QUIET, True, 4.0)
    task = asyncio.create_task(s.run(read, lambda: 0.0, clock=lambda: 5.0))
    await asyncio.sleep(0.25)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    s.feed(QUIET, False, 6.0)
    event = s.latest("speaker", 6.0)
    assert event is not None and event.yaw == pytest.approx(90)
