"""Where did the last loud sound and the user's last sentence come from? (F7, on request.)

A poller reads the robot's direction of arrival (DoA) at 10 Hz with the head yaw. Two events
are kept, the latest of each kind, each with a world yaw and a time:

- ``speaker``: the user's voice, averaged over their last utterance (local VAD start/stop).
- ``noise``: a loud mic chunk (e.g. something falling), averaged over the next 0.3 s.

Chunks and readings while Reachy speaks (plus an echo tail) are skipped, and events older than
MAX_AGE_S are never served. No network or SDK import: the DoA reader is injected.

Angles are degrees; yaw 0 = robot front, + = left (as in robot/motion.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from reachy_demo.robot.motion import LIMITS

log = logging.getLogger(__name__)

Kind = Literal["speaker", "noise"]
ReadDoa = Callable[[], tuple[float, bool] | None]  # (chip angle rad, speech) or None

POLL_S = 0.1  # the daemon refreshes its DoA cache at this rate
# shortcut: fixed PCM16 RMS (VAD_RMS talks at 500); calibrate with scripts/mic_check.py per room.
DEFAULT_LOUD_RMS = 3000.0
NOISE_COOLDOWN_S = 1.0  # one bang = one event, not one per chunk
SETTLE_S = 0.3  # DoA lags the sound: average readings this long after it
ECHO_TAIL_S = 0.5  # Windows playback latency + room echo after Reachy's own audio
MAX_AGE_S: dict[str, float] = {"speaker": 30.0, "noise": 120.0}
TURN_SPEED_DPS = 60.0  # slow, safe turns; wide turns get more time


@dataclass(frozen=True)
class Reading:
    at: float
    yaw: float
    speech: bool
    own_voice: bool


@dataclass(frozen=True)
class SoundEvent:
    kind: Kind
    yaw: float
    at: float


def wrap(deg: float) -> float:
    """Wrap to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def doa_to_yaw(angle_rad: float) -> float:
    """Chip angle (0 = left, pi/2 = front, pi = right) -> yaw relative to the head (+ = left)."""
    return wrap(90.0 - math.degrees(angle_rad))


def circular_mean(yaws: list[float]) -> float:
    """Mean of angles; a plain mean breaks across +-180."""
    r = [math.radians(y) for y in yaws]
    return math.degrees(math.atan2(sum(map(math.sin, r)), sum(map(math.cos, r))))


def plan_turn(
    target: float, body_now: float, head_max: float = LIMITS["yaw"]
) -> tuple[float, float] | None:
    """(body_yaw, head_yaw) facing `target` inside the motion limits, or None if out of reach.

    The body stays put if the head alone can reach (`head_max` from the body); otherwise it turns
    just enough.
    """
    body_max = LIMITS["body_yaw"]
    target = wrap(target)
    body = min(max(body_now, -body_max), body_max)
    if abs(target - body) > head_max:
        body = min(max(target - math.copysign(head_max, target - body), -body_max), body_max)
    if abs(target - body) > head_max + 1e-6:
        return None
    return body, target


def turn_seconds(swing_deg: float, minimum: float) -> float:
    return max(minimum, swing_deg / TURN_SPEED_DPS)


def describe(yaw: float) -> str:
    """Where `yaw` is, from the robot's point of view."""
    deg = wrap(yaw)
    if abs(deg) < 15:
        return "in front of me"
    if abs(deg) > 165:
        return "behind me"
    side = "left" if deg > 0 else "right"
    behind = ", behind me" if abs(deg) > 90 else ""
    return f"about {round(abs(deg) / 5) * 5} degrees to my {side}{behind}"


class SoundEvents:
    """Thread-safe latest sound events, fed by mic chunks, VAD state and DoA polls."""

    def __init__(self, loud_rms: float = DEFAULT_LOUD_RMS) -> None:
        self._loud_rms = loud_rms
        self._lock = threading.Lock()
        self._readings: deque[Reading] = deque(maxlen=300)  # 30 s: one long utterance
        self._events: dict[str, SoundEvent] = {}  # latest per kind; a new one replaces it
        self._speech_start: float | None = None
        self._speaking = False
        self._pending_noise: float | None = None
        self._last_noise = -math.inf
        self._own_voice_until = -math.inf  # monotonic end of Reachy's queued playback

    # --- inputs ------------------------------------------------------------------------

    def mark_own_voice(self, until: float) -> None:
        """Reachy's speaker plays until `until` (monotonic); called per played piece."""
        self._own_voice_until = max(self._own_voice_until, until)

    def own_voice(self, now: float) -> bool:
        return now < self._own_voice_until + ECHO_TAIL_S

    def feed(self, chunk: bytes, speaking: bool, now: float) -> None:
        """One 24 kHz PCM16 mic chunk and the VAD state after it."""
        if speaking != self._speaking:
            self._speaking = speaking
            (self._speech_started if speaking else self._speech_stopped)(now)
        if self.own_voice(now):
            return
        x = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype="<i2").astype(np.float64)
        if not len(x) or float(np.sqrt(np.mean(x**2))) < self._loud_rms:
            return
        with self._lock:
            if now - self._last_noise >= NOISE_COOLDOWN_S:
                self._last_noise = self._pending_noise = now

    def _speech_started(self, now: float) -> None:
        with self._lock:
            self._speech_start = now

    def _speech_stopped(self, now: float) -> None:
        with self._lock:
            start, self._speech_start = self._speech_start, None
            if start is None:
                return
            window = [r for r in self._readings if start <= r.at <= now and not r.own_voice]
            # The chip's speech flag gates the average; use every reading if it never fired.
            yaws = [r.yaw for r in window if r.speech] or [r.yaw for r in window]
            if yaws:
                self._events["speaker"] = SoundEvent("speaker", circular_mean(yaws), start)

    def add_reading(self, doa: tuple[float, bool] | None, head_yaw: float, now: float) -> None:
        """One DoA poll; also settles a pending loud sound."""
        with self._lock:
            if doa is not None:
                # [verify] the mic array sits in the head, so its angle is relative to the head.
                angle, speech = doa
                yaw = wrap(head_yaw + doa_to_yaw(angle))
                self._readings.append(Reading(now, yaw, speech, self.own_voice(now)))
            pending = self._pending_noise
            if pending is None or now < pending + SETTLE_S:
                return
            self._pending_noise = None
            yaws = [
                r.yaw for r in self._readings if pending - POLL_S <= r.at <= now and not r.own_voice
            ]
            if yaws:
                self._events["noise"] = SoundEvent("noise", circular_mean(yaws), pending)

    # --- output ------------------------------------------------------------------------

    def latest(self, kind: Kind, now: float) -> SoundEvent | None:
        """The latest event of `kind`, unless older than MAX_AGE_S."""
        with self._lock:
            event = self._events.get(kind)
        return event if event is not None and now - event.at <= MAX_AGE_S[kind] else None

    # --- poller ------------------------------------------------------------------------

    async def run(
        self,
        read_doa: ReadDoa,
        head_yaw: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Poll DoA until cancelled. A failed read skips that reading (fail small)."""
        warned = False
        while True:
            try:
                doa = await asyncio.to_thread(read_doa)
            except (OSError, ValueError, KeyError, TypeError) as e:
                if not warned:
                    log.warning("DoA read failed (sound direction degraded): %s", e)
                    warned = True
                doa = None
            self.add_reading(doa, head_yaw(), clock())
            await asyncio.sleep(POLL_S)


async def with_sound(
    main: Awaitable[None], sound: SoundEvents, read_doa: ReadDoa, head_yaw: Callable[[], float]
) -> None:
    """Run the DoA poller beside `main`; stop it when `main` ends."""
    poller = asyncio.create_task(sound.run(read_doa, head_yaw))
    try:
        await main
    finally:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller
