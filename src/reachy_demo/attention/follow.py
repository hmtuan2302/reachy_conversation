"""Follow a target with head and body until told to stop (F4 tracking, on request).

Two inputs steer one target:

- **Target in the image**: a reader reports its centre, about 10-20 times a second. For "track
  me" it is the robot daemon's own face detector (YuNet, on the robot, read over REST); for
  "track my phone" it is perception/objects.py (detector + tracker on the laptop).
- **Voice** (people only): the direction of the user's last sentence (audio/sound_events.py).
  It finds the speaker when no face is in view, and switches to a new speaker who talks from
  elsewhere.

Poses still come only from robot/motion.py: tracking is one long move (like a dance) whose pose
eases toward the target this class updates. So `stop_head` and a spoken stop end it, other moves
are refused while it runs, and every pose is clamped. The head turns alone up to HEAD_REACH from
the body; past that the body turns too.

Angles are degrees; yaw 0 = robot front, + = left; pitch + = down (as in robot/motion.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Callable

from reachy_demo.audio.sound_events import SoundEvents, plan_turn, wrap
from reachy_demo.perception.head_pose import TAN_HALF_H, TAN_HALF_V
from reachy_demo.robot.motion import AXES, LIMITS, NEUTRAL, Motion, Move, Pose

log = logging.getLogger(__name__)

ReadFace = Callable[[], tuple[float, float] | None]  # face centre, x right / y down, in [-1, 1]
SetTracking = Callable[[bool], bool]  # daemon face detector on/off; False: no camera

GAIN = 0.6  # share of the face offset corrected per reading; < 1 since frames lag the head
HEAD_REACH = 25.0  # head turns alone this far from the body, then the body turns too
PITCH_LIMIT = 25.0
SPEED_DPS = 90.0  # fastest the target is chased
POLL_S = 0.1
LOST_S = 1.0  # no face this long: the next sentence turns Reachy toward the voice
VOICE_SWITCH_DEG = 35.0  # a sentence from this far off: look at that speaker (DoA error ~20°)
ANCHOR_S = 4.0  # after a voice turn, ignore faces more than ANCHOR_DEG from the voice this long
ANCHOR_DEG = 25.0
MAX_TRACK_S = 1800.0  # shortcut: tracking ends after 30 min like a dance does; raise if needed
STEERED = ("yaw", "pitch", "body_yaw")


class Follow:
    def __init__(
        self,
        motion: Motion,
        read_face: ReadFace,
        set_tracking: SetTracking,
        sound: SoundEvents | None = None,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = POLL_S,
        give_up_s: float | None = None,
        on_lost: Callable[[], None] | None = None,
        name: str = "the user",
    ) -> None:
        """`read_face`: the target's centre (x right, y down, in [-1, 1]) or None.
        `set_tracking`: turns its detector on (True if it runs) or off.
        `give_up_s`: stop after the target is unseen this long, then call `on_lost`."""
        self._give_up_s = give_up_s
        self._on_lost = on_lost
        self.name = name
        self._motion = motion
        self._read_face = read_face
        self._set_tracking = set_tracking
        self._sound = sound
        self._clock = clock
        self._poll_s = poll_s
        self._task: asyncio.Task[None] | None = None
        self._move: Move | None = None
        self._reset()

    def _reset(self) -> None:
        self._target: Pose | None = None  # set from the start pose on the move's first tick
        self._pose: Pose = dict(NEUTRAL)
        self._t = 0.0
        self._last_face = -math.inf
        self._started = self._clock()
        self._voice_seen = -math.inf  # start time of the last sentence acted on
        self._anchor: tuple[float, float] | None = None  # (voice yaw, when)

    @property
    def blocked(self) -> bool:
        """Another long move (dance, mimic, other tracking) holds the head."""
        return self._motion.dancing

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> tuple[str | None, bool]:
        """Start following. Returns (error, face detector on); voice-only without a camera."""
        if self.active:
            return None, True
        try:
            faces = await asyncio.to_thread(self._set_tracking, True)
        except (OSError, ValueError, KeyError) as e:  # fail small: follow the voice only
            log.warning("face detector unavailable: %s", e)
            faces = False
        self._reset()
        self._move = Move(f"follow {self.name}", True, MAX_TRACK_S, self._pose_at, dict(NEUTRAL))
        if error := self._motion.submit(self._move):
            await self._detector_off()
            return error, faces
        self._task = asyncio.create_task(self._run())
        log.info("following %s (detector %s)", self.name, "on" if faces else "off")
        return None, faces

    def stop(self) -> None:
        """End tracking: the head eases back to neutral. The loop then turns the detector off."""
        if self.active:
            self._motion.stop()

    # --- the long move ------------------------------------------------------------------

    def _pose_at(self, t: float, start: Pose) -> Pose:
        if self._target is None:  # first tick: hold where the head is
            self._target, self._pose, self._t = dict(start), dict(start), t
        step = SPEED_DPS * max(t - self._t, 0.0)
        self._t = t
        self._pose = {
            k: self._pose[k] + max(-step, min(step, self._target[k] - self._pose[k])) for k in AXES
        }
        return dict(self._pose)

    # --- the loop -----------------------------------------------------------------------

    async def _run(self) -> None:
        warned = False
        try:
            while self._motion.dancing:  # stop/close/an error end the move, and so this loop
                try:
                    face = await asyncio.to_thread(self._read_face)
                except Exception as e:  # noqa: BLE001 - fail small: skip one reading
                    if not warned:
                        log.warning("target read failed (skipping): %s", e)
                        warned = True
                    face = None
                now = self._clock()
                self.step(face, now)
                unseen = now - max(self._last_face, self._started)
                if self._give_up_s is not None and unseen > self._give_up_s:
                    log.info("lost %s for %.0f s, stopping", self.name, unseen)
                    self._motion.stop()
                    if self._on_lost:
                        self._on_lost()
                    break
                await asyncio.sleep(self._poll_s)
        finally:
            await self._detector_off()
            log.info("stopped following %s", self.name)

    async def _detector_off(self) -> None:
        with contextlib.suppress(OSError, ValueError, KeyError):
            await asyncio.to_thread(self._set_tracking, False)

    def step(self, face: tuple[float, float] | None, now: float) -> None:
        """Update the target from one face reading and the latest sentence's direction."""
        if self._target is None:
            return  # the move has not started yet
        pose = self._pose
        voice = self._sound.latest("speaker", now) if self._sound else None
        if voice is not None and voice.at > self._voice_seen:
            self._voice_seen = voice.at
            lost = now - self._last_face > LOST_S
            if lost or abs(wrap(voice.yaw - pose["yaw"])) > VOICE_SWITCH_DEG:
                log.info("turning toward the voice at %.0f°", voice.yaw)
                self._aim(voice.yaw, 0.0)
                self._anchor = (voice.yaw, now)
                return
        if face is None:
            return  # hold: the user may step out of view; their next sentence finds them
        x, y = face
        yaw = pose["yaw"] - math.degrees(math.atan(x * TAN_HALF_H))
        pitch = pose["pitch"] + math.degrees(math.atan(y * TAN_HALF_V))
        if (
            self._anchor is not None
            and now - self._anchor[1] < ANCHOR_S
            and abs(wrap(yaw - self._anchor[0])) > ANCHOR_DEG
        ):
            return  # someone else's face while turning to the speaker
        self._last_face = now
        self._aim(
            pose["yaw"] + GAIN * (yaw - pose["yaw"]), pose["pitch"] + GAIN * (pitch - pose["pitch"])
        )

    def _aim(self, yaw: float, pitch: float) -> None:
        assert self._target is not None
        reach = LIMITS["body_yaw"] + HEAD_REACH
        plan = plan_turn(max(-reach, min(reach, wrap(yaw))), self._target["body_yaw"], HEAD_REACH)
        assert plan is not None  # the clip above keeps it in reach
        body, head = plan
        self._target.update(
            yaw=head, body_yaw=body, pitch=max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch))
        )
