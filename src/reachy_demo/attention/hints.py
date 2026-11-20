"""Where to look when the face isn't in view: reader wrappers around a face reader for Follow.

- `with_body`: no face, but a person's body is in view (back turned, too close, head above the
  frame): aim at where their head must be, the top of the person box, or above the frame when
  the box touches its top edge.
- `chasing`: the face was just lost at the edge of the frame or while moving fast: keep turning
  the way it went for CHASE_S, so it shows up again instead of the head stopping.

Readings are (x right, y down, in [-1, 1], optional frame time) as Follow takes them; values
past ±1 mean "beyond the frame edge" and make Follow turn further.
"""

from __future__ import annotations

import time
from collections.abc import Callable

Reading = tuple[float, float] | tuple[float, float, float]
Read = Callable[[], Reading | None]
ReadHeads = Callable[[], list[tuple[float, float, float]]]  # head points of every person seen

CHASE_S = 1.5  # keep turning this long after losing the face
EDGE = 0.5  # lost beyond this share of the half-frame: it left by that side
FAST = 1.0  # or moving faster than this (image units per second) when lost
BEYOND = 1.3  # a chase reading this far out: Follow turns about 25-30° per reading
MAX_BODY_JUMP = 0.8  # a body this far sideways from the last face is someone else


def with_body(read_face: Read, read_heads: ReadHeads) -> Read:
    """`read_face`, or, while it sees nothing, the head point of the body nearest the last face."""
    last: list[tuple[float, float] | None] = [None]

    def read() -> Reading | None:
        face = read_face()
        if face is not None:
            last[0] = (face[0], face[1])
            return face
        heads = read_heads()
        if not heads:
            return None
        ref = last[0] or (0.0, 0.0)

        def dist(h: tuple[float, float, float]) -> float:
            return abs(h[0] - ref[0])  # sideways only: a head above the frame is far in y

        best = min(heads, key=dist)
        if last[0] is not None and dist(best) > MAX_BODY_JUMP:
            return None
        return best

    return read


def chasing(read: Read, clock: Callable[[], float] = time.monotonic) -> Read:
    """`read`, but for CHASE_S after a loss at the edge or at speed, a point beyond that edge."""
    # (x, y, clock time) of the last two readings
    seen: list[tuple[float, float, float]] = []

    def read_chasing() -> Reading | None:
        r = read()
        now = clock()
        if r is not None:
            seen[:] = [*seen[-1:], (r[0], r[1], now)]
            return r
        if not seen or now - seen[-1][2] > CHASE_S:
            return None
        x, y, t = seen[-1]
        vx = (x - seen[0][0]) / (t - seen[0][2]) if len(seen) == 2 and t > seen[0][2] else 0.0
        if abs(vx) > FAST:
            side = 1.0 if vx > 0 else -1.0
        elif abs(x) > EDGE:
            side = 1.0 if x > 0 else -1.0
        elif abs(y) > EDGE:  # left by the top or bottom: chase vertically
            return x, (BEYOND if y > 0 else -BEYOND)
        else:
            return None  # lost in the middle (turned away, covered): hold, don't guess
        return side * BEYOND, max(-1.0, min(1.0, y))  # no frame time: from the current pose

    return read_chasing
