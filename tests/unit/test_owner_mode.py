import asyncio
import io
from collections.abc import AsyncIterator

from reachy_demo.agent.tools.owner import (
    GUEST,
    NOT_ENROLLED,
    NOT_OWNER,
    UNSURE,
    Heard,
    OwnerMode,
    OwnerTools,
    gated_turns,
)
from reachy_demo.memory.store import OWNER
from reachy_demo.voice.speaker_id import Verdict


class FakeGate:
    def __init__(self, *verdicts: Verdict, score: float | None = 0.5) -> None:
        self.verdicts = list(verdicts)
        self.takes = 0
        self.score = score

    def take(self) -> tuple[Verdict, float | None]:
        self.takes += 1
        return self.verdicts.pop(0), self.score


async def _turns(*texts: str) -> AsyncIterator[str]:
    for t in texts:
        yield t


async def _run(
    mode: OwnerMode, *texts: str, said: list[str] | None = None, stop: bool | None = None
) -> tuple[list[str], str]:
    out = io.StringIO()

    async def say(text: str) -> None:
        if said is not None:
            said.append(text)

    async def is_stop(text: str) -> bool:
        assert stop is not None, "classified a turn that should not be"
        return stop

    got = [t async for t in gated_turns(_turns(*texts), mode, out, say=say, is_stop=is_stop)]
    await asyncio.sleep(0)  # let the background "say it again" run
    return got, out.getvalue()


def _mode(*verdicts: Verdict, on: bool = False) -> tuple[OwnerMode, FakeGate]:
    gate = FakeGate(*verdicts)
    mode = OwnerMode(gate)
    mode.on = on
    return mode, gate


async def test_mode_off_yields_all() -> None:
    mode, _ = _mode("guest", "unsure", "owner")
    got, _ = await _run(mode, "a", "b", "c")
    assert got == ["a", "b", "c"]


async def test_mode_on_drops_guest() -> None:
    mode, _ = _mode("guest", "owner", on=True)
    got, log = await _run(mode, "dance", "hi")
    assert got == ["hi"]
    assert "ignored guest: dance" in log


async def test_mode_on_unsure_asks_again() -> None:
    said: list[str] = []
    mode, _ = _mode("unsure", on=True)
    got, log = await _run(mode, "dance", said=said)
    assert got == [] and said == [UNSURE]
    assert "unsure: dance" in log


async def test_mode_off_still_takes_each_turn() -> None:
    mode, gate = _mode("owner", "guest", "owner")
    await _run(mode, "a", "b", "c")
    assert gate.takes == 3


async def test_logs_each_score() -> None:
    mode, _ = _mode("owner")
    _, log = await _run(mode, "a")
    assert "(voice: owner 0.50)" in log


def test_speaker_follows_mode() -> None:
    mode, _ = _mode()
    mode.last = "guest"
    assert mode.speaker() == OWNER
    mode.on = True
    assert mode.speaker() == GUEST
    mode.last = "owner"
    assert mode.speaker() == OWNER


async def test_toggle_on_and_off() -> None:
    mode, _ = _mode()
    tools = OwnerTools(mode)
    assert await tools.call("owner_mode", '{"on": true}') == {"ok": True, "owner_mode": True}
    assert mode.on
    assert await tools.call("owner_mode", '{"on": false}') == {"ok": True, "owner_mode": False}
    assert not mode.on


async def test_guest_cannot_toggle() -> None:
    mode, _ = _mode()
    mode.last = "unsure"
    assert await OwnerTools(mode).call("owner_mode", '{"on": true}') == NOT_OWNER
    assert not mode.on


async def test_bad_arguments() -> None:
    mode, _ = _mode()
    result = await OwnerTools(mode).call("owner_mode", '{"on": "yes"}')
    assert "bad_arguments" in result["error"]


async def test_not_enrolled() -> None:
    assert await OwnerTools(OwnerMode(None)).call("owner_mode", '{"on": true}') == NOT_ENROLLED


async def test_no_gate_counts_as_owner() -> None:
    mode = OwnerMode(None)
    mode.on = True
    got, _ = await _run(mode, "a", "b")
    assert got == ["a", "b"]


async def test_turns_carry_their_verdict_and_leave_last_alone() -> None:
    mode, _ = _mode("guest", "owner")
    got, _ = await _run(mode, "a", "b")
    assert [(t, t.verdict) for t in got if isinstance(t, Heard)] == [("a", "guest"), ("b", "owner")]
    assert mode.last == "owner"  # untouched: set only when an answer starts (begin)


def test_begin_sets_last_from_the_answered_turn() -> None:
    mode, _ = _mode()
    seen = []
    for text in (Heard("only follow me", "guest"), "typed text"):
        mode.begin(text)
        seen.append(mode.last)
    assert seen == ["guest", "owner"]


async def test_dropped_guest_does_not_change_last() -> None:
    mode, _ = _mode("guest", on=True)
    mode.begin(Heard("dance", "owner"))
    await _run(mode, "chatter")
    assert mode.speaker() == OWNER


async def test_short_stop_passes_in_owner_mode() -> None:
    gate = FakeGate("unsure", score=None)
    mode = OwnerMode(gate)
    mode.on = True
    got, _ = await _run(mode, "stop", stop=True)
    assert got == ["stop"] and isinstance(got[0], Heard) and got[0].verdict == "unsure"


async def test_short_non_stop_asks_again() -> None:
    said: list[str] = []
    gate = FakeGate("unsure", score=None)
    mode = OwnerMode(gate)
    mode.on = True
    got, _ = await _run(mode, "dance", said=said, stop=False)
    assert got == [] and said == [UNSURE]


async def test_say_error_does_not_end_turns() -> None:
    from reachy_demo.voice.tts import TTSError

    async def say(text: str) -> None:
        raise TTSError("network down")

    mode, _ = _mode("unsure", "owner", on=True)
    out = io.StringIO()
    got = [t async for t in gated_turns(_turns("a", "b"), mode, out, say=say)]
    await asyncio.sleep(0)
    assert got == ["b"] and "speech error: network down" in out.getvalue()


async def test_say_does_not_block_turns() -> None:
    async def say(text: str) -> None:
        await asyncio.Event().wait()  # plays "forever"

    mode, _ = _mode("unsure", "owner", on=True)
    async with asyncio.timeout(1):
        got = [t async for t in gated_turns(_turns("a", "b"), mode, io.StringIO(), say=say)]
    assert got == ["b"]
