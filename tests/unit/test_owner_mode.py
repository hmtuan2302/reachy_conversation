import io
from collections.abc import AsyncIterator

from reachy_demo.agent.tools.owner import (
    GUEST,
    NOT_ENROLLED,
    NOT_OWNER,
    UNSURE,
    OwnerMode,
    OwnerTools,
    gated_turns,
)
from reachy_demo.memory.store import OWNER
from reachy_demo.voice.speaker_id import Verdict


class FakeGate:
    def __init__(self, *verdicts: Verdict) -> None:
        self.verdicts = list(verdicts)
        self.takes = 0

    def take(self) -> tuple[Verdict, float | None]:
        self.takes += 1
        return self.verdicts.pop(0), 0.5


async def _turns(*texts: str) -> AsyncIterator[str]:
    for t in texts:
        yield t


async def _run(
    mode: OwnerMode, *texts: str, said: list[str] | None = None
) -> tuple[list[str], str]:
    out = io.StringIO()

    async def say(text: str) -> None:
        if said is not None:
            said.append(text)

    got = [t async for t in gated_turns(_turns(*texts), mode, out, say=say)]
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
