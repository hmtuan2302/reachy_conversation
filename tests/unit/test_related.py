from datetime import UTC, datetime

import numpy as np

from reachy_demo.memory.related import PUSH_TOP, related_block
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore, Speaker
from reachy_demo.memory.vectors import VectorIndex
from tests.unit.fake_embedder import FakeEmbedder

NOW = datetime(2026, 11, 17, 19, 0, tzinfo=UTC)
EMB = FakeEmbedder({"python": "code", "bug": "code", "websocket": "code", "claude": "code"})
NONE: frozenset[int] = frozenset()


def _setup() -> tuple[MemoryStore, VectorIndex]:
    s = MemoryStore(":memory:", lambda: NOW.timestamp())
    s._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    return s, VectorIndex(EMB.model)


def _fact(s: MemoryStore, idx: VectorIndex, text: str, pid: int = OWNER_ID, **kw: object) -> int:
    fid = s.add_fact(subject_person_id=pid, subject_text="Tuan", text=text, **kw)  # type: ignore[arg-type]
    idx.put("fact", fid, EMB.embed([text], "doc")[0])
    return fid


def _q(text: str) -> np.ndarray:
    v: np.ndarray = EMB.embed([text], "query")[0]
    return v


def _block(
    s: MemoryStore, idx: VectorIndex, text: str, speaker: Speaker = OWNER, **kw: frozenset[int]
) -> str:
    return related_block(
        s, idx, speaker, _q(text),
        exclude_facts=kw.get("facts", NONE), exclude_episodes=kw.get("episodes", NONE), now=NOW,
    )  # fmt: skip


def test_related_picks_top_above_threshold_only() -> None:
    s, idx = _setup()
    for t in ["codes in Python", "fixed a websocket bug", "uses Claude", "python bug again"]:
        _fact(s, idx, t)
    _fact(s, idx, "sister Mai lives in Hanoi")
    block = _block(s, idx, "python bug websocket")
    assert block.startswith("[related memories")
    lines = block.splitlines()[1:]
    assert len(lines) == PUSH_TOP and not any("Mai" in line for line in lines)


def test_related_empty_when_nothing_close() -> None:
    s, idx = _setup()
    _fact(s, idx, "sister Mai lives in Hanoi")
    assert _block(s, idx, "python bug") == ""


def test_related_skips_briefing_ids_and_sensitive() -> None:
    s, idx = _setup()
    shown = _fact(s, idx, "codes in Python")
    _fact(s, idx, "python bug password", sensitive=True)
    assert _block(s, idx, "python bug", facts=frozenset({shown})) == ""


def test_related_guest_never_sees_owner_facts() -> None:
    s, idx = _setup()
    _fact(s, idx, "codes in Python")
    assert _block(s, idx, "python", speaker=Speaker(2, "guest")) == ""


def test_related_does_not_touch_facts() -> None:
    s, idx = _setup()
    fid = _fact(s, idx, "codes in Python")
    assert "Python" in _block(s, idx, "python")
    f = s.fact(fid)
    assert f is not None and f.use_count == 0


def test_related_includes_episodes_and_cuts_long_lines() -> None:
    s, idx = _setup()
    long = "Fixed a websocket bug " + "x " * 200
    t = NOW.timestamp()
    e = s.add_episode(
        session_id="a", start=t - 86400, end=t, person_ids=[OWNER_ID], summary=long, topics=[]
    )
    idx.put("episode", e, EMB.embed(["Fixed a websocket bug"], "doc")[0])
    block = _block(s, idx, "websocket bug")
    line = block.splitlines()[1]
    assert line.startswith("- conversation") and len(line) <= 242
