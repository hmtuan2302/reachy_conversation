from datetime import datetime, timedelta, timezone

from reachy_demo.memory.recall import recall
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore, Speaker

TZ = timezone(timedelta(hours=7))
NOW = datetime(2026, 11, 17, 19, 0, tzinfo=TZ)
DAY = 86400.0


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _store() -> tuple[MemoryStore, Clock]:
    c = Clock(NOW.timestamp())
    return MemoryStore(":memory:", c), c


def _fact(s: MemoryStore, text: str, subject: str = "Tuan", person: int = OWNER_ID) -> int:
    return s.add_fact(subject_person_id=person, subject_text=subject, text=text)


def test_finds_fact_with_source_and_age_tag() -> None:
    s, _ = _store()
    _fact(s, "Tuan has a dog named Bin.")
    r = recall(s, OWNER, "dog name", now=NOW)
    assert r == {"found": True, "results": ["fact: Tuan has a dog named Bin. [said, just now]"]}


def test_abstains_on_no_match_and_on_stopwords_only() -> None:
    s, _ = _store()
    _fact(s, "Tuan has a dog named Bin.")
    assert recall(s, OWNER, "favourite colour", now=NOW) == {"found": False}
    assert recall(s, OWNER, "what is my", now=NOW) == {"found": False}
    assert recall(s, OWNER, "", now=NOW) == {"found": False}


def test_counts_use() -> None:
    s, _ = _store()
    fid = _fact(s, "Tuan has a dog named Bin.")
    recall(s, OWNER, "dog", now=NOW)
    f = s.fact(fid)
    assert f is not None and f.use_count == 1


def test_time_window_lists_conversations_without_matching_words() -> None:
    s, _ = _store()
    y = NOW.timestamp() - DAY
    s.add_episode(
        session_id="a",
        start=y,
        end=y + 600,
        person_ids=[OWNER_ID],
        summary="Talked about the exam and Da Lat.",
        topics=[],
    )
    s.add_episode(
        session_id="b",
        start=y - 5 * DAY,
        end=y - 5 * DAY,
        person_ids=[OWNER_ID],
        summary="Talked about cooking.",
        topics=[],
    )
    r = recall(s, OWNER, "what did we talk about", when="yesterday", now=NOW)
    assert r["found"] is True
    assert r["results"] == ["conversation Mon 16 Nov 2026 19:00: Talked about the exam and Da Lat."]


def test_old_fact_found_in_its_time_window_marked_no_longer_true() -> None:
    s, c = _store()
    c.t = NOW.timestamp() - 400 * DAY
    old = _fact(s, "Tuan lives in Saigon.")
    c.t = NOW.timestamp() - 10 * DAY
    new = _fact(s, "Tuan lives in Hanoi.")
    s.invalidate_fact(old, valid_to=None, replaced_by=new)
    c.t = NOW.timestamp()
    r = recall(s, OWNER, "lives", when="2025-11-20", now=NOW)
    assert len(r["results"]) == 1 and "Saigon" in r["results"][0]
    assert "no longer true since" in r["results"][0]
    assert ["Hanoi" in x for x in recall(s, OWNER, "lives", now=NOW)["results"]] == [True]


def test_unparsed_time_searches_all_time_with_note() -> None:
    s, _ = _store()
    _fact(s, "Tuan has a dog named Bin.")
    r = recall(s, OWNER, "dog", when="the other day", now=NOW)
    assert r["found"] is True and "could not understand" in r["note"]


def test_person_filter_by_name_and_me() -> None:
    s, _ = _store()
    s._db.execute("INSERT INTO people(id, role, name) VALUES (2, 'guest', 'Lan')")
    _fact(s, "Tuan likes green tea.")
    _fact(s, "Lan likes black tea.", subject="Lan", person=2)
    assert recall(s, OWNER, "tea", person="Lan", now=NOW)["results"] == [
        "fact: Lan likes black tea. [said, just now]"
    ]
    assert recall(s, OWNER, "tea", person="me", now=NOW)["results"] == [
        "fact: Tuan likes green tea. [said, just now]"
    ]


def test_guest_cannot_read_owner_facts() -> None:
    s, _ = _store()
    s._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    _fact(s, "Tuan's address is 12 Ly Thuong Kiet.")
    assert recall(s, Speaker(2, "guest"), "address", now=NOW) == {"found": False}


def test_near_duplicates_collapse() -> None:
    s, _ = _store()
    _fact(s, "Tuan has a dog named Bin.")
    _fact(s, "Tuan has a dog named Bin!")
    assert len(recall(s, OWNER, "dog", now=NOW)["results"]) == 1


def test_limit_is_clamped() -> None:
    s, _ = _store()
    for i in range(15):
        _fact(s, f"Tuan visited city number{i} with the dog.")
    assert len(recall(s, OWNER, "dog", limit=100, now=NOW)["results"]) == 10


# -- hybrid (vectors) ------------------------------------------------------------------------

from reachy_demo.memory.vectors import VectorIndex  # noqa: E402
from tests.unit.fake_embedder import FakeEmbedder  # noqa: E402

AI = FakeEmbedder({"copilot": "aitool", "claude": "aitool", "tool": "aitool", "ai": "aitool"})


def _indexed(s: MemoryStore, *ids: int) -> VectorIndex:
    idx = VectorIndex(AI.model)
    for i in ids:
        f = s.fact(i)
        assert f is not None
        idx.put("fact", i, AI.embed([f.text], "doc")[0])
    return idx


def _q(text: str):  # type: ignore[no-untyped-def]
    return AI.embed([text], "query")[0]


def test_recall_finds_paraphrase_without_shared_words() -> None:
    s, _ = _store()
    fid = _fact(s, "Tuan codes with Claude.")
    q = "which AI tool"
    r = recall(s, OWNER, q, now=NOW, index=_indexed(s, fid), query_vec=_q(q))
    assert r["found"] and "Claude" in r["results"][0]
    assert recall(s, OWNER, q, now=NOW)["found"] is False  # keywords alone miss it


def test_recall_vector_path_respects_guest_scope() -> None:
    s, _ = _store()
    s._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    fid = _fact(s, "Tuan codes with Claude.")
    guest = Speaker(2, "guest")
    r = recall(s, guest, "AI tool", now=NOW, index=_indexed(s, fid), query_vec=_q("AI tool"))
    assert r == {"found": False}


def test_recall_time_window_returns_expired_by_vector() -> None:
    s, c = _store()
    c.t = NOW.timestamp() - 10 * DAY
    old = _fact(s, "Tuan uses Copilot.")
    c.t = NOW.timestamp()
    new = _fact(s, "Tuan uses Claude.")
    s.invalidate_fact(old, valid_to=None, replaced_by=new)
    idx = _indexed(s, old, new)
    r = recall(s, OWNER, "AI tool", now=NOW, when="last week", index=idx, query_vec=_q("AI tool"))
    assert any("Copilot" in line and "no longer true" in line for line in r["results"])
