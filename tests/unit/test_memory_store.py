import sqlite3
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.memory import store as store_mod
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore, Speaker, match_query

GUEST = Speaker(2, "guest")
LATEST = len(store_mod.MIGRATIONS)


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _store(clock: Clock | None = None) -> MemoryStore:
    return MemoryStore(":memory:", clock or Clock())


def test_fresh_db_has_owner_and_schema_v1() -> None:
    s = _store()
    owner = s.person(OWNER_ID)
    assert owner is not None and owner.role == "owner" and owner.name is None
    assert s.schema_version == LATEST


def test_reopen_is_a_noop(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    s = MemoryStore(db, Clock())
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan likes tea.")
    s.close()
    s2 = MemoryStore(db, Clock())
    assert s2.schema_version == LATEST
    assert len(s2.search_facts("tea", OWNER)) == 1


def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "m.db"
    MemoryStore(db, Clock()).close()
    monkeypatch.setattr(
        store_mod, "MIGRATIONS", [*store_mod.MIGRATIONS, "CREATE TABLE x(a); BOGUS;"]
    )
    with pytest.raises(sqlite3.Error):
        MemoryStore(db, Clock())
    monkeypatch.undo()
    again = MemoryStore(db, Clock())
    assert again.schema_version == LATEST
    assert (tmp_path / f"m.db.bak-{LATEST}").exists()  # backup taken before the failed migration


def test_fts_ignores_diacritics_and_case() -> None:
    s = _store()
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuấn", text="Tuấn thích cà phê sữa đá.")
    assert len(s.search_facts("tuan ca phe", OWNER)) == 1


@pytest.mark.parametrize("q", ['"', "tea*", "AND OR NOT", "- tea", "(tea", "NEAR(tea)"])
def test_fts_operators_in_query_are_plain_words(q: str) -> None:
    s = _store()
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan likes tea.")
    s.search_facts(q, OWNER)  # must not raise


def test_match_query_drops_stopwords_and_empty() -> None:
    assert match_query("what is my") is None
    assert match_query("What is my dog's name?") == '"dog" OR "name"'


def test_guest_sees_only_own_and_shared_facts() -> None:
    s = _store()
    s._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    s.add_fact(
        subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan's address is secret street."
    )
    s.add_fact(subject_person_id=2, subject_text="Lan", text="Lan likes the street food.")
    assert [f.text for f, _ in s.search_facts("street", GUEST)] == ["Lan likes the street food."]
    assert len(s.search_facts("street", OWNER)) == 2


def test_invalidate_keeps_history_and_time_window_finds_it() -> None:
    clock = Clock(1000.0)
    s = _store(clock)
    old = s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan lives in Saigon.")
    clock.t = 5000.0
    new = s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan lives in Hanoi.")
    s.invalidate_fact(old, valid_to=None, replaced_by=new)
    assert [f.text for f, _ in s.search_facts("lives", OWNER)] == ["Tuan lives in Hanoi."]
    past = s.search_facts("lives", OWNER, start=0.0, end=2000.0)
    assert [f.text for f, _ in past] == ["Tuan lives in Saigon."]
    f = s.fact(old)
    assert f is not None and f.expired == 5000.0


def test_episodes_by_time_window_without_matching_words() -> None:
    s = _store()
    s.add_episode(
        session_id="a",
        start=100,
        end=200,
        person_ids=[1],
        summary="Talked about Da Lat.",
        topics=["trip"],
    )
    s.add_episode(
        session_id="b", start=900, end=950, person_ids=[1], summary="Talked about exams.", topics=[]
    )
    hits = s.search_episodes("what did we talk", OWNER, start=0, end=300)
    assert [e.summary for e, _ in hits] == ["Talked about Da Lat."]
    assert s.search_episodes("what did we", OWNER) == []


def test_sessions_without_episode() -> None:
    s = _store()
    s.log_event("turn", {"user": "hi"}, person_id=1, session_id="s1")
    s.log_event("turn", {"user": "yo"}, person_id=1, session_id="s2")
    s.add_episode(session_id="s1", start=0, end=1, person_ids=[1], summary="x", topics=[])
    assert s.sessions_without_episode(exclude=None) == [("s2", 1)]
    assert s.sessions_without_episode(exclude="s2") == []


def test_follow_ups_window() -> None:
    day = 86400.0
    clock = Clock(10 * day)
    s = _store(clock)
    s.add_follow_up(1, "Ask about the exam.", due=11 * day)
    s.add_follow_up(1, "Ask about the trip.", due=20 * day)
    s.add_follow_up(1, "Ask how the cat is.", due=None)
    texts = [f.text for f in s.open_follow_ups(1, now=10 * day)]
    assert texts == ["Ask how the cat is.", "Ask about the exam."]


def test_forget_person_deletes_everything_about_them() -> None:
    s = _store()
    s.set_name(OWNER_ID, "Tuan")
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan likes tea.")
    s.add_episode(session_id="a", start=0, end=1, person_ids=[1], summary="tea talk", topics=[])
    s.add_follow_up(1, "Ask about tea.", due=None)
    s.log_event("turn", {"user": "I like tea"}, person_id=1, session_id="a")
    s.forget_person(OWNER_ID)
    assert s.search_facts("tea", OWNER) == []
    assert s.search_episodes("tea", OWNER) == []
    assert s.all_open_follow_ups(1) == []
    assert s.last_turns(5) == []
    p = s.person(OWNER_ID)
    assert p is not None and p.name is None


def test_backup_copies_data(tmp_path: Path) -> None:
    s = _store()
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan likes tea.")
    s.backup(tmp_path / "b.db")
    assert len(MemoryStore(tmp_path / "b.db", Clock()).search_facts("tea", OWNER)) == 1


def test_same_follow_up_twice_is_stored_once() -> None:
    s = _store()
    a = s.add_follow_up(1, "Ask how the math exam went.", due=None)
    assert s.add_follow_up(1, "Ask how the math exam went", due=None) == a
    s.add_follow_up(1, "Ask how the trip to Da Lat was.", due=None)
    assert len(s.all_open_follow_ups(1)) == 2
    s.close_follow_up(a)
    s.add_follow_up(1, "Ask how the math exam went.", due=None)  # the old one is done: new one
    assert len(s.all_open_follow_ups(1)) == 2


def test_set_name_keeps_earlier_names_and_find_person_uses_them() -> None:
    s = _store()
    s.set_name(OWNER_ID, "Ha")
    s.set_name(OWNER_ID, "Tuan")
    s.set_name(OWNER_ID, "Tuan")  # same name again: no duplicate alias
    p = s.person(OWNER_ID)
    assert p is not None and (p.name, p.aliases) == ("Tuan", ["Ha"])
    found = s.find_person("ha")
    assert found is not None and found.id == OWNER_ID
    s.set_name(OWNER_ID, "Ha")  # back to Ha: Ha leaves the aliases, Tuan joins
    p = s.person(OWNER_ID)
    assert p is not None and (p.name, p.aliases) == ("Ha", ["Tuan"])


def test_migration_v2_moves_names_and_language_out_of_the_wrong_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "m.db"
    monkeypatch.setattr(store_mod, "MIGRATIONS", store_mod.MIGRATIONS[:1])  # a v1 database
    v1 = MemoryStore(db, Clock())
    v1._db.execute("UPDATE people SET name='Tuan', language='vi' WHERE id=1")
    v1._db.execute("INSERT INTO people(id, role, language) VALUES (2, 'guest', 'en')")
    for text in ["The owner's name is Ha.", "The owner's dog is named Samon.",
                 "Tuan's sister's name is Mai."]:  # fmt: skip
        v1.add_fact(subject_person_id=OWNER_ID, subject_text="owner", text=text)
    v1._db.execute(
        "INSERT INTO facts(subject_person_id, subject_text, text, source, created) "
        "VALUES (2, 'guest', 'The owner''s name is Lan.', 'said', 1)"
    )
    v1.close()
    monkeypatch.undo()

    s = MemoryStore(db, Clock())
    assert s.schema_version == LATEST
    assert (tmp_path / "m.db.bak-1").exists()
    owner, guest = s.person(1), s.person(2)
    assert owner is not None and (owner.name, owner.aliases) == ("Tuan", ["Ha"])
    assert guest is not None and (guest.name, guest.aliases) == ("Lan", [])  # name came from a fact
    current = [f.text for f in s.current_facts(OWNER, OWNER_ID)]
    assert current == ["The owner's dog is named Samon.", "Tuan's sister's name is Mai.",
                       "Tuan wants answers in Vietnamese."]  # fmt: skip
    cols = [r[1] for r in s._db.execute("PRAGMA table_info(people)")]
    assert "language" not in cols


def test_description_round_trip_and_v3_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "m.db"
    monkeypatch.setattr(store_mod, "MIGRATIONS", store_mod.MIGRATIONS[:2])  # a v2 database
    MemoryStore(db, Clock()).close()
    monkeypatch.undo()
    s = MemoryStore(db, Clock())
    assert s.schema_version == LATEST and (tmp_path / "m.db.bak-2").exists()
    p = s.person(1)
    assert p is not None and p.description == ""
    s.set_description(1, "  Tuan is calm.  ")
    p = s.person(1)
    assert p is not None and p.description == "Tuan is calm."


def _vec(*hot: int) -> np.ndarray:
    v = np.zeros(256, np.float32)
    v[list(hot)] = 1.0
    return v / np.linalg.norm(v)


def _tea(s: MemoryStore, text: str = "Tuan likes tea.", pid: int = OWNER_ID, **kw: object) -> int:
    return s.add_fact(subject_person_id=pid, subject_text="Tuan", text=text, **kw)  # type: ignore[arg-type]


def test_vec_roundtrip_by_model() -> None:
    s = _store()
    f = _tea(s)
    s.set_vec("fact", f, "m1", _vec(1))
    s.set_vec("fact", f, "m2", _vec(2))
    s.set_vec("fact", f, "m1", _vec(3))  # upsert
    [(fid, v)] = s.load_vecs("fact", "m1")
    assert fid == f and np.allclose(v, _vec(3))
    assert len(s.load_vecs("fact", "m2")) == 1 and s.load_vecs("episode", "m1") == []


def test_delete_facts_cascades_vectors() -> None:
    s = _store()
    f = _tea(s)
    s.set_vec("fact", f, "m", _vec(1))
    s.delete_facts([f])
    assert s.load_vecs("fact", "m") == []


def test_forget_person_cascades_vectors() -> None:
    s = _store()
    f = _tea(s)
    e = s.add_episode(session_id="a", start=0, end=1, person_ids=[1], summary="tea", topics=[])
    s.set_vec("fact", f, "m", _vec(1))
    s.set_vec("episode", e, "m", _vec(2))
    s.forget_person(OWNER_ID)
    assert s.load_vecs("fact", "m") == [] and s.load_vecs("episode", "m") == []


def test_update_fact_drops_its_vector() -> None:
    s = _store()
    f = _tea(s)
    s.set_vec("fact", f, "m", _vec(1))
    s.update_fact(f, "Tuan likes green tea.")
    assert s.load_vecs("fact", "m") == []


def test_missing_vecs_skips_embedded_and_expired() -> None:
    s = _store()
    a, b, c = _tea(s, "a fact"), _tea(s, "b fact"), _tea(s, "c fact")
    s.set_vec("fact", a, "m", _vec(1))
    s.invalidate_fact(c, valid_to=None, replaced_by=None)
    assert s.missing_vecs("fact", "m", 10) == [(b, "b fact")]
    e = s.add_episode(session_id="x", start=0, end=1, person_ids=[1], summary="sum", topics=[])
    assert s.missing_vecs("episode", "m", 10) == [(e, "sum")]
    assert s.missing_vecs("fact", "other", 1) == [(a, "a fact")]


def test_allowed_ids_guest_sees_own_and_shared_only() -> None:
    s = _store()
    s._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    own = _tea(s, "secret")
    guest = _tea(s, "guest fact", pid=2)
    shared = _tea(s, "shared fact")
    s._db.execute("UPDATE facts SET shared=1 WHERE id=?", (shared,))
    assert s.allowed_ids("fact", GUEST) == {guest, shared}
    assert s.allowed_ids("fact", OWNER) == {own, guest, shared}
    assert s.allowed_ids("fact", OWNER, person_id=2) == {guest}


def test_allowed_ids_time_window_includes_expired() -> None:
    clock = Clock(1000.0)
    s = _store(clock)
    old = _tea(s, "Tuan lives in Saigon.")
    clock.t = 5000.0
    new = _tea(s, "Tuan lives in Hanoi.")
    s.invalidate_fact(old, valid_to=None, replaced_by=new)
    assert s.allowed_ids("fact", OWNER) == {new}
    assert s.allowed_ids("fact", OWNER, start=0.0, end=2000.0) == {old}


def test_allowed_ids_sensitive_and_episode_scope() -> None:
    s = _store()
    s._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    secret = _tea(s, "secret", sensitive=True)
    plain = _tea(s, "plain")
    assert s.allowed_ids("fact", OWNER, include_sensitive=False) == {plain}
    assert secret in s.allowed_ids("fact", OWNER)
    mine = s.add_episode(session_id="a", start=0, end=1, person_ids=[1], summary="x", topics=[])
    theirs = s.add_episode(session_id="b", start=0, end=1, person_ids=[2], summary="y", topics=[])
    assert s.allowed_ids("episode", GUEST) == {theirs}
    assert s.allowed_ids("episode", OWNER) == {mine, theirs}


def test_by_ids_keep_order_and_skip_missing() -> None:
    s = _store()
    a, b = _tea(s, "a"), _tea(s, "b")
    assert [f.id for f in s.facts_by_ids([b, 999, a])] == [b, a]
    e = s.add_episode(session_id="a", start=0, end=1, person_ids=[1], summary="x", topics=[])
    assert [x.id for x in s.episodes_by_ids([e, 5])] == [e]
