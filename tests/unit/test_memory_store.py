import sqlite3
from pathlib import Path

import pytest

from reachy_demo.memory import store as store_mod
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore, Speaker, match_query

GUEST = Speaker(2, "guest")


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
    assert s.schema_version == 2


def test_reopen_is_a_noop(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    s = MemoryStore(db, Clock())
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan likes tea.")
    s.close()
    s2 = MemoryStore(db, Clock())
    assert s2.schema_version == 2
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
    assert again.schema_version == 2
    assert (tmp_path / "m.db.bak-2").exists()  # backup taken before the failed migration


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
    assert s.schema_version == 2
    assert (tmp_path / "m.db.bak-1").exists()
    owner, guest = s.person(1), s.person(2)
    assert owner is not None and (owner.name, owner.aliases) == ("Tuan", ["Ha"])
    assert guest is not None and (guest.name, guest.aliases) == ("Lan", [])  # name came from a fact
    current = [f.text for f in s.current_facts(OWNER, OWNER_ID)]
    assert current == ["The owner's dog is named Samon.", "Tuan's sister's name is Mai.",
                       "Tuan wants answers in Vietnamese."]  # fmt: skip
    cols = [r[1] for r in s._db.execute("PRAGMA table_info(people)")]
    assert "language" not in cols
