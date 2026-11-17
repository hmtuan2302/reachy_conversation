from datetime import UTC

from reachy_demo.memory.briefing import DEFAULT_MEMORY_TOKENS, RULE, ago, build_briefing
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore
from reachy_demo.ops.tokens import est_tokens

DAY = 86400.0
NOW = 100 * DAY


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _store(t: float = NOW) -> tuple[MemoryStore, Clock]:
    c = Clock(t)
    return MemoryStore(":memory:", c), c


def _brief(s: MemoryStore) -> str:
    return build_briefing(s, OWNER, NOW, UTC)


def test_first_conversation_without_name() -> None:
    s, _ = _store()
    text = _brief(s)
    assert "whose name you don't know yet" in text
    assert "this is your first conversation" in text
    assert text.endswith(RULE)


def test_name_and_last_talked() -> None:
    s, c = _store(NOW - 2 * DAY)
    s.set_name(OWNER_ID, "Tuan")
    s.log_event("turn", {"user": "hi"}, person_id=OWNER_ID, session_id="a")
    c.t = NOW
    text = _brief(s)
    assert "You are talking to Tuan (owner); last talked 2 days ago." in text


def test_earlier_name_shown() -> None:
    s, _ = _store()
    s.set_name(OWNER_ID, "Ha")
    s.set_name(OWNER_ID, "Tuan")
    assert "You are talking to Tuan (owner; earlier name: Ha)" in _brief(s)


def test_five_newest_conversations_and_due_follow_up() -> None:
    s, _ = _store()
    for i in range(7):
        s.add_episode(
            session_id=f"s{i}",
            start=NOW - (7 - i) * DAY,
            end=NOW - (7 - i) * DAY + 60,
            person_ids=[OWNER_ID],
            summary=f"talk {i}",
            topics=[],
        )
    s.add_follow_up(OWNER_ID, "Ask how the exam went.", due=NOW + DAY)
    text = _brief(s)
    assert [f"talk {i}" in text for i in range(7)] == [False, False, True, True, True, True, True]
    assert text.index("talk 6") < text.index("talk 2")  # newest first
    assert "- Ask how the exam went. (due" in text


def test_sensitive_and_expired_facts_are_never_shown() -> None:
    s, _ = _store()
    s.add_fact(
        subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan's PIN is 1234.", sensitive=True
    )
    old = s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan lives in Saigon.")
    s.invalidate_fact(old, valid_to=None, replaced_by=None)
    s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan has a dog named Bin.")
    text = _brief(s)
    assert "PIN" not in text and "Saigon" not in text
    assert "- Tuan has a dog named Bin. (said, just now)" in text


def test_preferences_get_their_own_block() -> None:
    s, _ = _store()
    s.add_fact(
        subject_person_id=OWNER_ID,
        subject_text="Tuan",
        text="Tuan wants short answers.",
        category="preference",
    )
    text = _brief(s)
    assert text.index("Preferences:") < text.index("Tuan wants short answers.")


def _crowded() -> MemoryStore:
    s, _ = _store()
    for i in range(5):
        s.add_episode(
            session_id=f"s{i}",
            start=NOW - DAY + i,
            end=NOW - DAY + i,
            person_ids=[OWNER_ID],
            summary=f"episode {i} " + "x" * 300,
            topics=[],
        )
    for i in range(10):
        s.add_fact(
            subject_person_id=OWNER_ID,
            subject_text="Tuan",
            text=f"fact {i} " + "y" * 150,
            importance=1 - i / 20,
        )
    return s


def test_cap_trims_facts_first_keeps_five_and_all_episodes() -> None:
    text = _brief(_crowded())
    assert est_tokens(text) <= DEFAULT_MEMORY_TOKENS or text.count("fact ") == 5
    assert 5 <= text.count("fact ") < 10
    assert all(f"episode {i}" in text for i in range(5))
    assert "fact 0" in text  # highest score kept


def test_ago() -> None:
    assert ago(30) == "just now"
    assert ago(60) == "1 minute ago"
    assert ago(3 * 3600) == "3 hours ago"
    assert ago(1.5 * DAY) == "yesterday"
    assert ago(3 * DAY) == "3 days ago"
    assert ago(21 * DAY) == "3 weeks ago"
    assert ago(90 * DAY) == "3 months ago"


def test_briefing_respects_cap_tokens() -> None:
    s = _crowded()
    small = build_briefing(s, OWNER, NOW, UTC, cap_tokens=300)
    assert est_tokens(small) <= 300 or small.count("fact ") == 5
    big = build_briefing(s, OWNER, NOW, UTC, cap_tokens=10_000)
    assert small.count("fact ") < big.count("fact ")


def test_briefing_parts_ids_match_text() -> None:
    from reachy_demo.memory.briefing import build_briefing_parts

    s, _ = _store()
    fid = s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan has a dog Bin.")
    eid = s.add_episode(session_id="a", start=NOW - DAY, end=NOW - DAY, person_ids=[OWNER_ID],
                        summary="Talked about Bin.", topics=[])  # fmt: skip
    b = build_briefing_parts(s, OWNER, NOW, UTC)
    assert b.fact_ids == {fid} and b.episode_ids == {eid}
    assert b.text == _brief(s)
