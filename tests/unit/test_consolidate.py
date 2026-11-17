import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from reachy_demo.memory.consolidate import Consolidator, worth_keeping
from reachy_demo.memory.prompts import ask_json
from reachy_demo.memory.store import OWNER, OWNER_ID, MemoryStore, Speaker

NOW = 1_763_380_000.0  # 2025-11-17


class Clock:
    def __call__(self) -> float:
        return NOW


class FakeChat:
    """Replies in order; a dict is sent as JSON, a str as is, an Exception is raised."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    async def __call__(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        yield json.dumps(reply) if isinstance(reply, dict) else reply


def _fact(text: str, **kw: Any) -> dict[str, Any]:
    return {
        "subject": "Tuan",
        "text": text,
        "text_original": "",
        "category": "other",
        "importance": 0.5,
        "sensitive": False,
        **kw,
    }


def _extract(*facts: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {"facts": list(facts), "follow_ups": [], "name": None, **kw}


def _setup(*replies: Any) -> tuple[MemoryStore, Consolidator, FakeChat]:
    store = MemoryStore(":memory:", Clock())
    store.set_name(OWNER_ID, "Tuan")
    chat = FakeChat(*replies)
    return store, Consolidator(store, chat, Clock()), chat


def _texts(store: MemoryStore, q: str) -> list[str]:
    return [f.text for f, _ in store.search_facts(q, OWNER)]


async def test_add_when_nothing_similar_uses_one_call() -> None:
    store, c, chat = _setup(_extract(_fact("Tuan has a dog named Bin.")))
    changes = await c.consolidate("tôi có con chó tên Bin", OWNER)
    assert [ch.action for ch in changes] == ["ADD"]
    assert _texts(store, "dog") == ["Tuan has a dog named Bin."]
    assert len(chat.calls) == 1


async def test_noop_when_already_known() -> None:
    store, c, _ = _setup(_extract(_fact("Tuan's dog is called Bin.")), {"action": "NOOP"})
    store.add_fact(
        subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan has a dog named Bin."
    )
    assert [ch.action for ch in await c.consolidate("my dog is Bin", OWNER)] == ["NOOP"]
    assert len(_texts(store, "dog")) == 1


async def test_update_merges_text() -> None:
    store, c, _ = _setup(
        _extract(_fact("Tuan's dog Bin is a corgi.")),
        {"action": "UPDATE", "id": 1, "text": "Tuan has a corgi named Bin."},
    )
    store.add_fact(
        subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan has a dog named Bin."
    )
    await c.consolidate("Bin is a corgi", OWNER)
    assert _texts(store, "Bin") == ["Tuan has a corgi named Bin."]


async def test_invalidate_expires_old_and_links_new() -> None:
    store, c, _ = _setup(_extract(_fact("Tuan lives in Hanoi.")), {"action": "INVALIDATE", "id": 1})
    store.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan lives in Saigon.")
    changes = await c.consolidate("I moved to Hanoi", OWNER)
    assert _texts(store, "lives") == ["Tuan lives in Hanoi."]
    old = store.fact(1)
    assert old is not None and old.expired == NOW
    assert (
        store._db.execute("SELECT replaced_by FROM facts WHERE id=1").fetchone()[0]
        == changes[0].fact_id
    )


async def test_decide_with_unknown_id_falls_back_to_add() -> None:
    store, c, _ = _setup(
        _extract(_fact("Tuan lives in Hanoi.")), {"action": "INVALIDATE", "id": 99}
    )
    store.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan lives in Saigon.")
    assert [ch.action for ch in await c.consolidate("I moved to Hanoi", OWNER)] == ["ADD"]
    assert len(_texts(store, "lives")) == 2


async def test_bad_json_retried_once_then_dropped() -> None:
    store, c, chat = _setup("sure! here you go", "still not json")
    assert await c.consolidate("I like tea", OWNER) == []
    assert len(chat.calls) == 2
    assert "not valid JSON" in chat.calls[1][-1]["content"]


async def test_fenced_json_is_accepted() -> None:
    fenced = "```json\n" + json.dumps(_extract(_fact("Tuan likes tea."))) + "\n```"
    store, c, _ = _setup(fenced)
    await c.consolidate("I like tea", OWNER)
    assert _texts(store, "tea") == ["Tuan likes tea."]


async def test_llm_error_is_swallowed() -> None:
    _, c, _ = _setup(RuntimeError("network"))
    assert await c.consolidate("I like tea", OWNER) == []


async def test_guest_facts_are_stored_under_the_guest_only() -> None:
    store, c, chat = _setup(
        _extract(
            _fact("Lan likes tea.", category="preference"),
            name="Lan",
            follow_ups=[{"text": "Ask Lan about tea", "due": None}],
        )
    )
    store._db.execute("INSERT INTO people(id, role) VALUES (2, 'guest')")
    guest = Speaker(2, "guest")
    changes = await c.consolidate("I'm Lan, I like tea", guest)
    assert [ch.action for ch in changes] == ["ADD"]
    assert [f.text for f in store.current_facts(guest, 2)] == ["Lan likes tea."]
    assert store.current_facts(OWNER, OWNER_ID) == []
    # just facts: no name, no follow-up, no description, and only the extract call
    g = store.person(2)
    assert g is not None and (g.name, g.description) == (None, "")
    assert store.all_open_follow_ups(2) == [] and len(chat.calls) == 1


async def test_owner_self_fact_rewrites_description() -> None:
    store, c, chat = _setup(
        _extract(_fact("Tuan is calm and curious.", category="bio")),
        {"action": "ADD"},  # decide: the engineer fact matches on the subject "Tuan"
        {"description": "Tuan is a calm, curious engineer."},
    )
    store.add_fact(
        subject_person_id=OWNER_ID,
        subject_text="Tuan",
        text="Tuan works as an engineer.",
        category="bio",
        source="said",
    )
    store.add_fact(
        subject_person_id=OWNER_ID,
        subject_text="Samon",
        text="Samon is a Corgi.",
        category="relation",
        source="said",
    )
    await c.consolidate("I'm calm and curious", OWNER)
    p = store.person(OWNER_ID)
    assert p is not None and p.description == "Tuan is a calm, curious engineer."
    sent = chat.calls[-1][1]["content"]
    assert "engineer" in sent and "curious" in sent and "Corgi" not in sent


async def test_plan_fact_leaves_description_alone() -> None:
    store, c, chat = _setup(_extract(_fact("Tuan has an exam tomorrow.", category="plan")))
    await c.consolidate("I have an exam tomorrow", OWNER)
    p = store.person(OWNER_ID)
    assert p is not None and p.description == "" and len(chat.calls) == 1


@pytest.mark.parametrize(("explicit", "stored"), [(False, 0), (True, 1)])
async def test_sensitive_only_when_asked_to_remember(explicit: bool, stored: int) -> None:
    store, c, _ = _setup(_extract(_fact("Tuan's bank PIN is 1234.", sensitive=True)))
    await c.consolidate("my PIN is 1234", OWNER, explicit=explicit)
    assert len(store.search_facts("PIN", OWNER)) == stored


async def test_name_on_person_language_as_setting_fact() -> None:
    store, c, _ = _setup(
        _extract(_fact("Tuấn wants answers in Vietnamese.", category="setting"), name="Tuấn")
    )
    await c.consolidate("tên tôi là Tuấn, nói tiếng Việt nhé", OWNER)
    p = store.person(OWNER_ID)
    assert p is not None and (p.name, p.aliases) == ("Tuấn", ["Tuan"])  # earlier name kept
    assert [f.category for f, _ in store.search_facts("Vietnamese", OWNER)] == ["setting"]


async def test_follow_up_added_and_later_closed_by_matching_fact() -> None:
    store, c, _ = _setup(
        _extract(
            _fact("Tuan's exam is on 2025-11-21.", category="plan"),
            follow_ups=[{"text": "Ask how the exam went.", "due": "2025-11-21"}],
        ),
        _extract(_fact("Tuan said the exam went well.")),
        {"action": "ADD"},
    )
    await c.consolidate("thứ sáu này tôi thi", OWNER)
    assert [f.text for f in store.all_open_follow_ups(OWNER_ID)] == ["Ask how the exam went."]
    await c.consolidate("thi ổn rồi", OWNER)
    assert store.all_open_follow_ups(OWNER_ID) == []


async def test_marks_event_consolidated_and_keeps_evidence() -> None:
    store, c, _ = _setup(_extract(_fact("Tuan likes tea.")))
    eid = store.log_event("turn", {"user": "I like tea"}, person_id=OWNER_ID, session_id="s")
    await c.consolidate("I like tea", OWNER, event_id=eid)
    assert store.unconsolidated_turns(since=0, limit=10) == []
    assert store._db.execute("SELECT evidence_json FROM facts").fetchone()[0] == f"[{eid}]"


async def test_ask_json_timeout_returns_none() -> None:
    async def slow(_m: list[dict[str, str]]) -> AsyncIterator[str]:
        import asyncio

        await asyncio.sleep(1)
        yield "{}"

    assert await ask_json(slow, "s", "u", timeout_s=0.01) is None


@pytest.mark.parametrize(
    ("text", "keep"),
    [
        ("hi", False),
        ("cảm ơn nhé", False),
        ("turn left", False),
        ("look up again please", False),
        ("I have a sister named Mai", True),
        ("remember this", True),
        ("thứ sáu này tôi thi", True),
    ],
)
def test_worth_keeping(text: str, keep: bool) -> None:
    assert worth_keeping(text) is keep


async def test_own_name_goes_to_person_not_facts() -> None:
    store, c, _ = _setup(_extract(_fact("The owner's name is Ha."), name="Ha"))
    await c.consolidate("my name is Ha", OWNER)
    p = store.person(OWNER_ID)
    assert p is not None and p.name == "Ha"
    assert store.search_facts("name", OWNER) == []


async def test_name_change_expires_old_name_facts() -> None:
    store, c, _ = _setup(_extract(name="Tuan"))
    store.add_fact(subject_person_id=OWNER_ID, subject_text="owner", text="The owner's name is Ha.")
    sister = store.add_fact(
        subject_person_id=OWNER_ID, subject_text="Tuan", text="Tuan's sister's name is Mai."
    )
    await c.consolidate("sorry, just call me Tuan", OWNER)
    assert [f.id for f, _ in store.search_facts("name", OWNER)] == [sister]


async def test_same_name_again_also_closes_the_old_name_fact() -> None:
    store, c, _ = _setup(_extract(name="Ha"))
    store.add_fact(subject_person_id=OWNER_ID, subject_text="owner", text="The owner's name is Ha.")
    await c.consolidate("my name is Ha", OWNER)
    assert store.search_facts("name", OWNER) == []
    p = store.person(OWNER_ID)
    assert p is not None and p.name == "Ha"


@pytest.mark.parametrize(
    ("text", "own"),
    [
        ("The owner's name is Ha.", True),
        ("Tuan goes by Tom.", True),
        ("Ha wants to be called Tuan.", True),
        ("Tuan's sister's name is Mai.", False),
        ("The owner's dog is named Samon.", False),
    ],
)
def test_own_name_fact(text: str, own: bool) -> None:
    from reachy_demo.memory.store import own_name_fact

    assert own_name_fact(text, "Ha", "Tuan") is own


async def test_llm_closes_follow_ups_it_says_were_answered() -> None:
    store, c, chat = _setup()
    exam = store.add_follow_up(OWNER_ID, "Ask how the exam went.", None)
    trip = store.add_follow_up(OWNER_ID, "Ask how the trip to Da Lat was.", None)
    # the id as a string counts; 999 was never offered and "x" isn't an id
    chat.replies.append(_extract(answered=[str(exam), 999, "x"]))
    await c.consolidate("thi ổn rồi, được 9 điểm", OWNER)
    assert [f.id for f in store.all_open_follow_ups(OWNER_ID)] == [trip]
    sent = chat.calls[0][1]["content"]
    assert f"{exam}: Ask how the exam went." in sent and f"{trip}: Ask how the trip" in sent


async def test_bad_answered_field_closes_nothing() -> None:
    store, c, chat = _setup(_extract(answered="all"))
    store.add_follow_up(OWNER_ID, "Ask how the exam went.", None)
    await c.consolidate("I went to the market today", OWNER)
    assert len(store.all_open_follow_ups(OWNER_ID)) == 1
