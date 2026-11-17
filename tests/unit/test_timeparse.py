from datetime import datetime, timedelta, timezone

import pytest

from reachy_demo.memory.timeparse import parse

TZ = timezone(timedelta(hours=7))
NOW = datetime(2026, 11, 17, 19, 5, tzinfo=TZ)  # a Tuesday


def _d(y: int, m: int, d: int, h: int = 0) -> float:
    return datetime(y, m, d, h, tzinfo=TZ).timestamp()


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("today", _d(2026, 11, 17), _d(2026, 11, 18)),
        ("hôm nay", _d(2026, 11, 17), _d(2026, 11, 18)),
        ("what did we talk about yesterday?", _d(2026, 11, 16), _d(2026, 11, 17)),
        ("hôm qua", _d(2026, 11, 16), _d(2026, 11, 17)),
        ("hôm kia", _d(2026, 11, 15), _d(2026, 11, 16)),
        ("the day before yesterday", _d(2026, 11, 15), _d(2026, 11, 16)),
        ("this morning", _d(2026, 11, 17, 5), _d(2026, 11, 17, 12)),
        ("sáng nay", _d(2026, 11, 17, 5), _d(2026, 11, 17, 12)),
        ("tối nay", _d(2026, 11, 17, 18), _d(2026, 11, 18)),
        ("last night", _d(2026, 11, 16, 18), _d(2026, 11, 17, 5)),
        ("tối qua", _d(2026, 11, 16, 18), _d(2026, 11, 17, 5)),
        ("3 days ago", _d(2026, 11, 14), _d(2026, 11, 15)),
        ("3 ngày trước", _d(2026, 11, 14), _d(2026, 11, 15)),
        ("this week", _d(2026, 11, 16), _d(2026, 11, 23)),
        ("last week", _d(2026, 11, 9), _d(2026, 11, 16)),
        ("tuần trước", _d(2026, 11, 9), _d(2026, 11, 16)),
        ("last month", _d(2026, 10, 1), _d(2026, 11, 1)),
        ("tháng này", _d(2026, 11, 1), _d(2026, 12, 1)),
        ("on Sunday", _d(2026, 11, 15), _d(2026, 11, 16)),
        ("last Tuesday", _d(2026, 11, 10), _d(2026, 11, 11)),
        ("Tuesday", _d(2026, 11, 17), _d(2026, 11, 18)),
        ("thứ hai", _d(2026, 11, 16), _d(2026, 11, 17)),
        ("thứ 6", _d(2026, 11, 13), _d(2026, 11, 14)),
        ("chủ nhật", _d(2026, 11, 15), _d(2026, 11, 16)),
        ("2026-11-02", _d(2026, 11, 2), _d(2026, 11, 3)),
        ("ngày 20/10", _d(2026, 10, 20), _d(2026, 10, 21)),
        ("25/12", _d(2025, 12, 25), _d(2025, 12, 26)),  # no year, in the future → last year
    ],
)
def test_parse(text: str, start: float, end: float) -> None:
    assert parse(text, NOW) == (start, end)


@pytest.mark.parametrize("text", ["", "my dog", "something about Mai", "31/02", "2026-13-01"])
def test_unrecognized_is_none(text: str) -> None:
    assert parse(text, NOW) is None
