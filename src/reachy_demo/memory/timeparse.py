"""Time words → a time window, in code so temporal questions don't depend on LLM arithmetic.

Vietnamese and English. `parse("hôm qua", now)` → (start, end) epoch seconds of yesterday in
`now`'s timezone, or None when nothing is recognized.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

Window = tuple[float, float]

WEEKDAYS_EN = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# "thứ hai".."thứ bảy" = Monday..Saturday, "chủ nhật" = Sunday; digits too ("thứ 2").
WEEKDAYS_VI = {
    "thứ hai": 0,
    "thứ 2": 0,
    "thứ ba": 1,
    "thứ 3": 1,
    "thứ tư": 2,
    "thứ 4": 2,
    "thứ năm": 3,
    "thứ 5": 3,
    "thứ sáu": 4,
    "thứ 6": 4,
    "thứ bảy": 5,
    "thứ 7": 5,
    "chủ nhật": 6,
    "cn": 6,
}
PARTS = {  # part of today → (start hour, end hour)
    "this morning": (5, 12),
    "sáng nay": (5, 12),
    "this afternoon": (12, 18),
    "chiều nay": (12, 18),
    "this evening": (18, 24),
    "tonight": (18, 24),
    "tối nay": (18, 24),
}


def _day(d: datetime, offset: int = 0) -> tuple[datetime, datetime]:
    start = (d + timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _w(a: datetime, b: datetime) -> Window:
    return a.timestamp(), b.timestamp()


def parse(text: str, now: datetime) -> Window | None:
    try:
        return _parse(" ".join(text.lower().split()), now)
    except ValueError:  # impossible date such as 31/02
        return None


def _parse(t: str, now: datetime) -> Window | None:
    if not t:
        return None

    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        d = now.replace(year=int(m[1]), month=int(m[2]), day=int(m[3]))
        return _w(*_day(d))
    if m := re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b", t):  # d/m[/y], Vietnamese order
        d = now.replace(year=int(m[3] or now.year), month=int(m[2]), day=int(m[1]))
        if d.date() > now.date() and not m[3]:
            d = d.replace(year=d.year - 1)
        return _w(*_day(d))
    if m := re.search(r"\b(\d+)\s*(day|days|ngày)\s*(ago|trước)", t):
        return _w(*_day(now, -int(m[1])))
    if m := re.search(r"\b(\d+)\s*(week|weeks|tuần)\s*(ago|trước)", t):
        start, _ = _day(now, -7 * int(m[1]) - now.weekday())
        return _w(start, start + timedelta(days=7))

    if "last night" in t or "tối qua" in t or "đêm qua" in t:
        y, today = _day(now, -1)
        return _w(y + timedelta(hours=18), today + timedelta(hours=5))
    if "day before yesterday" in t or "hôm kia" in t:
        return _w(*_day(now, -2))
    if "yesterday" in t or "hôm qua" in t:
        return _w(*_day(now, -1))

    for phrase, (h0, h1) in PARTS.items():
        if phrase in t:
            start, _ = _day(now)
            return _w(start + timedelta(hours=h0), start + timedelta(hours=h1))
    if "today" in t or "hôm nay" in t:
        return _w(*_day(now))

    last = bool(re.search(r"\blast\b|trước|rồi", t))
    if "week" in t or "tuần" in t:
        start, _ = _day(now, -now.weekday() - (7 if last else 0))
        return _w(start, start + timedelta(days=7))
    if "month" in t or "tháng" in t:
        first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if last:
            prev = first - timedelta(days=1)
            return _w(prev.replace(day=1), first)
        nxt = (first + timedelta(days=32)).replace(day=1)
        return _w(first, nxt)

    wd = next((i for i, name in enumerate(WEEKDAYS_EN) if name in t), None)
    if wd is None:
        wd = next((i for name, i in WEEKDAYS_VI.items() if re.search(rf"\b{name}\b", t)), None)
    if wd is not None:
        back = (now.weekday() - wd) % 7
        if last and back == 0:
            back = 7
        return _w(*_day(now, -back))
    return None
