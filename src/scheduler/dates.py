from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import dateparser

from .models import Slot, TimeWindow


DEFAULT_TZ = "Asia/Kolkata"
WORKDAY_START = time(9, 00)
WORKDAY_END = time(18, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(17, 0)
MORNING_START = time(9, 00)
MORNING_END = time(12, 0)
EVENING_START = time(16, 0)
EVENING_END = time(19, 0)


def aware(dt: datetime, timezone: str = DEFAULT_TZ) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo(timezone))
    return dt.astimezone(ZoneInfo(timezone))


def parse_anchor(current_date: str, timezone: str = DEFAULT_TZ) -> datetime:
    return datetime.combine(date.fromisoformat(current_date), time(9, 0), ZoneInfo(timezone))


def default_duration(edge_flags: list[str] | None = None) -> int:
    flags = set(edge_flags or [])
    if "needs_short_call" in flags:
        return 25
    return 30


def next_weekday(anchor: datetime, weekday: int, include_today: bool = False) -> date:
    days = (weekday - anchor.weekday()) % 7
    if days == 0 and not include_today:
        days = 7
    return (anchor + timedelta(days=days)).date()


def weekday_in_next_week(anchor: datetime, weekday: int) -> date:
    monday = anchor.date() + timedelta(days=(7 - anchor.weekday()))
    return monday + timedelta(days=weekday)


def _is_date_context(lowered: str, m: re.Match) -> bool:
    """Return True if a bare number match is part of a date, not a time."""
    after = lowered[m.end(1):]
    before = lowered[:m.start(1)]
    # Number followed by month name, e.g. "23 May", "27th June"
    if re.match(r"\s*(?:st|nd|rd|th)?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", after):
        return True
    # Number preceded by month name, e.g. "May 27"
    if re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+$", before):
        return True
    # Number followed by date separator, e.g. "27/05", "27-05"
    if re.match(r"[/-]\d{1,2}", after):
        return True
    # Number preceded by date separator, e.g. "05/27" (the 27 part)
    if re.search(r"\d{1,2}[/-]$", before):
        return True
    return False


def parse_time_text(text: str) -> time | None:
    lowered = text.lower()
    pattern = r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|ist)?\b"
    matches = list(re.finditer(pattern, lowered))
    if not matches:
        return None
    # Prefer matches with an explicit am/pm/ist suffix
    best = None
    for m in matches:
        if m.group(3):
            best = m
            break
    # Fall back to first match that is not part of a date expression
    if best is None:
        for m in matches:
            if not _is_date_context(lowered, m):
                best = m
                break
    if best is None:
        return None
    hour = int(best.group(1))
    minute = int(best.group(2) or "0")
    suffix = best.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    if suffix in (None, "ist") and 1 <= hour <= 7:
        hour += 12
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return time(hour, minute)
    return None


def parse_time_range(text: str) -> tuple[time, time] | None:
    lowered = text.lower()
    match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|to|and)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        lowered,
    )
    if not match:
        return None
    start_suffix = match.group(3)
    end_suffix = match.group(6)
    suffix = end_suffix or start_suffix
    start_text = f"{match.group(1)}:{match.group(2) or '00'} {start_suffix or suffix or ''}".strip()
    end_text = f"{match.group(4)}:{match.group(5) or '00'} {end_suffix or suffix or ''}".strip()
    start = parse_time_text(start_text)
    end = parse_time_text(end_text)
    if not start or not end:
        return None
    return start, end


def infer_date(text: str, current_date: str, timezone: str = DEFAULT_TZ) -> date | None:
    text = text.replace("tommorow", "tomorrow")
    lowered = text.lower()
    anchor = parse_anchor(current_date, timezone)
    weekdays = {
        "monday": 0,
        "mon": 0,
        "tuesday": 1,
        "tue": 1,
        "wednesday": 2,
        "wed": 2,
        "thursday": 3,
        "thu": 3,
        "friday": 4,
        "fri": 4,
        "saturday": 5,
        "sat": 5,
        "sunday": 6,
        "sun": 6,
    }
    if "tomorrow" in lowered:
        return (anchor + timedelta(days=1)).date()
    if "today" in lowered:
        return anchor.date()
    for name, weekday in weekdays.items():
        if re.search(rf"\b{name}\b", lowered):
            if "next week" in lowered:
                return weekday_in_next_week(anchor, weekday)
            return next_weekday(anchor, weekday, include_today=True)
    _MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    _KEYWORD_RE = r"(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)"
    has_date_cue = bool(
        re.search(
            rf"\b(\d{{1,2}}[/-]\d{{1,2}}|\d{{4}}-\d{{2}}-\d{{2}}|{_MONTH_RE}|{_KEYWORD_RE})\b",
            lowered,
        )
    )
    if not has_date_cue:
        return None


    parsed = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": anchor.replace(tzinfo=None),
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    return parsed.date() if parsed else None


def infer_day_part(text: str) -> tuple[time, time]:
    lowered = text.lower()
    time_range = parse_time_range(text)
    if time_range:
        return time_range
    after_match = re.search(r"\bafter\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lowered)
    if after_match:
        parsed = parse_time_text(after_match.group(0).replace("after", "").strip())
        return parsed or EVENING_START, WORKDAY_END
    before_match = re.search(r"\bbefore\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lowered)
    if before_match:
        parsed = parse_time_text(before_match.group(0).replace("before", "").strip())
        return WORKDAY_START, parsed or time(12, 0)
    if "morning" in lowered:
        return MORNING_START, MORNING_END
    if "afternoon" in lowered:
        return AFTERNOON_START, AFTERNOON_END
    if "evening" in lowered or "late" in lowered:
        return EVENING_START, EVENING_END
    exact = parse_time_text(text)
    if exact:
        end_dt = datetime.combine(date.today(), exact) + timedelta(minutes=30)
        return exact, end_dt.time()
    return WORKDAY_START, WORKDAY_END


def parse_window(text: str, current_date: str, timezone: str = DEFAULT_TZ) -> TimeWindow | None:
    text = text.replace("tommorow", "tomorrow")
    slot_date = infer_date(text, current_date, timezone)
    if slot_date is None:
        if "next week" in text.lower():
            anchor = parse_anchor(current_date, timezone)
            monday = anchor.date() + timedelta(days=(7 - anchor.weekday()))
            start = datetime.combine(monday, WORKDAY_START, ZoneInfo(timezone))
            end = datetime.combine(monday + timedelta(days=4), WORKDAY_END, ZoneInfo(timezone))
            return TimeWindow(start=start, end=end, label=text, timezone=timezone)
        return None
    start_t, end_t = infer_day_part(text)
    start = datetime.combine(slot_date, start_t, ZoneInfo(timezone))
    end = datetime.combine(slot_date, end_t, ZoneInfo(timezone))
    if end <= start:
        end = start + timedelta(minutes=30)
    return TimeWindow(start=start, end=end, label=text, timezone=timezone)


def parse_slot(text: str, current_date: str, duration_minutes: int = 30, timezone: str = DEFAULT_TZ) -> Slot | None:
    text = text.replace("tommorow", "tomorrow")
    if parse_time_range(text):
        return None
    slot_date = infer_date(text, current_date, timezone)
    slot_time = parse_time_text(text)
    if slot_date is None or slot_time is None:
        return None
    start = datetime.combine(slot_date, slot_time, ZoneInfo(timezone))
    end = start + timedelta(minutes=duration_minutes)
    return Slot(start=start, end=end, timezone=timezone, label=text)
