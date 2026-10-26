from scheduler.dates import parse_slot, parse_time_text, parse_window


def test_tomorrow_from_task_anchor():
    slot = parse_slot("tomorrow 5pm", "2026-05-23", timezone="Asia/Kolkata")
    assert slot is not None
    assert slot.start.isoformat() == "2026-05-24T17:00:00+05:30"


def test_next_week_tuesday_afternoon_window():
    window = parse_window("next week Tuesday afternoon", "2026-05-23", "Asia/Kolkata")
    assert window is not None
    assert window.start.isoformat() == "2026-05-26T13:00:00+05:30"
    assert window.end.isoformat() == "2026-05-26T17:00:00+05:30"


def test_after_4_window():
    window = parse_window("Tuesday after 4", "2026-05-23", "Asia/Kolkata")
    assert window is not None
    assert window.start.isoformat() == "2026-05-26T16:00:00+05:30"
    assert window.end.isoformat() == "2026-05-26T18:30:00+05:30"


def test_bare_time_is_not_exact_slot():
    assert parse_slot("3:30", "2026-05-23", timezone="Asia/Kolkata") is None


def test_time_range_is_window_not_exact_slot():
    window = parse_window("Wednesday the 27th between 11:00 and 13:00", "2026-05-23", "Asia/Kolkata")
    assert window is not None
    assert window.start.isoformat() == "2026-05-27T11:00:00+05:30"
    assert window.end.isoformat() == "2026-05-27T13:00:00+05:30"
    assert parse_slot("Wednesday 27th 11:00-13:00", "2026-05-23", timezone="Asia/Kolkata") is None


# --- New tests for bug fixes ---


def test_parse_time_text_prefers_pm_suffix_over_date_number():
    """'Sat 23 May, 5:00 PM' must extract 17:00, not 23:00."""
    from datetime import time

    result = parse_time_text("Sat 23 May, 5:00 PM Asia/Kolkata")
    assert result == time(17, 0)


def test_parse_time_text_skips_month_context():
    """'May 27th 3pm' must extract 15:00, not 27."""
    from datetime import time

    result = parse_time_text("May 27th 3pm")
    assert result == time(15, 0)


def test_parse_time_text_skips_numeric_date():
    """'27/05 at 3:30' must extract 15:30, not 27 or 05."""
    from datetime import time

    result = parse_time_text("27/05 at 3:30")
    assert result == time(15, 30)


def test_month_date_exact_slot():
    """'May 27th 3pm' should parse as an exact slot on May 27."""
    slot = parse_slot("May 27th 3pm", "2026-05-23", timezone="Asia/Kolkata")
    assert slot is not None
    assert slot.start.isoformat() == "2026-05-27T15:00:00+05:30"


def test_saturday_on_saturday_resolves_to_today():
    """'saturday 5pm' on a Saturday (2026-05-23) should resolve to today."""
    slot = parse_slot("saturday 5pm", "2026-05-23", timezone="Asia/Kolkata")
    assert slot is not None
    assert slot.start.date().isoformat() == "2026-05-23"
    assert slot.start.hour == 17


def test_find_free_slots_skips_weekends():
    """Mock calendar must not propose Saturday/Sunday slots."""
    from scheduler.calendar.mock import MockCalendarClient
    from scheduler.models import TimeWindow
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    # Window spanning Friday evening through Monday morning
    window = TimeWindow(
        start=datetime(2026, 5, 29, 17, 0, tzinfo=tz),
        end=datetime(2026, 6, 1, 12, 0, tzinfo=tz),
        label="test",
        timezone="Asia/Kolkata",
    )
    slots = MockCalendarClient().find_free_slots([window], 30, "Asia/Kolkata", limit=5)
    assert slots
    for slot in slots:
        assert slot.start.weekday() < 5, f"Got weekend slot: {slot.start}"


def test_find_free_slots_skips_late_hours():
    """Mock calendar must not propose slots after 18:00."""
    from scheduler.calendar.mock import MockCalendarClient
    from scheduler.models import TimeWindow
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    window = TimeWindow(
        start=datetime(2026, 5, 25, 16, 0, tzinfo=tz),
        end=datetime(2026, 5, 25, 23, 0, tzinfo=tz),
        label="test",
        timezone="Asia/Kolkata",
    )
    slots = MockCalendarClient().find_free_slots([window], 30, "Asia/Kolkata", limit=10)
    assert slots
    for slot in slots:
        assert slot.start.time() <= time(18, 0), f"Got late slot: {slot.start}"


def test_full_month_name_parsing():
    """'9th june 10 am' should successfully parse as June 9th at 10 AM."""
    slot = parse_slot("9th june 10 am", "2026-05-23", timezone="Asia/Kolkata")
    assert slot is not None
    assert slot.start.month == 6
    assert slot.start.day == 9
    assert slot.start.hour == 10


