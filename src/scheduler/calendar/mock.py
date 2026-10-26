from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from scheduler.calendar.base import CalendarClient
from scheduler.models import EventPayload, Slot, TimeWindow


class MockCalendarClient(CalendarClient):
    def __init__(self) -> None:
        self.events: list[EventPayload] = []
        self.busy: list[Slot] = [
            Slot(
                start=datetime(2026, 5, 26, 15, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 26, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                label="Existing investor call",
            ),
        ]

    def freebusy(self, start: datetime, end: datetime, timezone: str) -> list[Slot]:
        return [
            slot
            for slot in self.busy
            if slot.start < end and slot.end > start
        ]

    def find_free_slots(
        self,
        windows: list[TimeWindow],
        duration_minutes: int,
        timezone: str,
        limit: int = 3,
        avoid: set[str] | None = None,
    ) -> list[Slot]:
        avoid = avoid or set()
        found: list[Slot] = []
        tz = ZoneInfo(timezone)
        for window in windows:
            cursor = window.start.astimezone(tz)
            # Detect if window start/end is explicitly outside standard hours (9:30 AM to 6:00 PM)
            window_start_local = window.start.astimezone(tz)
            window_end_local = window.end.astimezone(tz)
            ignore_work_hours = (
                window_start_local.time() >= time(18, 0)
                or window_end_local.time() <= time(9, 30)
            )
            while cursor + timedelta(minutes=duration_minutes) <= window.end.astimezone(tz):
                if cursor.weekday() >= 5:
                    # Skip to next Monday 9:30
                    days_ahead = 7 - cursor.weekday()
                    cursor = datetime.combine(
                        (cursor + timedelta(days=days_ahead)).date(),
                        time(9, 30),
                        tz,
                    )
                    continue
                if not ignore_work_hours and (cursor.time() < time(9, 30) or cursor.time() > time(18, 0)):
                    cursor += timedelta(minutes=30)
                    continue
                candidate = Slot(
                    start=cursor,
                    end=cursor + timedelta(minutes=duration_minutes),
                    timezone=timezone,
                    label=window.label,
                )
                if candidate.key() not in avoid and not self.freebusy(candidate.start, candidate.end, timezone):
                    found.append(candidate)
                    if len(found) >= limit:
                        return found
                cursor += timedelta(minutes=30)
        return found

    def create_event(
        self,
        title: str,
        attendees: list[str],
        slot: Slot,
        timezone: str,
        description: str = "",
        location: str = "",
    ) -> EventPayload:
        meet = "" if location else "https://meet.google.com/mock-abc-defg-hij"
        event = EventPayload(
            event_id=f"mock-{len(self.events) + 1}",
            title=title,
            attendees=attendees,
            start=slot.start,
            end=slot.end,
            timezone=timezone,
            meet_link=meet,
            html_link="https://calendar.google.com/calendar/event?eid=mock",
            location=location or None,
            raw={"description": description},
        )
        self.events.append(event)
        self.busy.append(slot)
        return event
