from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from scheduler.calendar.base import CalendarClient
from scheduler.models import EventPayload, Slot, TimeWindow


SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarClient(CalendarClient):
    def __init__(
        self,
        calendar_id: str = "primary",
        credentials_path: str | None = None,
        token_path: str | None = None,
    ) -> None:
        self.calendar_id = calendar_id
        self.credentials_path = Path(
            credentials_path or os.getenv("GOOGLE_CALENDAR_CREDENTIALS", "credentials.json")
        )
        self.token_path = Path(token_path or os.getenv("GOOGLE_CALENDAR_TOKEN", "token.json"))
        self.service = build("calendar", "v3", credentials=self._credentials())

    def _credentials(self) -> Credentials:
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            if not self.credentials_path.exists():
                raise FileNotFoundError(
                    f"Missing Google OAuth client file: {self.credentials_path}. "
                    "Set GOOGLE_CALENDAR_CREDENTIALS or place credentials.json in repo root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        self.token_path.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def freebusy(self, start: datetime, end: datetime, timezone: str) -> list[Slot]:
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": timezone,
            "items": [{"id": self.calendar_id}],
        }
        response = self.service.freebusy().query(body=body).execute()
        busy = response["calendars"][self.calendar_id].get("busy", [])
        return [
            Slot(start=datetime.fromisoformat(item["start"]), end=datetime.fromisoformat(item["end"]), timezone=timezone)
            for item in busy
        ]

    def find_free_slots(
        self,
        windows: list[TimeWindow],
        duration_minutes: int,
        timezone: str,
        limit: int = 3,
        avoid: set[str] | None = None,
    ) -> list[Slot]:
        from datetime import time as dt_time
        from zoneinfo import ZoneInfo as ZI

        avoid = avoid or set()
        found: list[Slot] = []
        tz = ZI(timezone)
        for window in windows:
            cursor = window.start
            # Detect if window start/end is explicitly outside standard hours (9:30 AM to 6:00 PM)
            window_start_local = window.start.astimezone(tz)
            window_end_local = window.end.astimezone(tz)
            ignore_work_hours = (
                window_start_local.time() >= dt_time(18, 0)
                or window_end_local.time() <= dt_time(9, 30)
            )
            while cursor + timedelta(minutes=duration_minutes) <= window.end:
                local = cursor.astimezone(tz)
                if local.weekday() >= 5:
                    days_ahead = 7 - local.weekday()
                    cursor = datetime.combine(
                        (local + timedelta(days=days_ahead)).date(),
                        dt_time(9, 30),
                        tz,
                    )
                    continue
                if not ignore_work_hours and (local.time() < dt_time(9, 30) or local.time() > dt_time(18, 0)):
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
        event_body = {
            "summary": title,
            "description": description,
            "start": {"dateTime": slot.start.isoformat(), "timeZone": timezone},
            "end": {"dateTime": slot.end.isoformat(), "timeZone": timezone},
            "attendees": [{"email": email} for email in attendees if email],
        }
        if location:
            event_body["location"] = location
        else:
            event_body["conferenceData"] = {
                "createRequest": {
                    "requestId": str(uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        created = (
            self.service.events()
            .insert(
                calendarId=self.calendar_id,
                body=event_body,
                conferenceDataVersion=0 if location else 1,
                sendUpdates="all" if attendees else "none",
            )
            .execute()
        )
        meet_link = "" if location else (created.get("hangoutLink") or created.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri") or "")
        return EventPayload(
            event_id=created["id"],
            title=created.get("summary", title),
            attendees=attendees,
            start=slot.start,
            end=slot.end,
            timezone=timezone,
            meet_link=meet_link,
            html_link=created.get("htmlLink"),
            location=created.get("location"),
            raw=created,
        )
