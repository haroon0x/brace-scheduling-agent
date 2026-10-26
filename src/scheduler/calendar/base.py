from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from scheduler.models import EventPayload, Slot, TimeWindow


class CalendarClient(ABC):
    @abstractmethod
    def freebusy(self, start: datetime, end: datetime, timezone: str) -> list[Slot]:
        raise NotImplementedError

    @abstractmethod
    def find_free_slots(
        self,
        windows: list[TimeWindow],
        duration_minutes: int,
        timezone: str,
        limit: int = 3,
        avoid: set[str] | None = None,
    ) -> list[Slot]:
        raise NotImplementedError

    @abstractmethod
    def create_event(
        self,
        title: str,
        attendees: list[str],
        slot: Slot,
        timezone: str,
        description: str = "",
        location: str = "",
    ) -> EventPayload:
        raise NotImplementedError
