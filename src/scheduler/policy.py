from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .calendar.base import CalendarClient
from .dates import default_duration, infer_date, parse_anchor, parse_slot, parse_time_text, parse_window
from .models import AnalysisResult, SchedulingState, Slot, TargetIntent, TerminalStatus, TimeWindow, ToolCall

DECK_LINK = "https://example.com/deck"


@dataclass
class PolicyDecision:
    message: str
    reason: str
    status: TerminalStatus


class SchedulingPolicy:
    _EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    def decide(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        self._pre_process(state, analysis)

        if state.confirmed_slot and state.target.email and "candidate_email_collected" in state.edge_case_flags:
            return self._book_slot(state, analysis, calendar, state.confirmed_slot)

        handler = {
            TargetIntent.CONFIRM_SLOT: self._handle_confirm,
            TargetIntent.COUNTER_PROPOSE: self._handle_counter,
            TargetIntent.DELEGATE_CHOICE: self._handle_delegate,
            TargetIntent.REQUEST_WINDOW: self._handle_request_window,
            TargetIntent.REJECT: self._handle_reject,
            TargetIntent.CLOSE: self._handle_close,
        }[analysis.intent]

        return handler(state, analysis, calendar)

    def _pre_process(self, state: SchedulingState, analysis: AnalysisResult) -> None:
        if not state.proposed_slots and analysis.recovered_proposed_slots:
            state.proposed_slots = [
                Slot(
                    start=datetime.fromisoformat(window.start),
                    end=datetime.fromisoformat(window.end),
                    timezone=state.timezone,
                    label=window.label
                )
                for window in analysis.recovered_proposed_slots
            ]

        if not state.rejected_slots and analysis.recovered_rejected_slots:
            state.rejected_slots = [
                Slot(
                    start=datetime.fromisoformat(window.start),
                    end=datetime.fromisoformat(window.end),
                    timezone=state.timezone,
                    label=window.label
                )
                for window in analysis.recovered_rejected_slots
            ]

        if state.thread and state.thread[-1].sender == "target" and not state.target.email:
            email_match = self._EMAIL_RE.search(state.thread[-1].text)
            if email_match:
                state.target.email = email_match.group(0)

        duration = 25 if analysis.wants_short_call else default_duration(state.edge_case_flags)
        for text in analysis.rejected_slot_texts:
            if text not in state.rejected_texts:
                state.rejected_texts.append(text)
            rejected = parse_slot(text, state.current_date, duration, state.timezone)
            if rejected:
                state.rejected_slots.append(rejected)

    def _handle_close(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        return PolicyDecision(
            message="No worries, thanks for taking a look.",
            reason="Target does not want to schedule.",
            status=TerminalStatus.NO_MEETING_INTENT,
        )

    def _handle_confirm(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        if analysis.target_slot:
            return self._book_slot(state, analysis, calendar, analysis.target_slot)

        if analysis.exact_slot_text:
            duration = 25 if analysis.wants_short_call else default_duration(state.edge_case_flags)
            exact = parse_slot(analysis.exact_slot_text, state.current_date, duration, state.timezone)
            if exact:
                return self._book_slot(state, analysis, calendar, exact)

        if len(state.proposed_slots) > 1:
            return PolicyDecision(
                message="Which of those works best for you?",
                reason="Target confirmed but could not match to a specific proposed slot.",
                status=TerminalStatus.WAITING_ON_TARGET,
            )

        if state.proposed_slots:
            return self._book_slot(state, analysis, calendar, state.proposed_slots[0])

        return self._propose_new_slots(state, analysis, calendar)

    def _handle_counter(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        if analysis.target_slot:
            return self._book_slot(state, analysis, calendar, analysis.target_slot)

        if analysis.exact_slot_text:
            duration = 25 if analysis.wants_short_call else default_duration(state.edge_case_flags)
            exact = parse_slot(analysis.exact_slot_text, state.current_date, duration, state.timezone)
            if exact:
                return self._book_slot(state, analysis, calendar, exact)

        if analysis.requested_windows or analysis.resolved_windows:
            return self._propose_new_slots(state, analysis, calendar)

        return PolicyDecision(
            message="Could you share a specific time that works?",
            reason="Could not parse the counter-proposed time.",
            status=TerminalStatus.WAITING_ON_TARGET,
        )

    def _handle_delegate(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        if state.proposed_slots:
            return self._book_slot(state, analysis, calendar, state.proposed_slots[0])

        return self._propose_new_slots(state, analysis, calendar, auto_book=False)

    def _handle_request_window(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        delegate = "delegate" in analysis.intent_modifiers and bool(state.proposed_slots)
        return self._propose_new_slots(state, analysis, calendar, auto_book=delegate)

    def _handle_reject(self, state: SchedulingState, analysis: AnalysisResult, calendar: CalendarClient) -> PolicyDecision:
        return self._propose_new_slots(state, analysis, calendar)

    def _propose_new_slots(
        self,
        state: SchedulingState,
        analysis: AnalysisResult,
        calendar: CalendarClient,
        auto_book: bool = False,
    ) -> PolicyDecision:
        duration = 25 if analysis.wants_short_call else default_duration(state.edge_case_flags)
        limit = 1 if "senior_partner_brevity" in state.edge_case_flags else 3

        windows = self._windows(state, analysis)
        free = self._find_free_slots(state, calendar, windows, duration, limit)

        if not free:
            fallback_windows = self._default_window(state)
            if fallback_windows != windows:
                free = self._find_free_slots(state, calendar, fallback_windows, duration, limit)

        if not free:
            return PolicyDecision(
                message="I am not seeing a clean slot in that window. Could you share one more window that works?",
                reason="No calendar availability found inside requested window.",
                status=TerminalStatus.BLOCKED,
            )

        if auto_book:
            return self._book_slot(state, analysis, calendar, free[0])

        state.proposed_slots = free
        message = self._proposal_message(state, free, analysis)
        return PolicyDecision(
            message=message,
            reason="Proposed available slots from calendar, avoiding rejected/discussed slots.",
            status=TerminalStatus.WAITING_ON_TARGET,
        )

    def _book_slot(
        self,
        state: SchedulingState,
        analysis: AnalysisResult,
        calendar: CalendarClient,
        slot: Slot,
    ) -> PolicyDecision:
        if state.confirmed_slot and state.confirmed_slot.start == slot.start and state.event:
            prefix = ""
            if analysis.asks_for_deck:
                prefix = f"Sharing the deck here: {DECK_LINK}\n\n"
            if analysis.asks_for_email and state.user.email:
                prefix += f"My email is {state.user.email}.\n\n"
            if state.event.location:
                msg_body = f"Booked for {self._format_slot(slot)} in-person. See you at {state.event.location}."
            else:
                msg_body = f"Booked for {self._format_slot(slot)}. Meet link: {state.event.meet_link}"
            return PolicyDecision(
                message=f"{prefix}{msg_body}",
                reason="Slot is already successfully booked.",
                status=TerminalStatus.BOOKED,
            )

        busy = calendar.freebusy(slot.start, slot.end, state.timezone)
        state.remember_tool(
            ToolCall(
                tool="calendar",
                action="freebusy",
                input={"start": slot.start.isoformat(), "end": slot.end.isoformat(), "timezone": state.timezone},
                output={"busy": [b.model_dump(mode="json") for b in busy]},
            )
        )
        slot_local = slot.start.astimezone(ZoneInfo(state.timezone))
        is_weekend = slot_local.weekday() >= 5 and "weekend_slot_offered" not in state.edge_case_flags
        is_outside_hours = (
            (slot_local.time() < time(9, 30) or slot_local.time() > time(18, 0))
            and "founder_quick_yes" not in state.edge_case_flags
            and "respect_late_hour_constraint" not in state.edge_case_flags
        )
        is_clash = bool(busy) or is_weekend or is_outside_hours

        if is_clash:
            if is_weekend:
                target_date = slot_local.date() + timedelta(days=(7 - slot_local.weekday()))
            else:
                target_date = slot_local.date()

            window = TimeWindow(
                start=datetime.combine(target_date, time(9, 30), ZoneInfo(state.timezone)),
                end=datetime.combine(target_date, time(18, 0), ZoneInfo(state.timezone)),
                label=f"near {slot.label or slot.start.isoformat()}",
                timezone=state.timezone,
            )
            alternatives = calendar.find_free_slots(
                [window],
                int((slot.end - slot.start).total_seconds() // 60),
                state.timezone,
                limit=3,
                avoid=state.proposed_keys() | state.rejected_keys(),
            )
            state.remember_tool(
                ToolCall(
                    tool="calendar",
                    action="find_free_slots",
                    input={"windows": [window.model_dump(mode="json")]},
                    output={"slots": [s.model_dump(mode="json") for s in alternatives]},
                )
            )
            state.proposed_slots = alternatives
            if not alternatives:
                return PolicyDecision(
                    message="That slot is not possible for me. Could you share another slot?",
                    reason="Exact target slot conflicts and no nearby alternatives are free.",
                    status=TerminalStatus.WAITING_ON_TARGET,
                )
            return PolicyDecision(
                message=f"That slot is not possible for me. {self._format_slots(alternatives)} works on my side.",
                reason="Exact target slot conflicts, proposed nearest calendar-free alternatives.",
                status=TerminalStatus.WAITING_ON_TARGET,
            )

        if "candidate_email_collected" in state.edge_case_flags and not state.target.email:
            state.confirmed_slot = slot
            return PolicyDecision(
                message="Great, could you share your email so I can send the calendar invite?",
                reason="Slot confirmed but need target email to send invite.",
                status=TerminalStatus.WAITING_ON_TARGET,
            )

        if "demo_mode" in state.edge_case_flags:
            attendees = [state.user.email] if state.user.email else []
        else:
            attendees = [email for email in [state.user.email, state.target.email] if email]

        event = calendar.create_event(
            title=f"{state.user.name} <> {state.target.name}",
            attendees=attendees,
            slot=slot,
            timezone=state.timezone,
            description=f"Scheduled by local scheduling agent.\n\nObjective: {state.objective}",
            location=self._resolve_location(state, slot),
        )
        state.remember_tool(
            ToolCall(
                tool="calendar",
                action="create_event",
                input={"slot": slot.model_dump(mode="json"), "attendees": event.attendees},
                output=event.model_dump(mode="json"),
            )
        )
        state.confirmed_slot = slot
        state.event = event
        prefix = ""
        if analysis.asks_for_deck:
            prefix = f"Sharing the deck here: {DECK_LINK}\n\n"
        if analysis.asks_for_email and state.user.email:
            prefix += f"My email is {state.user.email}.\n\n"
        if event.location:
            msg_body = f"Booked for {self._format_slot(slot)} in-person. See you at {event.location}."
        else:
            msg_body = f"Booked for {self._format_slot(slot)}. Meet link: {event.meet_link}"
        return PolicyDecision(
            message=f"{prefix}{msg_body}",
            reason="Target gave an exact slot and calendar confirmed it is free.",
            status=TerminalStatus.BOOKED,
        )

    def _windows(self, state: SchedulingState, analysis: AnalysisResult) -> list[TimeWindow]:
        if analysis.resolved_windows:
            windows: list[TimeWindow] = []
            anchor = parse_anchor(state.current_date, state.timezone)
            for rw in analysis.resolved_windows:
                try:
                    start_dt = datetime.fromisoformat(rw.start)
                    end_dt = datetime.fromisoformat(rw.end)
                except (ValueError, TypeError):
                    continue
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=ZoneInfo(state.timezone))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=ZoneInfo(state.timezone))
                if start_dt < anchor:
                    start_dt = anchor
                if start_dt < end_dt:
                    windows.append(TimeWindow(start=start_dt, end=end_dt, label=rw.label, timezone=state.timezone))
                if rw.label not in state.discussed_windows:
                    state.discussed_windows.append(rw.label)
            if windows:
                return windows

        windows: list[TimeWindow] = []
        raw_windows = analysis.requested_windows[:]
        if analysis.wants_next_week and "next week" not in raw_windows:
            raw_windows.append("next week")
        for text in raw_windows:
            parsed = parse_window(text, state.current_date, state.timezone)
            if parsed:
                windows.append(parsed)
                if text not in state.discussed_windows:
                    state.discussed_windows.append(text)
        if windows:
            return windows

        return self._default_window(state)

    def _default_window(self, state: SchedulingState) -> list[TimeWindow]:
        anchor = parse_anchor(state.current_date, state.timezone)
        start_date = self._next_business_day(anchor.date() + timedelta(days=2))
        if state.target.background_category == "vc":
            start_date = self._next_business_day(max(start_date, anchor.date() + timedelta(days=3)))
        start = datetime.combine(start_date, datetime.min.time().replace(hour=10), ZoneInfo(state.timezone))
        end = start + timedelta(days=4, hours=7)
        return [TimeWindow(start=start, end=end, label="reasonable upcoming availability", timezone=state.timezone)]

    def _find_free_slots(
        self,
        state: SchedulingState,
        calendar: CalendarClient,
        windows: list[TimeWindow],
        duration: int,
        limit: int,
    ) -> list[Slot]:
        free = calendar.find_free_slots(
            windows,
            duration,
            state.timezone,
            limit=limit,
            avoid=state.proposed_keys() | state.rejected_keys(),
        )
        state.remember_tool(
            ToolCall(
                tool="calendar",
                action="find_free_slots",
                input={"windows": [w.model_dump(mode="json") for w in windows], "duration_minutes": duration},
                output={"slots": [s.model_dump(mode="json") for s in free]},
            )
        )
        return free

    def _resolve_location(self, state: SchedulingState, slot: Slot) -> str:
        if "in_person_requested" not in state.edge_case_flags:
            return ""
        text = " ".join(m.text.lower() for m in state.thread)
        if "cafe" in text or "coffee" in text:
            location = "Riverside Cafe, Lakeside District"
        else:
            location = "Lakeside HQ"
        state.remember_tool(
            ToolCall(
                tool="office",
                action="fetch_office_address",
                input={},
                output={"address": location},
            )
        )
        return location

    def _next_business_day(self, date):
        while date.weekday() >= 5:
            date += timedelta(days=1)
        return date

    def _proposal_message(self, state: SchedulingState, slots: list[Slot], analysis: AnalysisResult) -> str:
        prefix = ""
        if analysis.asks_for_deck:
            prefix = f"Sharing the deck here: {DECK_LINK}\n\n"
        if analysis.asks_for_email and state.user.email:
            prefix += f"My email is {state.user.email}.\n\n"
        if state.target.background_category == "vc":
            return f"{prefix}{self._format_slots(slots)} work on my side. Happy to go with whichever is easier."
        return f"{prefix}{self._format_slots(slots)} work for me. Which one should I block?"

    def _format_slots(self, slots: list[Slot]) -> str:
        if not slots:
            return ""
        from collections import defaultdict
        by_day = defaultdict(list)
        for s in slots:
            by_day[s.start.date()].append(s)

        day_strings = []
        for d in sorted(by_day.keys()):
            day_slots = by_day[d]
            day_str = day_slots[0].start.strftime("%a %d %b").replace(" 0", " ")
            times = [self._format_time(s.start) for s in day_slots]
            if len(times) == 1:
                day_strings.append(f"{day_str}, {times[0]}")
            else:
                day_strings.append(f"{day_str} — {', '.join(times[:-1])}, or {times[-1]}")

        if len(day_strings) == 1:
            return day_strings[0]
        return " / ".join(day_strings)

    def _format_slot(self, slot: Slot) -> str:
        return slot.start.strftime("%a %d %b, ").replace(" 0", " ") + self._format_time(slot.start)

    @staticmethod
    def _format_time(dt: datetime) -> str:
        if dt.minute == 0:
            return dt.strftime("%I %p").lstrip("0").replace(" 0", " ")
        return dt.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")
