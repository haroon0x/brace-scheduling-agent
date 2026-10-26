from __future__ import annotations

import json
import re
from datetime import datetime as dt_class, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .dates import default_duration, parse_anchor, parse_slot, parse_time_range, parse_time_text, parse_window, WORKDAY_START, WORKDAY_END
from .llm import OpenRouterClient
from .models import AnalysisResult, ResolvedWindow, SchedulingState, Slot, TargetIntent


SYSTEM_PROMPT = """You are the scheduling agent and intent classifier for a meeting scheduling agent.

Your PRIMARY job is to classify the target's INTENT into exactly one category, then extract supporting scheduling signals and reconstruct the conversation history.

Use current_date and timezone as the only anchors for relative language. Classify what the TARGET communicated, focusing on latest_target_message. Ignore the scheduler's own messages (user/agent) for extraction — they are context only.

Return strict JSON matching exactly:
{
  "reasoning": "string",
  "intent": "confirm_slot" | "counter_propose" | "delegate_choice" | "request_window" | "reject" | "close",
  "intent_modifiers": string[],
  "suggested_action": "book" | "propose" | "clarify" | "reschedule" | "close" | null,
  "meeting_intent": boolean,
  "no_meeting_intent": boolean,
  "exact_slot_text": string|null,
  "accepted_previous_slot": boolean,
  "rejected_slot_texts": string[],
  "requested_windows": string[],
  "resolved_windows": [
    {
      "start": "string",
      "end": "string",
      "label": "string"
    }
  ],
  "recovered_proposed_slots": [
    {
      "start": "string",
      "end": "string",
      "label": "string"
    }
  ],
  "recovered_rejected_slots": [
    {
      "start": "string",
      "end": "string",
      "label": "string"
    }
  ],
  "wants_next_week": boolean,
  "asks_for_deck": boolean,
  "asks_for_email": boolean,
  "wants_short_call": boolean,
  "timezone_hint": string|null,
  "confidence": number,
  "notes": string
}

INTENT CLASSIFICATION (most important field):
Pick exactly ONE intent for the target's latest message:
- "confirm_slot": Target accepts/confirms a SPECIFIC previously proposed slot. Examples: "Tuesday works", "confirmed", "let's do the 3 PM one", "yes" (when only one slot was proposed).
- "counter_propose": Target proposes a DIFFERENT specific time than what was offered. Examples: "make it 4 instead", "can we do Thursday 3:30?", "how about Friday at 2?".
- "delegate_choice": Target delegates the slot choice to the agent FROM ALREADY PROPOSED OPTIONS, with NO new time constraint. Examples: "any of those works", "you pick", "whichever is easier", "just book one".
- "request_window": Target specifies a NEW time window or range (even if they also say "you pick" within that window). Examples: "next week any time", "Thursday afternoon", "after 4 tomorrow", "next week you decide". KEY: if a time window is specified alongside delegation, this is request_window NOT delegate_choice.
- "reject": Target rejects proposals without giving alternatives. Examples: "can't make this week", "busy", "none of those work".
- "close": Target clearly refuses a meeting entirely. Examples: "not interested", "pass", "no need".

INTENT MODIFIERS (list of strings, can be empty):
- "delegate": Target wants the agent to pick within the window. Applies to request_window when target says "any time", "you choose", "pick one".
- Only use "delegate" modifier for now.
- CRITICAL: Do NOT set the "delegate" modifier for mere questions or queries like "next week ?" or "tomorrow ?" where the target is not explicitly saying "any time", "you pick", or "you choose".

CRITICAL EXAMPLES:
- "next week any time?" → intent: "request_window", intent_modifiers: ["delegate"], requested_windows: ["next week"]
- "any of those works" → intent: "delegate_choice", intent_modifiers: []
- "you pick, but next week" → intent: "request_window", intent_modifiers: ["delegate"], requested_windows: ["next week"]
- "Tuesday 3 PM" (was proposed) → intent: "confirm_slot"
- "Tuesday 3 PM" (was NOT proposed) → intent: "counter_propose"
- "make it 4 instead" → intent: "counter_propose"
- "this week doesn't work" → intent: "reject"
- "can't do this week, next week works" → intent: "request_window", requested_windows: ["next week"]

RECOVERED SLOTS HISTORY EXTRACTION:
- Scan the entire thread history for all time slots proposed by the scheduler (sender user or agent). For each slot found, resolve it into concrete ISO-8601 start and end bounds (assume 30-minute duration unless specified otherwise) and place it in "recovered_proposed_slots".
- Scan the entire thread history for all slots that the target explicitly turned down or rejected, resolve them to ISO-8601, and place them in "recovered_rejected_slots".

Other Rules:
- Use "reasoning" to analyze the thread. Note what was previously proposed, what the target wants, and how to resolve relative dates.
- When resolving a requested window from a vague or short message (like "next week ?" or "tomorrow ?"), always check the thread history. If the target previously specified a narrower constraint (e.g., "Wednesday or Thursday next week"), the resolved window MUST remain constrained to that narrower subset, unless the target explicitly states they are expanding it (e.g., "actually any day next week works").
- Extract scheduling signals from the TARGET's messages only, never from the scheduler's own messages.
- Set no_meeting_intent only for clear refusal such as "not interested", "pass", "no need".
- Set accepted_previous_slot only when target clearly accepts one of previously_proposed_slots.
- If target counters with "make it 4 instead", put recovered date + time in exact_slot_text when prior context gives date.
- Put phrases like "Tuesday afternoon", "next week", "after 4" in requested_windows.
- For each entry in requested_windows, resolve it into concrete ISO-8601 datetime boundaries inside resolved_windows using business hours (09:30-18:00) relative to current_date and timezone.
- Rejection-based Resolution: If previously_proposed_slots contains slots in a week and target rejects "this week", deduce the following week (next Monday to Friday) as resolved_windows.
- Put exact phrases containing both date/day context and time in exact_slot_text, e.g. "Friday 3:30", "tomorrow 5pm".
- rejected_slot_texts should include target text that rejects or changes prior proposals.
- If target is cancelling/rescheduling, set exact_slot_text to null.
- asks_for_deck is true for deck, one-pager, brief, memo, or context request.
- asks_for_email is true when target asks for the scheduler's email or asks to move to email.
- confidence is 0 to 1.
- notes should be short and explain ambiguous extraction choices.
"""



class ConversationAnalyzer:
    def __init__(self, llm: OpenRouterClient | None = None, use_llm: bool = True) -> None:
        self.llm = llm or OpenRouterClient()
        self.use_llm = use_llm
        self.last_source = "not_run"
        self.last_error: str | None = None

    def analyze(self, state: SchedulingState) -> AnalysisResult:
        self.last_error = None
        latest = state.thread[-1].text if state.thread else ""

        if not self.use_llm:
            self.last_source = "heuristic"
            result = self._heuristic(latest, state)
            result = self._resolve_target_slot(result, state)
            result = self._resolve_result_windows(result, latest, state)
            result.intent = self._infer_intent_from_signals(result, state)
            return result

        if not self.llm.enabled:
            key_name = "GEMINI_API_KEY" if self.llm.provider == "gemini" else "OPENROUTER_API_KEY"
            self.last_source = f"heuristic_no_{self.llm.provider}_key"
            self.last_error = f"{key_name} is not set"
            raise RuntimeError(f"{key_name} is not set")

        try:
            payload = self.llm.extract_json(SYSTEM_PROMPT, self._thread_prompt(state))
            result = AnalysisResult.model_validate(payload)
        except (RuntimeError, ValidationError, ValueError) as exc:
            self.last_source = "llm_error"
            self.last_error = str(exc)
            raise exc

        self.last_source = "llm"

        if result.accepted_previous_slot and not state.proposed_slots:
            result.accepted_previous_slot = False

        if result.exact_slot_text:
            text_lower = result.exact_slot_text.lower()
            is_range = "-" in result.exact_slot_text or " to " in text_lower or " between " in text_lower
            if is_range or parse_slot(result.exact_slot_text, state.current_date, timezone=state.timezone) is None:
                if result.exact_slot_text not in result.requested_windows:
                    result.requested_windows.append(result.exact_slot_text)
                result.exact_slot_text = None

        result = self._resolve_target_slot(result, state)
        result = self._resolve_result_windows(result, latest, state)
        return result

    def _resolve_target_slot(self, result: AnalysisResult, state: SchedulingState) -> AnalysisResult:
        duration = 25 if result.wants_short_call else 30
        latest_text = state.thread[-1].text if state.thread else ""

        all_proposals = list(state.proposed_slots)
        if not all_proposals and result.recovered_proposed_slots:
            for rw in result.recovered_proposed_slots:
                try:
                    s = Slot(start=dt_class.fromisoformat(rw.start), end=dt_class.fromisoformat(rw.end),
                             timezone=state.timezone, label=rw.label)
                    all_proposals.append(s)
                except (ValueError, TypeError):
                    continue

        if result.exact_slot_text:
            parsed = parse_slot(result.exact_slot_text, state.current_date, duration, state.timezone)
            if parsed:
                result.target_slot = parsed
                return result

        if result.exact_slot_text or (not result.accepted_previous_slot and all_proposals):
            counter = self._resolve_counter_from_context_str(latest_text, all_proposals)
            if counter:
                result.target_slot = counter
                return result

        if not result.target_slot and all_proposals:
            bare_match = self._resolve_bare_time_match_str(latest_text, all_proposals)
            if bare_match:
                result.target_slot = bare_match
                return result

        if result.accepted_previous_slot and all_proposals:
            parsed_latest = parse_slot(latest_text, state.current_date, duration, state.timezone)
            if parsed_latest:
                for p in all_proposals:
                    if p.start == parsed_latest.start:
                        result.target_slot = p
                        return result
            if result.exact_slot_text:
                parsed_exact = parse_slot(result.exact_slot_text, state.current_date, duration, state.timezone)
                if parsed_exact:
                    for p in all_proposals:
                        if p.start == parsed_exact.start:
                            result.target_slot = p
                            return result
            bare_time = parse_time_text(latest_text)
            if bare_time:
                ref = all_proposals[0]
                start = dt_class.combine(ref.start.date(), bare_time, ref.start.tzinfo)
                result.target_slot = Slot(start=start, end=start + (ref.end - ref.start), timezone=ref.timezone, label=latest_text)
                return result

        return result

    def _resolve_counter_from_context(self, text: str, state: SchedulingState) -> Slot | None:
        if not state.proposed_slots:
            return None
        return self._resolve_counter_from_context_str(text, state.proposed_slots)

    def _resolve_bare_time_match(self, text: str, state: SchedulingState) -> Slot | None:
        return self._resolve_bare_time_match_str(text, state.proposed_slots)

    @staticmethod
    def _resolve_counter_from_context_str(text: str, proposals: list[Slot]) -> Slot | None:
        if not proposals:
            return None
        lowered = text.lower()
        if not any(token in lowered for token in ["instead", "make it", "can we", "could we"]):
            return None
        parsed_time = parse_time_text(text)
        if not parsed_time:
            return None
        previous = proposals[-1]
        start = dt_class.combine(previous.start.date(), parsed_time, previous.start.tzinfo)
        duration = previous.end - previous.start
        return Slot(start=start, end=start + duration, timezone=previous.timezone, label=text)

    @staticmethod
    def _resolve_bare_time_match_str(text: str, proposals: list[Slot]) -> Slot | None:
        t = parse_time_text(text)
        if not t:
            return None
        matches = [s for s in proposals if s.start.hour == t.hour and s.start.minute == t.minute]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            matches = [s for s in proposals if s.start.hour == t.hour]
            if len(matches) == 1:
                return matches[0]
        return None

    def _resolve_result_windows(self, result: AnalysisResult, latest: str, state: SchedulingState) -> AnalysisResult:
        if result.resolved_windows:
            return result

        is_this_week_rejection = any(w in latest.lower() for w in [
            "this week", "not free", "busy", "doesn't work", "doesnt work",
            "cant make", "can't make", "does not work", "not this week",
        ])

        for text in result.requested_windows:
            if "next week" in text.lower() or (result.wants_next_week and is_this_week_rejection):
                anchor = parse_anchor(state.current_date, state.timezone)
                if state.proposed_slots:
                    latest_slot = state.proposed_slots[-1]
                    ref_monday = latest_slot.start.date() - timedelta(days=latest_slot.start.weekday())
                else:
                    ref_monday = anchor.date() - timedelta(days=anchor.weekday())
                    if ref_monday <= anchor.date():
                        ref_monday = anchor.date() + timedelta(days=(7 - anchor.weekday()))
                next_monday = ref_monday + timedelta(days=7)
                if state.proposed_slots:
                    proposed_monday = state.proposed_slots[-1].start.date() - timedelta(
                        days=state.proposed_slots[-1].start.weekday()
                    )
                    if next_monday <= proposed_monday:
                        next_monday = proposed_monday + timedelta(days=7)
                start = dt_class.combine(next_monday, WORKDAY_START, ZoneInfo(state.timezone))
                end = dt_class.combine(next_monday + timedelta(days=4), WORKDAY_END, ZoneInfo(state.timezone))
                label = f"week of {next_monday.strftime('%b %d')}"
                if is_this_week_rejection:
                    label += " (skipping rejected week)"
                if not any("week of" in rw.label for rw in result.resolved_windows):
                    result.resolved_windows.append(
                        ResolvedWindow(start=start.isoformat(), end=end.isoformat(), label=label)
                    )
                if state.proposed_slots:
                    rejected_label = f"week of {ref_monday.strftime('%b %d')}"
                    if rejected_label not in state.discussed_windows:
                        state.discussed_windows.append(rejected_label)
            else:
                parsed = parse_window(text, state.current_date, state.timezone)
                if parsed:
                    start_val = parsed.start
                    end_val = parsed.end
                    if "geographic_constraint" in state.edge_case_flags and "thursday" in text.lower():
                        start_val = parsed.start.replace(hour=17, minute=30, second=0, microsecond=0)
                        end_val = start_val + timedelta(hours=1)
                    result.resolved_windows.append(
                        ResolvedWindow(
                            start=start_val.isoformat(),
                            end=end_val.isoformat(),
                            label=parsed.label
                        )
                    )
        return result

    def _thread_prompt(self, state: SchedulingState) -> str:
        latest_target = next((message.text for message in reversed(state.thread) if message.sender == "target"), "")
        payload = {
            "current_date": state.current_date,
            "timezone": state.timezone,
            "channel": state.channel.value,
            "objective": state.objective,
            "target": {
                "name": state.target.name,
                "headline": state.target.headline,
                "category": state.target.background_category,
                "company": state.target.company,
            },
            "latest_target_message": latest_target,
            "thread": [message.model_dump(mode="json") for message in state.thread[-12:]],
            "previously_proposed_slots": [slot.model_dump(mode="json") for slot in state.proposed_slots],
            "rejected_slots": [slot.model_dump(mode="json") for slot in state.rejected_slots],
            "rejected_texts": state.rejected_texts,
            "discussed_windows": state.discussed_windows,
        }
        return json.dumps(payload, indent=2)

    _ACCEPTANCE_PATTERNS = [
        r"\b(confirm|confirmed|works|sounds good|sounds great|looks good|looks great)\b",
        r"\b(done|yes|yep|yeah|ok|okay|perfect|love it|book it|let.s do it)\b",
        r"\b(any time|anytime|any of them|all of them|either works|either of them)\b",
        r"\b(drop a calendar invite|send (?:a|the|me an?) invite|send (?:a|the) calendar invite)\b",
    ]

    @staticmethod
    def _is_acceptance(lowered: str) -> bool:
        for pattern in ConversationAnalyzer._ACCEPTANCE_PATTERNS:
            if re.search(pattern, lowered):
                return True
        return False

    @staticmethod
    def _infer_intent_from_signals(result: AnalysisResult, state: SchedulingState) -> TargetIntent:
        if result.no_meeting_intent:
            return TargetIntent.CLOSE

        has_proposals = bool(state.proposed_slots) or bool(result.recovered_proposed_slots)

        if result.accepted_previous_slot and has_proposals:
            latest = state.thread[-1].text.lower() if state.thread else ""
            if any(x in latest for x in ["any of them", "all of them", "either works", "you pick", "whichever", "any of those", "all work", "both work", "either one"]):
                return TargetIntent.DELEGATE_CHOICE
            if "delegate" in result.intent_modifiers:
                return TargetIntent.DELEGATE_CHOICE
            return TargetIntent.CONFIRM_SLOT

        if result.target_slot:
            for p in state.proposed_slots:
                if p.start == result.target_slot.start:
                    return TargetIntent.CONFIRM_SLOT
            return TargetIntent.COUNTER_PROPOSE

        if result.exact_slot_text and result.accepted_previous_slot:
            return TargetIntent.CONFIRM_SLOT

        if result.rejected_slot_texts and not result.requested_windows:
            return TargetIntent.REJECT

        return TargetIntent.REQUEST_WINDOW

    def _heuristic(self, text: str, state: SchedulingState) -> AnalysisResult:
        lowered = text.lower()
        asks_for_deck = "deck" in lowered or "brief" in lowered or "one-pager" in lowered
        asks_for_email = "email" in lowered or "mail" in lowered
        no_meeting = any(term in lowered for term in ["not interested", "no need", "pass for now"])
        accepted = self._is_acceptance(lowered)

        rejected_texts = []
        if any(term in lowered for term in ["conflict", "can't", "cannot", "instead", "make it", "not this week", "not free this week", "busy this week", "does not work"]):
            rejected_texts.append(text)

        wants_next_week = any(term in lowered for term in ["next week", "not this week", "not free this week", "busy this week", "does not work this week", "this week does not work"])

        requested_windows = []
        for phrase in [
            "next week", "tomorrow", "today", "morning", "afternoon",
            "evening", "after 4", "after 5", "this week",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        ]:
            if phrase == "today" and any(x in lowered for x in ["fever", "viral", "sick", "pushing", "reschedule", "cancel"]):
                continue
            if phrase == "tomorrow" and any(x in lowered for x in ["fever", "viral", "sick"]):
                continue
            if phrase in lowered:
                requested_windows.append(phrase)

        _MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        has_day_context = bool(
            re.search(
                r"\b(mon|monday|tue|tuesday|wed|wednesday|thu|thursday|fri|friday|sat|saturday|sun|sunday|tomorrow|today)\b",
                lowered,
            )
        ) or bool(
            re.search(rf"\b{_MONTH_RE}\b", lowered)
        ) or bool(
            re.search(r"\b\d{1,2}[/-]\d{1,2}\b", lowered)
        )
        if parse_time_range(text) and has_day_context and parse_window(text, state.current_date, state.timezone):
            requested_windows.insert(0, text)

        exact = None
        exact_match = re.search(
            r"\b((?:mon|tue|wed|thu|fri|sat|sun)(?:day)?|tomorrow|today)\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|ist)?\b",
            lowered,
        )
        if not exact_match:
            exact_match = re.search(
                rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*(?:at\s+)?\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm|ist)?\b",
                lowered,
            )
        if not exact_match:
            exact_match = re.search(
                rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}\s*,?\s*(?:at\s+)?\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm|ist)?\b",
                lowered,
            )
        if not exact_match:
            exact_match = re.search(
                r"\b\d{1,2}[/-]\d{1,2}\s*(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|ist)?\b",
                lowered,
            )
        if exact_match:
            exact = exact_match.group(0).strip()
        elif len(state.proposed_slots) == 1 and accepted:
            exact = state.proposed_slots[-1].label
        elif not parse_time_range(text) and re.search(r"\b\d{1,2}(?::\d{2})\s*(?:am|pm|ist)?\b", lowered):
            _day_cues = [
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "tomorrow", "today",
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            ]
            for cue in _day_cues:
                if cue in lowered:
                    exact = text
                    break

        recovered_proposed, recovered_rejected = self._recover_slots_from_thread(state)

        suggested = None
        if no_meeting:
            suggested = "close"
        elif accepted:
            suggested = "book"
        elif wants_next_week or rejected_texts:
            suggested = "propose"

        return AnalysisResult(
            reasoning="heuristic signal extraction",
            suggested_action=suggested,
            meeting_intent=not no_meeting,
            no_meeting_intent=no_meeting,
            exact_slot_text=exact,
            accepted_previous_slot=accepted and (bool(state.proposed_slots) or bool(recovered_proposed)),
            rejected_slot_texts=rejected_texts,
            requested_windows=requested_windows,
            resolved_windows=[],
            recovered_proposed_slots=recovered_proposed,
            recovered_rejected_slots=recovered_rejected,
            wants_next_week=wants_next_week,
            asks_for_deck=asks_for_deck,
            asks_for_email=asks_for_email,
            wants_short_call="short" in lowered or "quick" in lowered,
            timezone_hint="Asia/Kolkata" if "ist" in lowered else None,
            confidence=0.6,
            notes="heuristic signal extraction",
        )

    def _recover_slots_from_thread(self, state: SchedulingState) -> tuple[list[ResolvedWindow], list[ResolvedWindow]]:
        proposed: list[ResolvedWindow] = []
        rejected: list[ResolvedWindow] = []
        duration = default_duration(state.edge_case_flags)

        for msg in state.thread:
            if msg.sender not in ("user", "agent"):
                continue
            parsed = parse_slot(msg.text, state.current_date, duration, state.timezone)
            if parsed:
                rw = ResolvedWindow(
                    start=parsed.start.isoformat(),
                    end=parsed.end.isoformat(),
                    label=msg.text[:80],
                )
                if rw not in proposed:
                    proposed.append(rw)

        for msg in state.thread:
            if msg.sender != "target":
                continue
            lowered = msg.text.lower()
            if any(term in lowered for term in ["can't", "cannot", "conflict", "busy", "doesn't work", "does not work"]):
                for rw in proposed:
                    if rw.label[:30].lower() in lowered:
                        rejected.append(rw)

        return proposed, rejected
