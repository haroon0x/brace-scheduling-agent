from datetime import datetime
from zoneinfo import ZoneInfo

from scheduler.analyzer import ConversationAnalyzer
from scheduler.calendar.mock import MockCalendarClient
from scheduler.fixtures import FixtureStore
from scheduler.models import AnalysisResult, ResolvedWindow, Slot, TargetIntent, TerminalStatus
from scheduler.policy import SchedulingPolicy


def make_state(conversation_id="s02_mira_vc_linkedin_warm_peer_catchup"):
    store = FixtureStore("fixtures/actors.json", "fixtures/conversations.json")
    return store.make_state(conversation_id)


def test_busy_next_week_respected():
    state = make_state()
    state.add_target_message("This week is packed, can do next week Tuesday afternoon")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.REQUEST_WINDOW,
            requested_windows=["next week Tuesday afternoon"],
            wants_next_week=True,
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.WAITING_ON_TARGET
    assert state.proposed_slots
    assert all(slot.start.date().isoformat() == "2026-05-26" for slot in state.proposed_slots)


def test_rejected_slot_not_repeated():
    state = make_state()
    rejected = Slot(
        start=datetime(2026, 5, 26, 13, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        end=datetime(2026, 5, 26, 13, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        timezone="Asia/Kolkata",
        label="Tuesday afternoon",
    )
    state.rejected_slots.append(rejected)
    SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.REQUEST_WINDOW,
            requested_windows=["next week Tuesday afternoon"],
        ),
        MockCalendarClient(),
    )
    assert rejected.key() not in {slot.key() for slot in state.proposed_slots}


def test_exact_confirmation_books_when_free():
    state = make_state("s01_nadia_vc_linkedin_inbound_deck_then_slot")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.CONFIRM_SLOT,
            exact_slot_text="Friday 3:30",
            asks_for_deck=True,
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.BOOKED
    assert state.event is not None
    assert "meet.google.com" in state.event.meet_link


def test_calendar_conflict_proposes_alternative():
    state = make_state()
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.COUNTER_PROPOSE,
            exact_slot_text="Tuesday 3:00",
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.WAITING_ON_TARGET
    assert "not possible" in decision.message
    assert state.proposed_slots


def test_vc_default_does_not_offer_same_day():
    state = make_state("s04_farah_vc_email_cold_inbound_propose_two_slots")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(intent=TargetIntent.REQUEST_WINDOW),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.WAITING_ON_TARGET
    assert all(slot.start.date().isoformat() > "2026-05-23" for slot in state.proposed_slots)


def test_no_meeting_intent_terminal():
    state = make_state()
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.CLOSE,
            no_meeting_intent=True,
            meeting_intent=False,
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.NO_MEETING_INTENT


def test_heuristic_counter_uses_previous_slot_date():
    state = make_state("s01_nadia_vc_linkedin_inbound_deck_then_slot")
    tz = ZoneInfo("Asia/Kolkata")
    state.proposed_slots.append(
        Slot(
            start=datetime(2026, 5, 29, 15, 30, tzinfo=tz),
            end=datetime(2026, 5, 29, 16, 0, tzinfo=tz),
            timezone="Asia/Kolkata",
            label="Friday 3:30",
        )
    )
    state.add_target_message("Can we make it 4:00 instead?")
    target_slot = Slot(
        start=datetime(2026, 5, 29, 16, 0, tzinfo=tz),
        end=datetime(2026, 5, 29, 16, 30, tzinfo=tz),
        timezone="Asia/Kolkata",
        label="Can we make it 4:00 instead?",
    )
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(intent=TargetIntent.COUNTER_PROPOSE, target_slot=target_slot),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.BOOKED
    assert state.confirmed_slot is not None
    assert state.confirmed_slot.start.isoformat() == "2026-05-29T16:00:00+05:30"


def test_ambiguous_acceptance_after_multiple_slots_does_not_book():
    state = make_state()
    state.proposed_slots.extend(
        [
            Slot(
                start=datetime(2026, 5, 27, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 27, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Wed 11",
            ),
            Slot(
                start=datetime(2026, 5, 27, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 27, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Wed 11:30",
            ),
        ]
    )
    state.add_target_message("yes works")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.CONFIRM_SLOT,
            accepted_previous_slot=True,
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.WAITING_ON_TARGET
    assert state.event is None
    assert "Which" in decision.message


def test_reschedule_today_uses_next_business_day_window():
    state = make_state("s10_vikram_founder_linkedin_reschedule_due_to_illness")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.REQUEST_WINDOW,
            requested_windows=[],
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.WAITING_ON_TARGET
    assert state.proposed_slots
    assert all(slot.start.date().isoformat() == "2026-05-25" for slot in state.proposed_slots)


def test_default_window_skips_weekend():
    state = make_state("s01_nadia_vc_linkedin_inbound_deck_then_slot")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(intent=TargetIntent.REQUEST_WINDOW),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.WAITING_ON_TARGET
    assert state.proposed_slots
    assert all(slot.start.weekday() < 5 for slot in state.proposed_slots)


def test_not_free_this_week_pushes_to_next_week():
    state = make_state()
    # Propose some same-week slots first
    state.add_target_message("sure, let's do it")
    SchedulingPolicy().decide(
        state,
        AnalysisResult(intent=TargetIntent.REQUEST_WINDOW, meeting_intent=True),
        MockCalendarClient(),
    )
    # Now target rejects the week
    state.add_target_message("i am not free this week man")
    analyzer = ConversationAnalyzer(use_llm=False)
    analysis = analyzer.analyze(state)
    assert analysis.wants_next_week
    assert analysis.rejected_slot_texts


def test_bare_time_matches_proposed_slot():
    state = make_state()
    tz = ZoneInfo(state.timezone)
    proposed = [
        Slot(start=datetime(2026, 5, 28, 9, 30, tzinfo=tz), end=datetime(2026, 5, 28, 10, 0, tzinfo=tz)),
        Slot(start=datetime(2026, 5, 28, 10, 0, tzinfo=tz), end=datetime(2026, 5, 28, 10, 30, tzinfo=tz)),
        Slot(start=datetime(2026, 5, 28, 10, 30, tzinfo=tz), end=datetime(2026, 5, 28, 11, 0, tzinfo=tz)),
    ]
    state.proposed_slots = proposed
    state.add_target_message("9:30 works for me")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.CONFIRM_SLOT,
            meeting_intent=True,
            accepted_previous_slot=True,
            target_slot=proposed[0],
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.BOOKED
    assert state.confirmed_slot.start.hour == 9 and state.confirmed_slot.start.minute == 30


def test_in_person_booking_has_location_no_meet():
    state = make_state("s15_divya_engineer_linkedin_recruiting_in_person_tomorrow")
    policy = SchedulingPolicy()
    analysis = AnalysisResult(
        intent=TargetIntent.CONFIRM_SLOT,
        meeting_intent=True,
        accepted_previous_slot=True,
        recovered_proposed_slots=[
            ResolvedWindow(
                start="2026-05-22T16:30:00+05:30",
                end="2026-05-22T17:00:00+05:30",
                label="4:30 PM"
            )
        ]
    )
    decision = policy.decide(state, analysis, MockCalendarClient())
    assert decision.status == TerminalStatus.BOOKED
    assert "in-person" in decision.message
    assert "Lakeside HQ" in decision.message
    assert state.event.location == "Lakeside HQ"
    assert state.event.meet_link == ""


def test_resolved_windows_prioritization():
    state = make_state()
    from scheduler.models import ResolvedWindow
    analysis = AnalysisResult(
        intent=TargetIntent.REQUEST_WINDOW,
        meeting_intent=True,
        resolved_windows=[
            ResolvedWindow(
                start="2026-06-01T09:30:00",
                end="2026-06-05T18:00:00",
                label="week of June 1"
            )
        ]
    )

    policy = SchedulingPolicy()
    windows = policy._windows(state, analysis)
    assert len(windows) == 1
    assert windows[0].start.month == 6
    assert windows[0].start.day == 1


def test_blanket_acceptance_auto_books():
    """'any of those works' should be classified as DELEGATE_CHOICE by the LLM."""
    state = make_state()
    state.proposed_slots.extend(
        [
            Slot(
                start=datetime(2026, 5, 27, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 27, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Wed 11",
            ),
            Slot(
                start=datetime(2026, 5, 27, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 27, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Wed 11:30",
            ),
        ]
    )
    state.add_target_message("any of those works")
    decision = SchedulingPolicy().decide(
        state,
        AnalysisResult(
            intent=TargetIntent.DELEGATE_CHOICE,
            accepted_previous_slot=True,
        ),
        MockCalendarClient(),
    )
    assert decision.status == TerminalStatus.BOOKED
    assert state.confirmed_slot is not None
    assert state.confirmed_slot.label == "Wed 11"


def test_spacing_fixer():
    from scheduler.agent import SchedulingAgent
    assert SchedulingAgent._fix_spacing("Bookedfor Thu 28 May") == "Booked for Thu 28 May"
    assert SchedulingAgent._fix_spacing("Ihave a meeting atThe office") == "I have a meeting at The office"
    assert SchedulingAgent._fix_spacing("Mon25 May works") == "Mon 25 May works"


def test_next_week_skips_proposed_slots_week():
    state = make_state()
    # Assume we already proposed a slot on Wednesday May 27 (week of May 25)
    state.proposed_slots.append(
        Slot(
            start=datetime(2026, 5, 27, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
            end=datetime(2026, 5, 27, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
            timezone="Asia/Kolkata",
            label="Wed 11",
        )
    )
    state.add_target_message("this week doesn't work, next week works")
    
    analyzer = ConversationAnalyzer(use_llm=False)
    analysis = analyzer.analyze(state)
    assert analysis.wants_next_week
    assert len(analysis.resolved_windows) == 1
    # Since week of May 25 was proposed, next week should be week of June 1 (June 01 to June 05)
    start_dt = datetime.fromisoformat(analysis.resolved_windows[0].start)
    assert start_dt.date().isoformat() == "2026-06-01"


def test_delegated_weekend_request_after_multiple_proposals_does_not_ask_which():
    """'saturday works, you choose time' is REQUEST_WINDOW + delegate, not DELEGATE_CHOICE."""
    state = make_state("s10_vikram_founder_linkedin_reschedule_due_to_illness")
    state.proposed_slots.extend(
        [
            Slot(
                start=datetime(2026, 5, 25, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 25, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Mon 10",
            ),
            Slot(
                start=datetime(2026, 5, 25, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 25, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Mon 10:30",
            ),
            Slot(
                start=datetime(2026, 5, 25, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 25, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Mon 11",
            ),
        ]
    )
    state.add_target_message("saturday works for me, you can choose time based on your schedule")
    # With LLM, this would be REQUEST_WINDOW + delegate modifier
    analysis = AnalysisResult(
        intent=TargetIntent.REQUEST_WINDOW,
        intent_modifiers=["delegate"],
        requested_windows=["saturday"],
    )
    decision = SchedulingPolicy().decide(state, analysis, MockCalendarClient())
    # With delegate modifier + weekend skip, it auto-books the nearest Monday slot
    assert decision.status == TerminalStatus.BOOKED
    assert state.confirmed_slot is not None
    assert state.confirmed_slot.start.weekday() == 0  # Monday


def test_next_week_any_time_proposes_next_week():
    """The original bug: 'next week any time?' must find next-week slots, not book this week."""
    state = make_state()
    state.proposed_slots.extend(
        [
            Slot(
                start=datetime(2026, 5, 26, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 26, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Tue 11",
            ),
            Slot(
                start=datetime(2026, 5, 27, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                end=datetime(2026, 5, 27, 16, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
                timezone="Asia/Kolkata",
                label="Wed 4 PM",
            ),
        ]
    )
    state.add_target_message("next week any time ?")
    from scheduler.models import ResolvedWindow
    analysis = AnalysisResult(
        intent=TargetIntent.REQUEST_WINDOW,
        intent_modifiers=["delegate"],
        requested_windows=["next week"],
        resolved_windows=[
            ResolvedWindow(
                start="2026-06-01T09:30:00+05:30",
                end="2026-06-05T18:00:00+05:30",
                label="week of Jun 01",
            )
        ],
    )
    decision = SchedulingPolicy().decide(state, analysis, MockCalendarClient())
    # With delegate modifier, it should auto-book a slot in next week
    assert decision.status == TerminalStatus.BOOKED
    assert state.confirmed_slot is not None
    # Must be in June (next week), NOT in May (this week)
    assert state.confirmed_slot.start.month == 6
