import json
from datetime import datetime
from zoneinfo import ZoneInfo

from scheduler.analyzer import ConversationAnalyzer
from scheduler.fixtures import FixtureStore
from scheduler.models import Slot


class FakeLLM:
    enabled = True

    def __init__(self, payload):
        self.payload = payload
        self.system = ""
        self.user = ""

    def extract_json(self, system: str, user: str):
        self.system = system
        self.user = user
        return self.payload


FIXTURE_ACTORS = "fixtures/actors.json"
FIXTURE_CONVERSATIONS = "fixtures/conversations.json"


def make_state(conversation_id="s01_nadia_vc_linkedin_inbound_deck_then_slot"):
    store = FixtureStore(FIXTURE_ACTORS, FIXTURE_CONVERSATIONS)
    return store.make_state(conversation_id)


def valid_payload(**overrides):
    payload = {
        "meeting_intent": True,
        "no_meeting_intent": False,
        "exact_slot_text": None,
        "accepted_previous_slot": False,
        "rejected_slot_texts": [],
        "requested_windows": [],
        "wants_next_week": False,
        "asks_for_deck": False,
        "asks_for_email": False,
        "wants_short_call": False,
        "timezone_hint": None,
        "confidence": 0.8,
        "notes": "test",
    }
    payload.update(overrides)
    return payload


def test_prompt_contains_structured_memory():
    state = make_state()
    state.proposed_slots.append(
        Slot(
            start=datetime(2026, 5, 29, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
            end=datetime(2026, 5, 29, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
            timezone="Asia/Kolkata",
            label="Friday 3:30",
        )
    )
    fake = FakeLLM(valid_payload())
    ConversationAnalyzer(llm=fake).analyze(state)
    prompt = json.loads(fake.user)
    assert prompt["current_date"] == "2026-05-23"
    assert prompt["latest_target_message"].startswith("Sounds great")
    assert prompt["previously_proposed_slots"][0]["label"] == "Friday 3:30"
    assert prompt["target"]["category"] == "vc"


def test_llm_accepted_previous_slot_is_cleared_without_prior_proposal():
    state = make_state()
    fake = FakeLLM(valid_payload(accepted_previous_slot=True))
    analyzer = ConversationAnalyzer(llm=fake)
    result = analyzer.analyze(state)
    assert result.accepted_previous_slot is False
    assert analyzer.last_source == "llm"


def test_llm_failure_records_fallback_source():
    class BrokenLLM:
        enabled = True
        model = "broken-test-model"

        def extract_json(self, system: str, user: str):
            raise RuntimeError("boom")

    state = make_state()
    analyzer = ConversationAnalyzer(llm=BrokenLLM())
    import pytest
    with pytest.raises(RuntimeError, match="boom"):
        analyzer.analyze(state)
    assert analyzer.last_source == "llm_error"
    assert analyzer.last_error == "boom"


def test_heuristic_repairs_unparseable_llm_exact_slot():
    # Verify signal extraction under heuristic-only mode
    state = make_state()
    result = ConversationAnalyzer(use_llm=False).analyze(state)
    assert result.exact_slot_text == "friday 3:30"
    assert result.asks_for_deck is True


def test_llm_native_history_recovery():
    # Verify that LLM-native slot history recovery populated recovered slots correctly
    state = make_state()
    fake_payload = valid_payload(
        recovered_proposed_slots=[
            {
                "start": "2026-05-22T16:30:00+05:30",
                "end": "2026-05-22T17:00:00+05:30",
                "label": "Friday 4:30 PM"
            }
        ],
        recovered_rejected_slots=[
            {
                "start": "2026-05-22T15:00:00+05:30",
                "end": "2026-05-22T15:30:00+05:30",
                "label": "Friday 3:00 PM"
            }
        ]
    )
    fake = FakeLLM(fake_payload)
    result = ConversationAnalyzer(llm=fake).analyze(state)
    assert len(result.recovered_proposed_slots) == 1
    assert result.recovered_proposed_slots[0].label == "Friday 4:30 PM"
    assert len(result.recovered_rejected_slots) == 1
    assert result.recovered_rejected_slots[0].label == "Friday 3:00 PM"


def test_heuristic_merges_requested_windows_and_rejections():
    state = make_state("s02_mira_vc_linkedin_warm_peer_catchup")
    state.add_target_message("This week is impossible, can do next week Tuesday after 4")
    # In LLM-native mode, we mock the LLM returned payload which is trusted fully
    fake = FakeLLM(valid_payload(requested_windows=["next week", "Tuesday after 4"], wants_next_week=True))
    result = ConversationAnalyzer(llm=fake).analyze(state)
    assert "next week" in result.requested_windows
    assert result.wants_next_week is True


def test_llm_range_exact_slot_is_downgraded_to_window():
    state = make_state("s06_anita_vc_linkedin_busy_but_interested")
    fake = FakeLLM(valid_payload(exact_slot_text="Wednesday 27th 11:00-13:00"))
    result = ConversationAnalyzer(llm=fake).analyze(state)
    assert result.exact_slot_text is None
    assert "Wednesday 27th 11:00-13:00" in result.requested_windows


def test_heuristic_captures_day_time_range_window():
    state = make_state("s06_anita_vc_linkedin_busy_but_interested")
    result = ConversationAnalyzer(use_llm=False).analyze(state)
    assert result.exact_slot_text is None
    assert state.thread[-1].text in result.requested_windows
    assert result.resolved_windows
    assert result.resolved_windows[0].start == "2026-05-27T11:00:00+05:30"
    assert result.resolved_windows[0].end == "2026-05-27T13:00:00+05:30"


def test_heuristic_captures_weekend_requested_window_after_proposals():
    state = make_state("s10_vikram_founder_linkedin_reschedule_due_to_illness")
    state.add_target_message("saturday works for me, you can choose time based on your schedule")
    result = ConversationAnalyzer(use_llm=False).analyze(state)
    assert "saturday" in result.requested_windows
