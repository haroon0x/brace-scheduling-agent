from scheduler.agent import SchedulingAgent
from scheduler.analyzer import ConversationAnalyzer
from scheduler.calendar.mock import MockCalendarClient
from scheduler.cli import _run_eval_case
from scheduler.fixtures import FixtureStore
from scheduler.messages.linkedin import LinkedInMessageClient
from scheduler.models import TerminalStatus
from scheduler.policy import DECK_LINK, SchedulingPolicy


def test_agent_records_tool_calls_and_books():
    store = FixtureStore("fixtures/actors.json", "fixtures/conversations.json")
    state = store.make_state("s01_nadia_vc_linkedin_inbound_deck_then_slot")
    agent = SchedulingAgent(
        analyzer=ConversationAnalyzer(use_llm=False),
        policy=SchedulingPolicy(),
        calendar=MockCalendarClient(),
        messenger=LinkedInMessageClient(),
    )
    state = agent.step(state)
    assert state.status == TerminalStatus.BOOKED
    assert state.event is not None
    assert state.tool_calls[-1].tool == "message.linkedin"
    assert state.tool_calls[0].meta["source"] == "heuristic"
    assert DECK_LINK in state.last_message
    assert "Meet link" in state.last_message


def test_local_eval_cases_pass_without_llm():
    store = FixtureStore("fixtures/actors.json", "fixtures/conversations.json")
    rows = []
    for scenario in ["1", "6", "10"]:
        rows.extend(
            _run_eval_case(
                store,
                scenario,
                calendar="mock",
                use_llm=False,
                current_date="2026-05-23",
            )
        )
    assert rows
    assert all(row["passed"] for row in rows)
    assert any(row["step"] == "ambiguous_acceptance" for row in rows)


def test_local_eval_llm_mode_fails_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = FixtureStore("fixtures/actors.json", "fixtures/conversations.json")
    rows = _run_eval_case(
        store,
        "1",
        calendar="mock",
        use_llm=True,
        current_date="2026-05-23",
    )
    assert rows[0]["passed"] is False
    assert "expected LLM extraction" in rows[0]["failures"][0]
