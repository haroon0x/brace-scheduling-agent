from __future__ import annotations

import json
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import SchedulingAgent
from .analyzer import ConversationAnalyzer
from .calendar.google import GoogleCalendarClient
from .calendar.mock import MockCalendarClient
from .fixtures import FixtureStore
from .llm import OpenRouterClient
from .messages.email import EmailMessageClient
from .messages.linkedin import LinkedInMessageClient
from .models import Channel, SchedulingState, TerminalStatus
from .policy import DECK_LINK, SchedulingPolicy

app = typer.Typer(no_args_is_help=True)
console = Console()

# Default demo dataset ships in-repo (fictional fixtures). Point
# SCHEDULER_ACTORS / SCHEDULER_CONVERSATIONS at your own JSON files to run
# the agent against a different dataset.
DEFAULT_EVAL_SCENARIOS = (
    "s01_nadia_vc_linkedin_inbound_deck_then_slot,"
    "s05_deepak_vc_email_senior_partner_terse,"
    "s06_anita_vc_linkedin_busy_but_interested,"
    "s08_jai_founder_linkedin_warm_peer_quick_yes,"
    "s10_vikram_founder_linkedin_reschedule_due_to_illness,"
    "s13_samir_founder_linkedin_old_friend_ambiguous,"
    "s14_meera_engineer_linkedin_in_person_location,"
    "s17_siddharth_engineer_linkedin_classic_candidate_flow"
)


def default_actors_path() -> Path:
    return Path(os.environ.get("SCHEDULER_ACTORS", "fixtures/actors.json"))


def default_conversations_path() -> Path:
    return Path(os.environ.get("SCHEDULER_CONVERSATIONS", "fixtures/conversations.json"))


def build_agent(state: SchedulingState, calendar_mode: str, use_llm: bool) -> SchedulingAgent:
    if calendar_mode == "google":
        calendar = GoogleCalendarClient()
    else:
        calendar = MockCalendarClient()
    messenger = EmailMessageClient() if state.channel == Channel.EMAIL else LinkedInMessageClient()
    llm = OpenRouterClient() if use_llm else None
    return SchedulingAgent(
        analyzer=ConversationAnalyzer(llm=llm, use_llm=use_llm),
        policy=SchedulingPolicy(),
        calendar=calendar,
        messenger=messenger,
        llm=llm if (llm and llm.enabled) else None,
    )


def resolve_conversation_id(store: FixtureStore, scenario: str) -> str:
    scenario = scenario.strip()
    items = store.list_conversations()
    if scenario.isdigit():
        index = int(scenario) - 1
        if index < 0 or index >= len(items):
            raise typer.BadParameter(f"Scenario index must be between 1 and {len(items)}")
        return items[index]["conversation_id"]
    return scenario


def print_conversations(store: FixtureStore) -> None:
    table = Table(title="Scenarios")
    table.add_column("#", justify="right")
    table.add_column("Conversation")
    table.add_column("Target")
    table.add_column("Channel")
    table.add_column("Flags")
    for idx, item in enumerate(store.list_conversations(), start=1):
        table.add_row(
            str(idx),
            item["conversation_id"],
            item["target_actor_name"],
            item["channel"],
            ", ".join(item.get("edge_case_flags", [])[:3]),
        )
    console.print(table)


def print_thread(state: SchedulingState) -> None:
    lines = []
    for message in state.thread:
        who = state.user.name if message.sender in {"user", "agent"} else state.target.name
        lines.append(f"[bold]{who}[/bold]: {message.text}")
    console.print(Panel("\n\n".join(lines), title="Thread"))


def print_summary(state: SchedulingState) -> None:
    console.print(
        Panel(
            f"[bold]Status[/bold]: {state.status.value}\n"
            f"[bold]Reason[/bold]: {state.last_reason}\n"
            f"[bold]Last message[/bold]: {state.last_message}",
            title="Agent Output",
        )
    )


def _eval_checks(state: SchedulingState, scenario: str, step: str, require_llm: bool = False) -> list[str]:
    failures: list[str] = []
    if state.confirmed_slot and state.confirmed_slot.start.date().isoformat() < state.current_date:
        failures.append("confirmed slot is before current_date")
    for slot in state.proposed_slots:
        if slot.start.date().isoformat() < state.current_date:
            failures.append("proposed slot is before current_date")
            break
    if require_llm:
        analyzer_call = next((call for call in reversed(state.tool_calls) if call.tool == "analyzer"), None)
        source = analyzer_call.meta.get("source") if analyzer_call else None
        if source != "llm":
            error = analyzer_call.meta.get("error") if analyzer_call else "missing analyzer call"
            failures.append(f"expected LLM extraction, got {source or 'none'}: {error}")
    if scenario.startswith("s01_") and step == "initial":
        if state.status != TerminalStatus.BOOKED:
            failures.append("expected booked status")
        if state.event is None:
            failures.append("expected calendar event")
        if DECK_LINK not in state.last_message:
            failures.append("expected deck link in reply")
        if "Meet link" not in state.last_message and "in-person" not in state.last_message:
            failures.append("expected Meet link or in-person in reply")
        if state.confirmed_slot and state.confirmed_slot.start.date().isoformat() != "2026-05-29":
            failures.append("expected Friday slot after current_date, 2026-05-29")
    elif scenario.startswith("s05_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if len(state.proposed_slots) != 1:
            failures.append(f"expected exactly one proposed slot for senior partner, got {len(state.proposed_slots)}")
    elif scenario.startswith("s05_") and step == "confirmation":
        if state.status != TerminalStatus.BOOKED:
            failures.append("expected booked status")
        if state.event is None:
            failures.append("expected calendar event")
    elif scenario.startswith("s06_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if len(state.proposed_slots) < 2:
            failures.append("expected multiple proposed slots")
        if state.event is not None:
            failures.append("must not book before a precise acceptance")
        if any(slot.start.date().isoformat() != "2026-05-27" for slot in state.proposed_slots):
            failures.append("expected scenario 6 slots on 2026-05-27")
        if any(slot.start.hour < 11 or slot.end.hour > 13 for slot in state.proposed_slots):
            failures.append("expected scenario 6 slots inside 11:00-13:00 window")
    elif scenario.startswith("s06_") and step == "ambiguous_acceptance":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status after ambiguous acceptance")
        if state.event is not None:
            failures.append("must not book ambiguous acceptance")
        if "Which" not in state.last_message:
            failures.append("expected clarification prompt")
    elif scenario.startswith("s08_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if not state.proposed_slots:
            failures.append("expected proposed slots for tomorrow")
    elif scenario.startswith("s08_") and step == "confirmation":
        if state.status != TerminalStatus.BOOKED:
            failures.append("expected booked status")
        if state.event is None:
            failures.append("expected calendar event")
    elif scenario.startswith("s10_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if not state.proposed_slots:
            failures.append("expected proposed slots")
        if any(slot.start.date().isoformat() != "2026-05-25" for slot in state.proposed_slots):
            failures.append("expected next business day slots on 2026-05-25")
    elif scenario.startswith("s13_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if not state.proposed_slots:
            failures.append("expected proposed slots")
    elif scenario.startswith("s14_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if not state.proposed_slots:
            failures.append("expected proposed slots")
    elif scenario.startswith("s14_") and step == "confirmation":
        if state.status != TerminalStatus.BOOKED:
            failures.append("expected booked status")
        if state.event is None:
            failures.append("expected calendar event")
        else:
            if not state.event.location:
                failures.append("expected in-person location to be specified")
            if state.event.meet_link:
                failures.append("expected no Google Meet link for in-person meeting")
    elif scenario.startswith("s17_") and step == "initial":
        if state.status != TerminalStatus.WAITING_ON_TARGET:
            failures.append("expected waiting status")
        if "email" not in state.last_message.lower():
            failures.append("expected agent to ask for candidate's email")
    elif scenario.startswith("s17_") and step == "confirmation":
        if state.status != TerminalStatus.BOOKED:
            failures.append("expected booked status after email collection")
        if state.event is None:
            failures.append("expected calendar event")
        else:
            if "siddharth.bose@example.com" not in state.event.attendees:
                failures.append("expected candidate email in attendees list")
    else:
        if state.status not in {TerminalStatus.BOOKED, TerminalStatus.WAITING_ON_TARGET, TerminalStatus.NO_MEETING_INTENT}:
            failures.append(f"unexpected status {state.status.value}")
    return failures


def _run_eval_case(
    store: FixtureStore,
    scenario: str,
    calendar: str,
    use_llm: bool,
    current_date: str,
) -> list[dict[str, Any]]:
    conversation_id = resolve_conversation_id(store, scenario)
    state = store.make_state(conversation_id, current_date=current_date)
    agent = build_agent(state, calendar, use_llm=use_llm)
    rows: list[dict[str, Any]] = []

    try:
        state = agent.step(state)
        failures = _eval_checks(state, conversation_id, "initial", require_llm=use_llm)
        rows.append(_make_eval_row(state, conversation_id, "initial", failures))
    except RuntimeError as exc:
        failures = [f"expected LLM extraction but got error: {exc}"]
        rows.append(_make_eval_row(state, conversation_id, "initial", failures))
        return rows

    if conversation_id.startswith("s05_"):
        state.add_target_message("Confirmed.")
        try:
            state = agent.step(state)
            failures = _eval_checks(state, conversation_id, "confirmation", require_llm=use_llm)
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))
        except RuntimeError as exc:
            failures = [f"expected LLM extraction but got error: {exc}"]
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))

    elif conversation_id.startswith("s06_"):
        state.add_target_message("yes works")
        try:
            state = agent.step(state)
            failures = _eval_checks(state, conversation_id, "ambiguous_acceptance", require_llm=use_llm)
            rows.append(_make_eval_row(state, conversation_id, "ambiguous_acceptance", failures))
        except RuntimeError as exc:
            failures = [f"expected LLM extraction but got error: {exc}"]
            rows.append(_make_eval_row(state, conversation_id, "ambiguous_acceptance", failures))

    elif conversation_id.startswith("s08_"):
        state.add_target_message("18:30 works")
        try:
            state = agent.step(state)
            failures = _eval_checks(state, conversation_id, "confirmation", require_llm=use_llm)
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))
        except RuntimeError as exc:
            failures = [f"expected LLM extraction but got error: {exc}"]
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))

    elif conversation_id.startswith("s14_"):
        state.add_target_message("Riverside Cafe at 17:30 sounds great. Drop a calendar invite.")
        try:
            state = agent.step(state)
            failures = _eval_checks(state, conversation_id, "confirmation", require_llm=use_llm)
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))
        except RuntimeError as exc:
            failures = [f"expected LLM extraction but got error: {exc}"]
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))

    elif conversation_id.startswith("s17_"):
        state.add_target_message("siddharth.bose@example.com")
        try:
            state = agent.step(state)
            failures = _eval_checks(state, conversation_id, "confirmation", require_llm=use_llm)
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))
        except RuntimeError as exc:
            failures = [f"expected LLM extraction but got error: {exc}"]
            rows.append(_make_eval_row(state, conversation_id, "confirmation", failures))

    return rows


def _make_eval_row(state, scenario, step, failures):
    return {
        "scenario": scenario,
        "step": step,
        "status": state.status.value,
        "passed": not failures,
        "failures": failures,
        "last_reason": state.last_reason,
        "last_message": state.last_message,
        "proposed_slots": [f"{s.start.strftime('%a %d %b %I:%M %p')} ({s.label or ''})" for s in state.proposed_slots],
        "confirmed_slot": state.confirmed_slot.start.isoformat() if state.confirmed_slot else None,
        "tool_calls": [f"{t.tool}.{t.action}" for t in state.tool_calls],
        "event": {
            "title": state.event.title,
            "meet_link": state.event.meet_link,
            "location": state.event.location,
        } if state.event else None,
        "edge_case_flags": state.edge_case_flags,
    }


def print_eval_results(rows: list[dict[str, Any]], verbose: bool = False) -> None:
    table = Table(title="Local Eval")
    table.add_column("Scenario")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Result")
    table.add_column("Notes")
    for row in rows:
        table.add_row(
            row["scenario"],
            row["step"],
            row["status"],
            "PASS" if row["passed"] else "[bold red]FAIL[/bold red]",
            "; ".join(row["failures"]) or row["last_reason"],
        )
    console.print(table)
    if verbose:
        for row in rows:
            console.print(f"\n[bold]── {row['scenario']} / {row['step']} ──[/bold]")
            console.print(f"  Status: {row['status']}")
            console.print(f"  Message: {row.get('last_message', 'N/A')}")
            if row.get('proposed_slots'):
                console.print(f"  Proposed: {', '.join(row['proposed_slots'])}")
            if row.get('confirmed_slot'):
                console.print(f"  Confirmed: {row['confirmed_slot']}")
            if row.get('event'):
                e = row['event']
                loc = f"Location: {e['location']}" if e.get('location') else f"Meet: {e.get('meet_link', '')}" 
                console.print(f"  Event: {e['title']} | {loc}")
            if row.get('tool_calls'):
                console.print(f"  Tools: {' → '.join(row['tool_calls'])}")
            if row.get('edge_case_flags'):
                console.print(f"  Flags: {', '.join(row['edge_case_flags'])}")
            if row['failures']:
                console.print(f"  [red]Failures: {'; '.join(row['failures'])}[/red]")


def _save_eval_traces(rows: list[dict[str, Any]], use_llm: bool = False) -> None:
    """Persist eval run traces to eval_runs/ for later review."""
    trace_dir = Path("eval_runs")
    trace_dir.mkdir(exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    mode = "llm" if use_llm else "heuristic"
    filepath = trace_dir / f"eval_{mode}_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump({"timestamp": timestamp, "mode": mode, "results": rows}, f, indent=2)
    console.print(f"[dim]Trace saved: {filepath}[/dim]")


def _save_demo_trace(state: SchedulingState, scenario: str) -> None:
    """Persist a demo run trace for later review."""
    trace_dir = Path("eval_runs")
    trace_dir.mkdir(exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    filepath = trace_dir / f"demo_{scenario}_{timestamp}.json"
    trace = {
        "timestamp": timestamp,
        "scenario": scenario,
        "status": state.status.value,
        "last_message": state.last_message,
        "last_reason": state.last_reason,
        "thread": [{"sender": m.sender, "text": m.text} for m in state.thread],
        "proposed_slots": [s.model_dump(mode="json") for s in state.proposed_slots],
        "rejected_slots": [s.model_dump(mode="json") for s in state.rejected_slots],
        "confirmed_slot": state.confirmed_slot.model_dump(mode="json") if state.confirmed_slot else None,
        "event": state.event.model_dump(mode="json") if state.event else None,
        "tool_calls": [t.model_dump(mode="json") for t in state.tool_calls],
        "edge_case_flags": state.edge_case_flags,
    }
    with open(filepath, "w") as f:
        json.dump(trace, f, indent=2)
    console.print(f"[dim]Trace saved: {filepath}[/dim]")


def print_slots(state: SchedulingState) -> None:
    console.print("[bold]Proposed[/bold]")
    for slot in state.proposed_slots:
        console.print(f"- {slot.start.isoformat()} -> {slot.end.isoformat()} ({slot.label})")
    console.print("[bold]Rejected[/bold]")
    for slot in state.rejected_slots:
        console.print(f"- {slot.start.isoformat()} -> {slot.end.isoformat()} ({slot.label})")
    if state.confirmed_slot:
        console.print(f"[bold]Confirmed[/bold]: {state.confirmed_slot.start.isoformat()} -> {state.confirmed_slot.end.isoformat()}")


@app.command()
def demo(
    actors: Path = typer.Option(None, help="Path to actors JSON"),
    conversations: Path = typer.Option(None, help="Path to conversations JSON"),
    scenario: str | None = typer.Option(None, help="Conversation id or 1-based index"),
    calendar: str = typer.Option("mock", help="mock or google"),
    no_llm: bool = typer.Option(False, help="Disable OpenRouter extraction"),
    current_date: str = typer.Option("2026-05-23", help="Task anchor date"),
) -> None:
    """Run local evaluator harness."""
    load_dotenv()
    store = FixtureStore(actors or default_actors_path(), conversations or default_conversations_path())
    if scenario is None:
        print_conversations(store)
        scenario = typer.prompt("Pick scenario id or number")
    conversation_id = resolve_conversation_id(store, scenario)
    state = store.make_state(conversation_id, current_date=current_date)
    if "demo_mode" not in state.edge_case_flags:
        state.edge_case_flags.append("demo_mode")
    agent = build_agent(state, calendar, use_llm=not no_llm)

    console.print(Panel(f"{state.target.name}\n{state.target.headline}\n{state.objective}", title="Selected Target"))
    print_thread(state)

    while True:
        state = agent.step(state)
        print_summary(state)
        if agent.llm:
            console.print(
                f"[dim]LLM Cost so far: ${agent.llm.total_cost:.5f} "
                f"({agent.llm.total_prompt_tokens} prompt / {agent.llm.total_completion_tokens} completion tokens)[/dim]"
            )
        if state.status in {TerminalStatus.BOOKED, TerminalStatus.BLOCKED, TerminalStatus.NO_MEETING_INTENT}:
            console.print("[bold]Terminal state reached.[/bold]")
            if state.event:
                meet_or_loc = (
                    f"[bold]Location[/bold]: {state.event.location}"
                    if state.event.location
                    else f"[bold]Meet link[/bold]: {state.event.meet_link}"
                )
                console.print(
                    Panel(
                        f"[bold]Event[/bold]: {state.event.title}\n"
                        f"[bold]Time[/bold]: {state.confirmed_slot.start.isoformat()} → {state.confirmed_slot.end.isoformat()}\n"
                        f"{meet_or_loc}\n"
                        f"[bold]Attendees[/bold]: {', '.join(state.event.attendees) or 'none'}",
                        title="📅 Booked Event",
                    )
                )
            if agent.llm:
                console.print(
                    Panel(
                        f"[bold]Total LLM Cost[/bold]: [bold green]${agent.llm.total_cost:.5f}[/bold green]\n"
                        f"[bold]Prompt Tokens[/bold]: {agent.llm.total_prompt_tokens}\n"
                        f"[bold]Completion Tokens[/bold]: {agent.llm.total_completion_tokens}",
                        title="💰 Cost & Token Breakdown",
                    )
                )
            _save_demo_trace(state, conversation_id)
            # Inspection-only loop after terminal state
            while True:
                text = typer.prompt("Inspect (/state /tools /slots /thread /quit)")
                if text == "/quit":
                    break
                if text == "/state":
                    console.print_json(json.dumps(state.model_dump(mode="json"), indent=2))
                elif text == "/tools":
                    console.print_json(json.dumps([c.model_dump(mode="json") for c in state.tool_calls], indent=2))
                elif text == "/slots":
                    print_slots(state)
                elif text == "/thread":
                    print_thread(state)
                else:
                    console.print("[dim]Session ended. Use inspect commands or /quit.[/dim]")
            break
        text = typer.prompt("Target reply (/state /tools /slots /thread /quit)")
        if text == "/quit":
            _save_demo_trace(state, conversation_id)
            break
        if text == "/state":
            console.print_json(json.dumps(state.model_dump(mode="json"), indent=2))
            continue
        if text == "/tools":
            console.print_json(json.dumps([c.model_dump(mode="json") for c in state.tool_calls], indent=2))
            continue
        if text == "/slots":
            print_slots(state)
            continue
        if text == "/thread":
            print_thread(state)
            continue
        state.add_target_message(text)


@app.command()
def scenarios(
    actors: Path = typer.Option(None),
    conversations: Path = typer.Option(None, help="Path to conversations JSON"),
) -> None:
    """List available scenarios."""
    store = FixtureStore(actors or default_actors_path(), conversations or default_conversations_path())
    print_conversations(store)


@app.command("eval")
def eval_command(
    actors: Path = typer.Option(None, help="Path to actors JSON"),
    conversations: Path = typer.Option(None, help="Path to conversations JSON"),
    scenarios: str = typer.Option(DEFAULT_EVAL_SCENARIOS, help="Comma-separated scenario ids or 1-based indexes"),
    calendar: str = typer.Option("mock", help="mock or google"),
    use_llm: bool = typer.Option(False, "--llm", help="Use OpenRouter extraction instead of heuristic-only extraction"),
    current_date: str = typer.Option("2026-05-23", help="Task anchor date"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis per scenario"),
    save_traces: bool = typer.Option(True, help="Save run traces to eval_runs/ directory"),
) -> None:
    """Run deterministic local eval cases without interactive prompts."""
    load_dotenv()
    store = FixtureStore(actors or default_actors_path(), conversations or default_conversations_path())
    requested = [item.strip() for item in scenarios.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for scenario in requested:
        rows.extend(_run_eval_case(store, scenario, calendar, use_llm=use_llm, current_date=current_date))

    if save_traces:
        _save_eval_traces(rows, use_llm=use_llm)

    if json_output:
        console.print_json(json.dumps(rows, indent=2))
    else:
        print_eval_results(rows, verbose=verbose)

    if any(not row["passed"] for row in rows):
        raise typer.Exit(1)


@app.command("inspect-analyzer")
def inspect_analyzer(
    actors: Path = typer.Option(None),
    conversations: Path = typer.Option(None, help="Path to conversations JSON"),
    scenario: str = typer.Option("1", help="Conversation id or 1-based index"),
    no_llm: bool = typer.Option(False, help="Disable OpenRouter extraction"),
    prompt_only: bool = typer.Option(False, help="Print prompt payload without calling LLM"),
    current_date: str = typer.Option("2026-05-23", help="Task anchor date"),
) -> None:
    """Inspect analyzer context and extraction for one scenario."""
    load_dotenv()
    store = FixtureStore(actors or default_actors_path(), conversations or default_conversations_path())
    conversation_id = resolve_conversation_id(store, scenario)
    state = store.make_state(conversation_id, current_date=current_date)
    analyzer = ConversationAnalyzer(use_llm=not no_llm)
    if prompt_only:
        console.print(analyzer._thread_prompt(state))
        return
    result = analyzer.analyze(state)
    console.print_json(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()
