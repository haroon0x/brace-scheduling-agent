# Architecture Notes

## Approach

This is a local scheduling agent with a deterministic state machine. OpenRouter is used for two purposes: (1) extracting structured scheduling facts from natural language, and (2) naturalizing template policy messages into human-sounding responses. The policy layer decides when to propose, wait, block, or book.

## Key Tradeoffs

- CLI harness over web UI: faster to inspect, deterministic, and better for evaluation traces.
- Explicit state over hidden agent memory: proposed slots, rejected slots, confirmed slots, tool calls, and terminal status are visible.
- Calendar adapter interface over direct Google calls in policy: tests use `MockCalendarClient`, demo can use `GoogleCalendarClient`.
- LLM extraction over regex-only parsing: handles conversational language, while date parsing and booking rules remain bounded.
- Response naturalization with validation: LLM rewrites template messages but critical content (Meet links, deck URLs, emails) is validated to never be dropped. Short clarification messages skip naturalization to preserve intent.

## Two-Phase LLM Usage

**Phase 1 — Extraction (analyzer.py):**
The analyzer prompt asks the model to extract scheduling facts ONLY from the target's messages. It never asks the LLM to book, invent links, or infer calendar availability. The extraction output is a structured JSON object validated with Pydantic and merged with deterministic heuristics.

Key improvements:
- Cancellation/reschedule messages correctly set `exact_slot_text` to null
- Time ranges are downgraded from exact slots to windows
- Bare times without day context are rejected
- Target-only extraction: the scheduler's own messages are context, not extraction sources

**Phase 2 — Naturalization (agent.py):**
After the policy decides what to say, the `RESPONSE_SYSTEM` prompt rewrites the template message to match:
- The target's communication style from the thread
- Role-appropriate tone (VC → respectful, founder → peer, engineer → casual)
- Channel-appropriate length (LinkedIn DM → short, email → slightly more structured)
- The conversation's objective and context flags

## Google Calendar

`GoogleCalendarClient` uses OAuth installed-app flow and Google Calendar APIs:

- `freebusy().query(...)` for availability.
- `events().insert(..., conferenceDataVersion=1)` for event creation with Google Meet.
- `sendUpdates="all"` when attendee emails are available.
- Work-hours (09:30-18:00) and weekend filtering in `find_free_slots`.

The same `CalendarClient` interface is used by mock and Google implementations.

## Memory

`SchedulingState` is the durable memory object. It stores:

- full thread
- proposed slots
- rejected slots and rejected text
- discussed windows
- confirmed slot
- event payload
- tool calls
- terminal status
- edge case flags

This prevents repeated slot proposals, enables context-aware naturalization, and makes decisions auditable.

## Date Parsing

`parse_time_text` iterates all regex matches in the text and prefers matches with explicit am/pm suffix. It skips numbers that are part of date contexts (e.g. "23" in "23 May", "27" in "27/05") using the `_is_date_context` helper. This prevents the bug where "Sat 23 May, 5:00 PM" was misinterpreted as 23:00.

`infer_date` uses `include_today=True` for day names, so "Saturday" on a Saturday resolves to today rather than next week.

## Prompt And Context Engineering

The analyzer prompt is intentionally narrow. It receives a structured JSON payload:

- `current_date` and `timezone` for relative-date anchoring
- `latest_target_message` for turn-level focus
- full recent `thread` for context
- `previously_proposed_slots`
- `rejected_slots` and `rejected_texts`
- `discussed_windows`
- target role/category for context

The response naturalization prompt receives:

- target name, headline, company, background category
- conversation objective
- edge case flags
- channel
- recent thread (for style matching)
- the policy's draft message

Inspect analyzer context with:

```bash
uv run scheduler inspect-analyzer --scenario 1 --prompt-only
uv run scheduler inspect-analyzer --scenario 1
```
