# Architecture v2 — Single Source of Truth

## The Problem

Slot resolution is split across two components, with the wrong one doing the heavy lifting:

```
Analyzer (has LLM, sees full thread, has context)
  → produces AnalysisResult with STRINGS (exact_slot_text, accepted_previous_slot)
  → Policy then re-parses those strings against proposals, scans thread, reconstructs meaning

Result: two paths (LLM/heuristic) drift apart because heuristic doesn't recover
slots from thread history, doesn't resolve bare times, and has weaker acceptance
detection. Policy duplicates parsing logic that belongs in the analyzer.
```

## The Fix

**Move all slot resolution INTO the analyzer.** Policy shrinks to validation + execution.

```
Analyzer (has LLM, sees full thread, has context)
  → resolves everything to concrete Slot objects
  → produces AnalysisResult with target_slot: Slot | None, resolved_windows: list[ResolvedWindow]
  → heuristic path does same resolution using dates.py functions (not shadow, peer)

Policy (no text parsing, no thread scanning)
  → receives fully resolved slots
  → validates against calendar
  → books or proposes alternatives
```

## Before vs After

### AnalysisResult changes

| Before (strings) | After (resolved objects) |
|---|---|
| `exact_slot_text: str\|None` | `target_slot: Slot\|None` — fully resolved |
| `accepted_previous_slot: bool` | Keep for LLM compat, policy ignores |

`exact_slot_text` stays in model for LLM extraction. Analyzer resolves it to `target_slot`
in post-processing. Policy only reads `target_slot`.

### Policy simplification

| Method removed from policy | Who does it now |
|---|---|
| `_resolve_confirmed_slot()` (parses text, matches proposals) | Analyzer `_resolve_target_slot()` |
| `_counter_slot_from_previous()` (parses "make it X instead") | Analyzer `_resolve_target_slot()` step 2 |
| `_match_proposed_slot_by_time()` (bare time → proposal match) | Analyzer `_resolve_target_slot()` step 3 |
| `_windows()` thread text scanning for illness/reschedule | Analyzer `_resolve_result_windows()` |

Policy keeps: `_book_slot()`, `_propose_new_slots()`, `_find_free_slots()`, `_default_window()`,
`_next_business_day()`, message formatting. ~200 lines saved.

### Analyzer gains

1. `_resolve_target_slot(result, state)` — resolves exact_slot_text to concrete Slot:
   - Step 1: Direct parse via `parse_slot()` (handles "Friday 3:30")
   - Step 2: Counter from context — "4:00" + previous proposal's date
   - Step 3: Bare time match — "9:30" matching proposal's hour:minute

2. `_recover_slots_from_thread(state)` — scans the scheduler's (user/agent) messages for parseable slots.
   Returns `recovered_proposed_slots` and `recovered_rejected_slots`.
   This was LLM-only. Now heuristic has it too.

3. `_resolve_result_windows()` gains:
   - must_propose_next_day handling (moved from policy._windows)
   - illness/reschedule thread scanning (moved from policy._windows)

4. Improved heuristic acceptance regex:
   - Added: "sounds great", "looks great", "perfect", "love it", "book it", "drop a calendar invite"

5. Strengthened `_infer_intent_from_signals()`:
   - target_slot matching proposals → CONFIRM_SLOT (not COUNTER_PROPOSE)
   - accepted_previous_slot + proposals → CONFIRM_SLOT (handles ambiguous "yes works")

### Data flow (one turn)

```
Target message → state.thread
    ↓
Analyzer.analyze(state):
  1. LLM or heuristic extraction → raw AnalysisResult
  2. _resolve_target_slot() → concrete Slot from exact_slot_text
  3. _resolve_result_windows() → concrete ResolvedWindow objects
  4. _recover_slots_from_thread() → recovered proposed/rejected
  5. _infer_intent_from_signals() → TargetIntent (heuristic only)
    ↓
Policy.decide(state, analysis, calendar):
  1. _pre_process() → sync recovered slots, extract email, track rejections
  2. Dispatch intent handler:
     - CONFIRM_SLOT: if target_slot → _book_slot, else clarify/auto-book
     - COUNTER_PROPOSE: if target_slot → _book_slot, else propose/clarify
     - DELEGATE_CHOICE: auto-book first proposal
     - REQUEST_WINDOW: propose from resolved_windows
     - REJECT: propose alternatives
     - CLOSE: terminal message
    ↓
Agent.naturalize() → send via messenger
```

## Eval failures fixed by this

| Failure | Root cause | How this fixes it |
|---|---|---|
| s14 "sounds great" not booking | Acceptance regex missing patterns | Improved heuristic regex + target_slot resolution |
| s17 email gating not triggering | No slot recovery in heuristic | _recover_slots_from_thread() finds the scheduler's proposals |
| s08 "18:30 works" not booking | No bare-time match in _handle_confirm | Analyzer resolves bare time to target_slot |

## What doesn't change

- Calendar adapters (interface, mock, Google)
- Message adapters
- LLM client
- Fixtures
- CLI harness and eval infrastructure
- SchedulingState model
- LLM prompt (exact_slot_text still produced, just post-processed)
- `_windows()` window-to-TimeWindow conversion (kept in policy, just simplified)
