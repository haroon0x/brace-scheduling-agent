# Meeting Scheduling Agent

Local agentic scheduling system. It loads a demo dataset of actors and conversations, lets a tester play the target, checks availability, sends channel-specific replies, and books a Google Calendar event with a Meet link after confirmation.

## DEMO

<!-- Add a demo recording here. Run `uv run scheduler demo --calendar mock` and capture the terminal. -->

## Architecture Concept

```mermaid
flowchart TD
    classDef inputNode fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef brainNode fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8;
    classDef policyNode fill:#d1fae5,stroke:#047857,stroke-width:2px,color:#065f46;
    classDef dispatchNode fill:#fef08a,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef handlerNode fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b;
    classDef adapterNode fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#c2410c;
    classDef cloudNode fill:#f1f5f9,stroke:#475569,stroke-width:2px,color:#334155;
    classDef guardNode fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#9d174d;
    classDef outputNode fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#065f46;

    %% Input
    TargetReply[Inbound Target Message]:::inputNode

    %% Cognitive Layer
    subgraph CognitiveLayer ["cognitive layer — analyzer.py"]
        LLM[LLM — structured intent extraction]:::cloudNode
        Analyzer[Conversation Analyzer]:::brainNode
        LLM -->|TargetIntent + AnalysisResult| Analyzer
    end

    %% Policy Engine
    subgraph PolicyEngine ["policy engine — policy.py"]
        PreProcess[pre_process — state recovery, slot tracking, email extraction]:::policyNode
        Dispatcher{TargetIntent dispatch}:::dispatchNode

        ConfirmH["_handle_confirm"]:::handlerNode
        CounterH["_handle_counter"]:::handlerNode
        DelegateH["_handle_delegate"]:::handlerNode
        WindowH["_handle_request_window"]:::handlerNode
        RejectH["_handle_reject"]:::handlerNode
        CloseH["_handle_close"]:::handlerNode

        SelfClashGuard["self-clash guard\nmatches confirmed_slot + event?\n→ short-circuit to success"]:::guardNode
    end

    %% Adapters
    subgraph ExecutionLayer ["adapters & delivery"]
        Calendar[Calendar Client\ndynamic work-hours bypass]:::adapterNode
        Messenger[Channel Messenger\nemail / linkedin]:::adapterNode
    end

    %% Output
    OutboundMsg[Outbound Message Sent]:::outputNode

    %% Flow
    TargetReply -->|raw text| Analyzer
    Analyzer -->|cleaned intent + state| PreProcess
    PreProcess --> Dispatcher

    Dispatcher -->|CONFIRM_SLOT| ConfirmH
    Dispatcher -->|COUNTER_PROPOSE| CounterH
    Dispatcher -->|DELEGATE_CHOICE| DelegateH
    Dispatcher -->|REQUEST_WINDOW| WindowH
    Dispatcher -->|REJECT| RejectH
    Dispatcher -->|CLOSE| CloseH

    ConfirmH --> SelfClashGuard
    SelfClashGuard -->|no clash — proceed| Calendar
    SelfClashGuard -->|already confirmed — skip| Messenger

    CounterH & DelegateH & WindowH <-->|freebusy + slot queries| Calendar
    RejectH & CloseH --> Messenger
    Calendar --> Messenger
    Messenger --> OutboundMsg
```

### Architectural & Security Boundaries

```text
Local execution space (Local Execution Context):
- Never shares calendar tokens or active credentials with the LLM APIs.
- Retains complete state control in SchedulingState memory.
- Enforces strict work-hours and weekend constraints locally in code.

External API interactions (External API Services):
- OpenRouter/Gemini API: Receives ONLY conversational context; never decides on booking.
- Google Calendar API: Receives ONLY finalized, locally validated event requests.
```

## Quick Start & Setup

### 1. Install Dependencies
Make sure you have [uv](https://astral.sh) installed, then sync the dependencies:
```bash
# Sync all dependencies including dev tools
uv sync --extra dev
```

### 2. Configure Environment & Credentials
You can configure your environment variables in **two ways**:
1. **Using a `.env` file (Recommended)**: Create a `.env` file in the root of the project. The application automatically loads variables from `.env` using `python-dotenv` on startup.
2. **Using shell exports**: Run `export KEY=VALUE` directly in your terminal.

#### API Keys Configuration (Add to `.env` or export):
```bash
# If using Google Gemini (Recommended):
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# If using OpenRouter:
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-5-mini
```

#### Google Calendar Credentials:
1. Create a Google Cloud OAuth client for a Desktop app and download the client JSON.
2. Save it as `credentials.json` in the root directory of this repo, or specify the path in your `.env` or shell:
   ```bash
   GOOGLE_CALENDAR_CREDENTIALS=/path/to/credentials.json
   ```

### 3. Demo Dataset

The repo ships a synthetic dataset (`fixtures/actors.json`, `fixtures/conversations.json`) with fictional actors so everything runs out of the box. To run the demo against your own dataset:

```bash
export SCHEDULER_ACTORS=/path/to/actors.json
export SCHEDULER_CONVERSATIONS=/path/to/conversations.json
# or pass --actors/--conversations to each command
```

---

## Running the Interactive Demo

### Run with Google Calendar (Real Booking)
This mode reads free/busy slots from your primary calendar, checks availability dynamically, and blocks real events with Google Meet links:
```bash
uv run scheduler demo --calendar google
```
*Note: On the first run, a Google OAuth consent screen will open in your browser. Log in to authorize the app. The access token is saved locally as `token.json` for subsequent runs.*

### Run with Mock Calendar (Deterministic Simulation)
If you do not want to connect to a live Google calendar, use the mock calendar client:
```bash
uv run scheduler demo --calendar mock
```

---

## LLM Provider Selection
By default, the agent automatically detects whether `GEMINI_API_KEY` or `OPENROUTER_API_KEY` is set. If both are available, you can explicitly force a provider:
```bash
LLM_PROVIDER=gemini       # Force Google Gemini
# or
LLM_PROVIDER=openrouter   # Force OpenRouter
```

Analyzer debugging:

```bash
uv run scheduler inspect-analyzer --scenario 1 --prompt-only
uv run scheduler inspect-analyzer --scenario 1
```

## Google Calendar Setup

1. Create a Google Cloud OAuth client for a Desktop app.
2. Download the client JSON.
3. Save it as `credentials.json` in the repo root, or set:

```bash
export GOOGLE_CALENDAR_CREDENTIALS=/path/to/oauth-client.json
export GOOGLE_CALENDAR_TOKEN=token.json
```

4. Run:

```bash
uv run scheduler demo --calendar google
```

On first run, a browser OAuth flow opens. The app stores `token.json`, reads free/busy from the primary calendar, creates events, and requests Google Meet conferencing.

## Harness Commands

- `/state`: full scheduling state JSON
- `/tools`: tool calls
- `/slots`: proposed/rejected/confirmed slots
- `/thread`: current thread
- `/quit`: exit

After booking, the harness shows the event panel and enters an inspection-only mode.

## Tests

```bash
uv run pytest
```

44 tests covering date parsing, analyzer extraction/repair, policy decisions, calendar filtering, eval checks, and agent integration.

## Evaluation Smoke

```bash
bash scripts/run_evaluation.sh
```

Run the local evaluation scenarios with verbose outputs:
```bash
uv run scheduler eval --verbose
```

### Run Trace Logging
Every evaluation run or interactive demo session automatically saves full trace logs (including the final state, proposed/confirmed slots, exact message text, tool calls, and edge case flags) as JSON to the `eval_runs/` directory for trace analysis and debugging.
- Eval trace filename pattern: `eval_runs/eval_{mode}_{timestamp}.json`
- Demo trace filename pattern: `eval_runs/demo_{scenario}_{timestamp}.json`

## What Is Real Versus Mocked

- **Real**: Google Calendar free/busy, event creation, and Google Meet link when using `--calendar google`.
- **Mocked**: LinkedIn and email message delivery. The agent records and renders channel messages locally for inspection.
- **Mocked in tests**: calendar availability and Meet link, to keep tests deterministic.

## Notes

- All demo data in `fixtures/` is synthetic and fictional.
- The demo harness anchors scenario dates to `2026-05-23` (override with `--current-date`).
- Message adapters and the default demo calendar are intentionally local mocks so the project runs offline; see the architecture notes in `docs/`.
