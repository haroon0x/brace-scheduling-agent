from __future__ import annotations

from .analyzer import ConversationAnalyzer
from .calendar.base import CalendarClient
from .llm import OpenRouterClient
from .messages.base import MessageClient
from .models import SchedulingState, TerminalStatus, ToolCall
from .policy import DECK_LINK, SchedulingPolicy

RESPONSE_SYSTEM = """\
You are {persona}. Rewrite the draft scheduling message \
so it reads like you actually typed it — concise, warm, and human.

Hard rules:
- Keep EVERY time/date, link, and email address from the draft VERBATIM. Never drop or paraphrase factual content.
- Never contradict the drafted message. If the draft states that a slot is not possible or proposes specific alternative slots, do not imply that the target's original/previously rejected slot is still available or can still be booked.
- 1–3 sentences max. No filler, no padding, no exclamation-heavy enthusiasm.
- Mirror the target's energy from the thread. If they write lowercase and casual, you do too. If they are professional, match that.
- Tone by target role:
    VC/investor → respectful, unhurried, never pushy or salesy.
    Founder → peer-to-peer, direct, friendly.
    Engineer/candidate → casual, warm, approachable.
- Channel:
    LINKEDIN_DM → very short, chat-style, no subject lines or signatures.
    EMAIL → one notch more structured, still brief, no "Dear" or "Best regards".
- If proposing multiple time slots, present them on separate lines or with clear separators — never a run-on.
- Never open with "Hey there!", "Hi there!", or other generic chatbot greetings.
- Use the target's first name only when it flows naturally. Respond based on the context of the previous messages.
- Output ONLY the final message text. No quotes, labels, markdown, or explanation.
"""


class SchedulingAgent:
    def __init__(
        self,
        analyzer: ConversationAnalyzer,
        policy: SchedulingPolicy,
        calendar: CalendarClient,
        messenger: MessageClient,
        llm: OpenRouterClient | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.policy = policy
        self.calendar = calendar
        self.messenger = messenger
        self.llm = llm

    def step(self, state: SchedulingState) -> SchedulingState:
        analysis = self.analyzer.analyze(state)
        state.remember_tool(
            ToolCall(
                tool="analyzer",
                action="extract",
                input={
                    "latest_message": state.thread[-1].text if state.thread else "",
                    "use_llm": self.analyzer.use_llm,
                },
                output=analysis.model_dump(mode="json"),
                meta={
                    "source": self.analyzer.last_source,
                    "error": self.analyzer.last_error,
                    "model": getattr(self.analyzer.llm, "model", None),
                },
            )
        )
        decision = self.policy.decide(state, analysis, self.calendar)
        message = self._naturalize(state, decision)
        state.status = decision.status
        state.last_reason = decision.reason
        state.add_agent_message(message)
        call = self.messenger.send(message, state.target.actor_id)
        state.remember_tool(call)
        return state

    # ------------------------------------------------------------------
    # Response naturalization
    # ------------------------------------------------------------------

    def _naturalize(self, state, decision):
        if not self.llm or not self.llm.enabled:
            return decision.message
        # Short clarification/terminal messages are already human-sounding; skip LLM
        if len(decision.message) < 60 and decision.status != TerminalStatus.BOOKED:
            return decision.message
        try:
            prompt = self._response_prompt(state, decision)
            system = RESPONSE_SYSTEM.format(persona=self._persona(state))
            natural = self.llm.generate_text(system, prompt, temperature=0.4)
            if not natural:
                return decision.message
            return self._validate_response(natural, state, decision)
        except Exception:  # Rate limits, network errors, etc.
            return decision.message

    @staticmethod
    def _persona(state) -> str:
        persona = state.user.name or "the scheduling assistant"
        if state.user.headline:
            persona = f"{persona}, {state.user.headline}"
        return persona

    def _response_prompt(self, state, decision):
        thread_lines = []
        for msg in state.thread[-8:]:
            who = state.user.name if msg.sender in ("user", "agent") else state.target.name
            thread_lines.append(f"{who}: {msg.text}")
        first_name = state.target.name.split()[0] if state.target.name else "there"
        flags = ", ".join(state.edge_case_flags[:4]) if state.edge_case_flags else "none"
        return (
            f"Target: {state.target.name}"
            f"{', ' + state.target.headline if state.target.headline else ''}"
            f" ({state.target.background_category or 'unknown'})"
            f"{', ' + state.target.company if state.target.company else ''}\n"
            f"Target first name: {first_name}\n"
            f"Channel: {state.channel.value}\n"
            f"Objective: {state.objective}\n"
            f"Context flags: {flags}\n"
            f"Action: {decision.reason}\n\n"
            f"Recent thread:\n" + "\n".join(thread_lines) + "\n\n"
            f"Draft message to rewrite:\n{decision.message}"
        )

    def _validate_response(self, natural, state, decision):
        """Ensure the LLM didn't drop critical factual content or introduce spacing errors."""
        if decision.status == TerminalStatus.BOOKED and state.event:
            if state.event.meet_link and state.event.meet_link not in natural:
                natural = natural.rstrip() + f"\n{state.event.meet_link}"
        if DECK_LINK in decision.message and DECK_LINK not in natural:
            natural = f"{DECK_LINK}\n\n{natural}"
        if state.user.email and state.user.email in decision.message and state.user.email not in natural:
            natural = natural.rstrip() + f"\nMy email: {state.user.email}"
        return self._fix_spacing(natural)

    @staticmethod
    def _fix_spacing(text: str) -> str:
        """Repair common LLM spacing errors like 'Bookedfor', 'Mon25 May', 'meet atThe'."""
        import re
        # Fix lowercase-followed-by-uppercase without space: "atThe", "Bookedfor" -> won't catch
        # Actually fix: letter immediately followed by capital letter with no space
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Fix patterns like "Mon25" (word char then digit without space, when preceded by a day/month name)
        text = re.sub(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d)', r'\1 \2', text)
        # Fix "Bookedfor", "booked for" with missing space
        text = re.sub(r'(?i)\b(booked)(for)\b', r'\1 \2', text)
        # Fix "Ihave", "ihave" with missing space
        text = re.sub(r'(?i)\b([Ii])(have)\b', r'\1 \2', text)
        return text
