from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TerminalStatus(str, Enum):
    NEEDS_USER_REPLY = "needs_user_reply"
    WAITING_ON_TARGET = "waiting_on_target"
    BOOKED = "booked"
    BLOCKED = "blocked"
    NO_MEETING_INTENT = "no_meeting_intent"


class TargetIntent(str, Enum):
    """Primary classification of what the target wants.

    The LLM must pick exactly one.  Modifiers (e.g. "delegate") ride
    along in AnalysisResult.intent_modifiers.
    """
    CONFIRM_SLOT = "confirm_slot"        # "Tuesday works" / "confirmed" / "let's do 3 PM"
    COUNTER_PROPOSE = "counter_propose"  # "make it 4 instead" / "can we do Thursday?"
    DELEGATE_CHOICE = "delegate_choice"  # "you pick" / "any of those" / "just book one"
    REQUEST_WINDOW = "request_window"    # "next week any time" / "Thursday afternoon"
    REJECT = "reject"                    # "can't make this week" / "busy"
    CLOSE = "close"                      # "not interested" / "pass"


class Channel(str, Enum):
    LINKEDIN_DM = "LINKEDIN_DM"
    EMAIL = "EMAIL"


class Message(BaseModel):
    timestamp: str | None = None
    sender: Literal["user", "target", "agent", "system"]
    actor_id: str | None = None
    text: str


class ActorRef(BaseModel):
    actor_id: str
    name: str
    headline: str | None = None
    email: str | None = None
    linkedin: str | None = None
    background_category: str | None = None
    company: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TimeWindow(BaseModel):
    start: datetime
    end: datetime
    label: str
    timezone: str = "Asia/Kolkata"


class Slot(BaseModel):
    start: datetime
    end: datetime
    timezone: str = "Asia/Kolkata"
    label: str | None = None

    def key(self) -> str:
        return f"{self.start.isoformat()}::{self.end.isoformat()}::{self.timezone}"


class ToolCall(BaseModel):
    tool: str
    action: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class EventPayload(BaseModel):
    event_id: str
    title: str
    attendees: list[str] = Field(default_factory=list)
    start: datetime
    end: datetime
    timezone: str
    meet_link: str = ""
    html_link: str | None = None
    location: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ResolvedWindow(BaseModel):
    start: str
    end: str
    label: str


class AnalysisResult(BaseModel):
    intent: TargetIntent = TargetIntent.REQUEST_WINDOW
    intent_modifiers: list[str] = Field(default_factory=list)
    meeting_intent: bool = True
    no_meeting_intent: bool = False
    exact_slot_text: str | None = None
    target_slot: Slot | None = None
    accepted_previous_slot: bool = False
    rejected_slot_texts: list[str] = Field(default_factory=list)
    requested_windows: list[str] = Field(default_factory=list)
    resolved_windows: list[ResolvedWindow] = Field(default_factory=list)
    recovered_proposed_slots: list[ResolvedWindow] = Field(default_factory=list)
    recovered_rejected_slots: list[ResolvedWindow] = Field(default_factory=list)
    wants_next_week: bool = False
    asks_for_deck: bool = False
    asks_for_email: bool = False
    wants_short_call: bool = False
    timezone_hint: str | None = None
    confidence: float = 0.5
    notes: str = ""
    reasoning: str = ""
    suggested_action: str | None = None


class SchedulingState(BaseModel):
    session_id: str
    conversation_id: str
    current_date: str = "2026-05-23"
    timezone: str = "Asia/Kolkata"
    channel: Channel
    user: ActorRef
    target: ActorRef
    objective: str
    edge_case_flags: list[str] = Field(default_factory=list)
    thread: list[Message] = Field(default_factory=list)
    proposed_slots: list[Slot] = Field(default_factory=list)
    rejected_slots: list[Slot] = Field(default_factory=list)
    rejected_texts: list[str] = Field(default_factory=list)
    discussed_windows: list[str] = Field(default_factory=list)
    confirmed_slot: Slot | None = None
    status: TerminalStatus = TerminalStatus.NEEDS_USER_REPLY
    event: EventPayload | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    last_reason: str = ""
    last_message: str = ""

    def remember_tool(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    def add_agent_message(self, text: str) -> None:
        self.thread.append(Message(sender="agent", actor_id=self.user.actor_id, text=text))
        self.last_message = text

    def add_target_message(self, text: str) -> None:
        self.thread.append(Message(sender="target", actor_id=self.target.actor_id, text=text))

    def proposed_keys(self) -> set[str]:
        return {slot.key() for slot in self.proposed_slots}

    def rejected_keys(self) -> set[str]:
        return {slot.key() for slot in self.rejected_slots}
