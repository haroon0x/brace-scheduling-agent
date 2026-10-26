from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .models import ActorRef, Channel, Message, SchedulingState


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def actor_ref(raw: dict, background_category: str | None = None) -> ActorRef:
    profile = raw.get("profile", {})
    contacts = raw.get("contacts", {})
    current_role = raw.get("current_role") or {}
    return ActorRef(
        actor_id=raw["actor_id"],
        name=profile.get("name") or raw.get("target_actor_name") or "Unknown",
        headline=profile.get("headline"),
        email=contacts.get("email"),
        linkedin=contacts.get("linkedin"),
        background_category=raw.get("background_category") or background_category,
        company=current_role.get("company_name"),
        raw=raw,
    )


class FixtureStore:
    def __init__(self, actors_path: str | Path, conversations_path: str | Path):
        self.actors_doc = load_json(actors_path)
        self.conversations_doc = load_json(conversations_path)
        self.user_actor = actor_ref(self.actors_doc["user_actor"])
        self.actors = {
            actor["actor_id"]: actor_ref(actor)
            for actor in self.actors_doc.get("actors", [])
        }
        self.conversations = self.conversations_doc.get("conversations", [])

    def list_conversations(self) -> list[dict]:
        return self.conversations

    def make_state(self, conversation_id: str, current_date: str = "2026-05-23") -> SchedulingState:
        conversation = next(
            item for item in self.conversations if item["conversation_id"] == conversation_id
        )
        target = self.actors.get(conversation["target_actor_id"])
        if target is None:
            target = ActorRef(
                actor_id=conversation["target_actor_id"],
                name=conversation["target_actor_name"],
                headline=conversation.get("target_headline"),
                background_category=conversation.get("background_category"),
                company=conversation.get("target_current_company"),
            )
        elif target.background_category is None:
            target.background_category = conversation.get("background_category")

        messages = [
            Message(
                timestamp=item.get("timestamp"),
                sender=item["sender"],
                actor_id=item.get("actor_id"),
                text=item["text"],
            )
            for item in conversation.get("history", [])
        ]
        effective_date = current_date or conversation.get("current_date") or "2026-05-23"
        return SchedulingState(
            session_id=str(uuid4()),
            conversation_id=conversation_id,
            current_date=effective_date,
            timezone=conversation.get("user_timezone") or "Asia/Kolkata",
            channel=Channel(conversation["channel"]),
            user=self.user_actor,
            target=target,
            objective=conversation.get("user_objective", ""),
            edge_case_flags=conversation.get("edge_case_flags", []),
            thread=messages,
        )
