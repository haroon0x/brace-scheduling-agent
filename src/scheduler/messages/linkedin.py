from __future__ import annotations

from scheduler.messages.base import MessageClient
from scheduler.models import Channel, ToolCall


class LinkedInMessageClient(MessageClient):
    channel = Channel.LINKEDIN_DM

    def send(self, text: str, recipient: str) -> ToolCall:
        return ToolCall(
            tool="message.linkedin",
            action="send",
            input={"recipient": recipient, "text": text},
            output={"mocked": True, "channel": self.channel.value},
        )
