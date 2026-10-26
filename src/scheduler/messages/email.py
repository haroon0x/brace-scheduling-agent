from __future__ import annotations

from scheduler.messages.base import MessageClient
from scheduler.models import Channel, ToolCall


class EmailMessageClient(MessageClient):
    channel = Channel.EMAIL

    def send(self, text: str, recipient: str) -> ToolCall:
        return ToolCall(
            tool="message.email",
            action="send",
            input={"recipient": recipient, "text": text},
            output={"mocked": True, "channel": self.channel.value},
        )
