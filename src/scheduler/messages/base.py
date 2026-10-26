from __future__ import annotations

from abc import ABC, abstractmethod

from scheduler.models import Channel, ToolCall


class MessageClient(ABC):
    channel: Channel

    @abstractmethod
    def send(self, text: str, recipient: str) -> ToolCall:
        raise NotImplementedError
