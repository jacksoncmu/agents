"""Abstract LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent.types import Message, ToolCall


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send messages to the model and return the response."""
        ...
