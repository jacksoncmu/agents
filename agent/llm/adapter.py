"""Base adapter interface for provider message translation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent.llm.base import LLMResponse
from agent.types import Message


class BaseAdapter(ABC):
    """
    Translates between the internal message format and a provider's wire format.

    Each provider subclass implements these three methods.  The adapter itself
    is stateless — it only transforms data.
    """

    @abstractmethod
    def to_provider_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal Message list to the provider's message array."""
        ...

    @abstractmethod
    def to_provider_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert internal tool schemas (OpenAI-style ``parameters`` key) to
        whatever format the provider expects.
        """
        ...

    @abstractmethod
    def from_provider_response(self, raw: Any) -> LLMResponse:
        """Normalise a raw provider response object into an ``LLMResponse``."""
        ...
