"""Mock LLM provider for testing.

Responses are driven by a script: a list of LLMResponse objects returned in
order, one per call.  If the script is exhausted the provider raises
RuntimeError so tests fail loudly instead of silently looping.
"""
from __future__ import annotations

from typing import Any

from agent.llm.base import LLMProvider, LLMResponse
from agent.types import Message


class MockProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        if self._call_index >= len(self._responses):
            raise RuntimeError(
                f"MockProvider exhausted after {self._call_index} call(s). "
                "Add more responses to the script."
            )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

    @property
    def call_count(self) -> int:
        return self._call_index
