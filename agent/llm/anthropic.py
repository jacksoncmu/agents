"""Anthropic provider — adapter + LLMProvider implementation.

The module is split into two classes:

AnthropicAdapter
    Pure data-transformation: internal Message list → Anthropic wire format
    and Anthropic response → LLMResponse.  No network calls; fully testable
    without an API key.

AnthropicProvider
    Thin wrapper that calls the Anthropic SDK using the adapter.  Lazy-imports
    the ``anthropic`` package so the rest of the runtime stays importable even
    when the SDK is not installed.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agent.llm.adapter import BaseAdapter
from agent.llm.base import LLMProvider, LLMResponse
from agent.llm.config import ModelConfig
from agent.types import Message, MessageRole, ToolCall

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter — pure translation, no I/O
# ---------------------------------------------------------------------------

class AnthropicAdapter(BaseAdapter):
    """Translates between the internal format and the Anthropic Messages API."""

    # -- outbound: internal → Anthropic ------------------------------------

    def to_provider_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Mapping rules:

        Internal role          Anthropic role / content shape
        ─────────────────────────────────────────────────────
        user                   {"role":"user",      "content": "<str>"}
        assistant (no tools)   {"role":"assistant", "content": "<str>"}
        assistant (tools)      {"role":"assistant", "content": [text?, tool_use…]}
        tool_result            {"role":"user",      "content": [tool_result…]}
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.append(self._convert_message(msg))
        return result

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        if msg.role == MessageRole.user:
            return {"role": "user", "content": msg.content}

        if msg.role == MessageRole.assistant:
            if not msg.tool_calls:
                return {"role": "assistant", "content": msg.content}
            # Mixed content: optional leading text then tool_use blocks
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            return {"role": "assistant", "content": content}

        if msg.role == MessageRole.tool_result:
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                    **({"is_error": True} if tr.error else {}),
                }
                for tr in msg.tool_results
            ]
            return {"role": "user", "content": content}

        raise ValueError(f"Unknown message role: {msg.role!r}")

    def to_provider_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Internal tool schemas use the OpenAI-style ``parameters`` key.
        Anthropic uses ``input_schema`` with identical JSON Schema content.
        """
        converted = []
        for t in tools:
            converted.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["parameters"],
            })
        return converted

    # -- inbound: Anthropic → internal ------------------------------------

    def from_provider_response(self, raw: Any) -> LLMResponse:
        """
        Parse an ``anthropic.types.Message`` object.

        ``raw.content`` is a list of content blocks:
          {"type": "text",     "text": "..."}
          {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            block_type = getattr(block, "type", None) or block.get("type")

            if block_type == "text":
                text = getattr(block, "text", None) or block.get("text", "")
                text_parts.append(text)

            elif block_type == "tool_use":
                tc_id   = getattr(block, "id",    None) or block.get("id",    "")
                tc_name = getattr(block, "name",  None) or block.get("name",  "")
                tc_input = getattr(block, "input", None)
                if tc_input is None:
                    tc_input = block.get("input", {})
                # input may arrive as a dict or a JSON string
                if isinstance(tc_input, str):
                    tc_input = json.loads(tc_input)
                tool_calls.append(ToolCall(id=tc_id, name=tc_name, arguments=tc_input))

        return LLMResponse(content=" ".join(text_parts).strip(), tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Provider — wraps Anthropic SDK
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    LLMProvider implementation backed by the Anthropic Messages API.

    Lazy-imports ``anthropic`` so importing this module doesn't fail when the
    package isn't installed.

    Provider-specific config (from ``ModelConfig.extra``)
    ──────────────────────────────────────────────────────
    system      str   System prompt text.
    thinking    dict  Extended thinking config, e.g.
                      ``{"type": "enabled", "budget_tokens": 5000}``.
                      When present, temperature is forced to 1 (API requirement).
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._adapter = AnthropicAdapter()
        self._client: Any = None  # initialised lazily

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "The 'anthropic' package is required to use AnthropicProvider. "
                    "Install it with: pip install anthropic"
                ) from exc
            api_key = self.config.resolved_api_key("ANTHROPIC_API_KEY")
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        client = self._get_client()

        provider_messages = self._adapter.to_provider_messages(messages)
        provider_tools    = self._adapter.to_provider_tools(tools) if tools else []

        extra = self.config.extra
        thinking_cfg = extra.get("thinking")

        call_kwargs: dict[str, Any] = {
            "model":      self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages":   provider_messages,
        }
        if provider_tools:
            call_kwargs["tools"] = provider_tools
        if system := extra.get("system"):
            call_kwargs["system"] = system
        if thinking_cfg:
            call_kwargs["thinking"] = thinking_cfg
            # Extended thinking requires temperature=1
            call_kwargs["temperature"] = 1
        elif self.config.temperature is not None:
            call_kwargs["temperature"] = self.config.temperature

        log.debug(
            "AnthropicProvider.complete model=%s messages=%d tools=%d",
            self.config.model,
            len(provider_messages),
            len(provider_tools),
        )

        raw = client.messages.create(**call_kwargs)
        return self._adapter.from_provider_response(raw)
