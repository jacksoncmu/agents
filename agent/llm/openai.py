"""OpenAI provider — adapter + LLMProvider implementation.

OpenAIAdapter
    Pure data-transformation: internal Message list → OpenAI Chat Completions
    wire format and OpenAI response → LLMResponse.  No network calls; fully
    testable without an API key.

OpenAIProvider
    Thin wrapper that calls the OpenAI SDK using the adapter.  Lazy-imports
    the ``openai`` package so the rest of the runtime stays importable even
    when the SDK is not installed.

Key differences from the Anthropic format
──────────────────────────────────────────
- Tool results become one ``{"role":"tool"}`` message *per result*, not batched.
- Assistant tool_calls use ``{"type":"function","function":{"name":…,"arguments":"<json>"}}``
  where ``arguments`` is always a JSON-encoded string.
- System prompt is prepended as ``{"role":"system","content":"…"}`` rather than
  a top-level API parameter.
- Tools are wrapped in ``{"type":"function","function":{…}}``.
- ``max_tokens`` maps to ``max_tokens`` for most models; pass
  ``max_completion_tokens`` via ``extra`` for o1/o3 reasoning models.
- ``temperature`` is not sent when ``None`` (o1/o3 models reject it).
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

class OpenAIAdapter(BaseAdapter):
    """Translates between the internal format and the OpenAI Chat Completions API."""

    # -- outbound: internal → OpenAI ----------------------------------------

    def to_provider_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Mapping rules:

        Internal role          OpenAI role / shape
        ──────────────────────────────────────────────────────────────────────
        user                   {"role":"user",   "content": "<str>"}
        assistant (no tools)   {"role":"assistant","content": "<str>"}
        assistant (tools)      {"role":"assistant","content": <str|None>,
                                "tool_calls": [{type,function:{name,arguments}}]}
        tool_result            One {"role":"tool","tool_call_id":…,"content":…}
                               message per ToolResult (OpenAI does not batch).
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.extend(self._convert_message(msg))
        return result

    def _convert_message(self, msg: Message) -> list[dict[str, Any]]:
        if msg.role == MessageRole.user:
            return [{"role": "user", "content": msg.content}]

        if msg.role == MessageRole.assistant:
            if not msg.tool_calls:
                return [{"role": "assistant", "content": msg.content}]
            wire: dict[str, Any] = {
                "role": "assistant",
                # OpenAI allows None when content is absent
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            return [wire]

        if msg.role == MessageRole.tool_result:
            # Expand: one OpenAI message per ToolResult
            return [
                {
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                }
                for tr in msg.tool_results
            ]

        raise ValueError(f"Unknown message role: {msg.role!r}")

    def to_provider_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Internal schemas already use the ``parameters`` key (JSON Schema), which
        is exactly what OpenAI expects.  We only need to add the ``type: function``
        wrapper.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    # -- inbound: OpenAI → internal -----------------------------------------

    def from_provider_response(self, raw: Any) -> LLMResponse:
        """
        Parse an ``openai.types.chat.ChatCompletion`` object (or a compatible
        dict / SimpleNamespace for testing).

        ``raw.choices[0].message`` has:
          .content      str | None
          .tool_calls   list | None, each with .id, .type, .function.name,
                        .function.arguments (JSON string)
        """
        choice = _get(raw, "choices")[0]
        message = _get(choice, "message")

        content: str = _get(message, "content") or ""
        raw_tool_calls = _get(message, "tool_calls") or []

        tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            fn = _get(tc, "function")
            arguments = _get(fn, "arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append(
                ToolCall(
                    id=_get(tc, "id"),
                    name=_get(fn, "name"),
                    arguments=arguments,
                )
            )

        return LLMResponse(content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Provider — wraps OpenAI SDK
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    LLMProvider implementation backed by the OpenAI Chat Completions API.

    Lazy-imports ``openai`` so the rest of the runtime stays importable when
    the SDK is not installed.

    Provider-specific config (from ``ModelConfig.extra``)
    ──────────────────────────────────────────────────────
    system                str   System prompt prepended to every request.
    reasoning_effort      str   "low" | "medium" | "high" — o1/o3 models only.
    response_format       dict  e.g. ``{"type": "json_object"}``.
    max_completion_tokens int   Use instead of max_tokens for o1/o3 models.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._adapter = OpenAIAdapter()
        self._client: Any = None  # initialised lazily

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "The 'openai' package is required to use OpenAIProvider. "
                    "Install it with: pip install openai"
                ) from exc
            api_key = self.config.resolved_api_key("OPENAI_API_KEY")
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        client = self._get_client()
        extra = self.config.extra

        provider_messages = self._adapter.to_provider_messages(messages)
        provider_tools    = self._adapter.to_provider_tools(tools) if tools else []

        # System prompt prepended as first message
        if system := extra.get("system"):
            provider_messages = [{"role": "system", "content": system}] + provider_messages

        call_kwargs: dict[str, Any] = {
            "model":    self.config.model,
            "messages": provider_messages,
        }
        if provider_tools:
            call_kwargs["tools"] = provider_tools

        # Token limit: o1/o3 uses max_completion_tokens; everything else max_tokens
        if "max_completion_tokens" in extra:
            call_kwargs["max_completion_tokens"] = extra["max_completion_tokens"]
        else:
            call_kwargs["max_tokens"] = self.config.max_tokens

        # temperature: omit for o1/o3 (they reject it)
        if self.config.temperature is not None:
            call_kwargs["temperature"] = self.config.temperature

        # reasoning_effort: explicit extra["reasoning_effort"] takes priority over flag
        _reasoning_effort = extra.get("reasoning_effort")
        if _reasoning_effort is None and self.config.reasoning_mode:
            _reasoning_effort = "high"
        if _reasoning_effort:
            call_kwargs["reasoning_effort"] = _reasoning_effort
        if response_format := extra.get("response_format"):
            call_kwargs["response_format"] = response_format

        log.debug(
            "OpenAIProvider.complete model=%s messages=%d tools=%d",
            self.config.model,
            len(provider_messages),
            len(provider_tools),
        )

        raw = client.chat.completions.create(**call_kwargs)
        return self._adapter.from_provider_response(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str) -> Any:
    """Attribute or dict access — works for both SDK objects and dicts/SimpleNamespace."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)
