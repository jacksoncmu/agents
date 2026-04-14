"""Provider and model configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """
    Provider-agnostic model configuration.

    Portable fields
    ---------------
    provider    : "anthropic" | "openai" (more can be added via registry)
    model       : model ID as the provider spells it, e.g. "claude-sonnet-4-6"
    api_key     : falls back to the provider's conventional env var when empty
    base_url    : override the SDK's default endpoint (proxies, local models)
    max_tokens  : maximum tokens in the response
    temperature : sampling temperature; None lets the provider use its default

    Provider-specific fields
    ------------------------
    Put everything that doesn't map across providers into `extra`.  The
    adapter for each provider picks out the keys it understands and ignores
    the rest.

    Anthropic examples::

        extra={"thinking": {"type": "enabled", "budget_tokens": 5000}}
        extra={"system": "You are a helpful assistant."}

    OpenAI examples::

        extra={"reasoning_effort": "high"}
        extra={"response_format": {"type": "json_object"}}
    """

    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_api_key(self, env_var: str) -> str:
        """Return api_key if set, otherwise the value of *env_var*."""
        return self.api_key or os.environ.get(env_var, "")

    @staticmethod
    def anthropic(
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str = "",
        max_tokens: int = 4096,
        temperature: float | None = None,
        **extra: Any,
    ) -> "ModelConfig":
        """Convenience constructor for Anthropic models."""
        return ModelConfig(
            provider="anthropic",
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=extra,
        )

    @staticmethod
    def openai(
        model: str = "gpt-4o",
        *,
        api_key: str = "",
        max_tokens: int = 4096,
        temperature: float | None = None,
        **extra: Any,
    ) -> "ModelConfig":
        """Convenience constructor for OpenAI models."""
        return ModelConfig(
            provider="openai",
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=extra,
        )
