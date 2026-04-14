"""Provider factory — resolves a ModelConfig into a concrete LLMProvider."""
from __future__ import annotations

from agent.llm.base import LLMProvider
from agent.llm.config import ModelConfig

# Registry maps provider name → zero-arg factory that accepts ModelConfig.
# New providers are added here without touching engine or config code.
_REGISTRY: dict[str, type] = {}


def register_provider(name: str, cls: type) -> None:
    """Register a provider class under *name*."""
    _REGISTRY[name] = cls


def provider_from_config(config: ModelConfig) -> LLMProvider:
    """Instantiate and return the provider described by *config*."""
    cls = _REGISTRY.get(config.provider)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"Unknown provider {config.provider!r}. "
            f"Available providers: {available}"
        )
    return cls(config)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from agent.llm.anthropic import AnthropicProvider  # noqa: PLC0415
    register_provider("anthropic", AnthropicProvider)


_register_builtins()
