"""Output limiting for tool execution results.

Enforces hard character caps per tool type.  When a tool's output exceeds
its cap, the full payload is stored in a ``BlobStore`` and the tool result
is replaced with a short reference stub.

Tool type classification
------------------------
Each tool is assigned a *tool type* that determines its output cap.
Types are assigned explicitly via ``OutputLimitConfig.tool_types`` or
fall back to the ``default`` type.

Built-in caps::

    file_read:  20 000 chars
    search:     10 000 chars
    shell:      15 000 chars
    default:    15 000 chars

Reference stub format::

    [OUTPUT TRUNCATED — {n} chars exceeded {cap} cap]
    Full output stored as: {blob_ref}
    Retrieve with: blob_store.get("{blob_ref}")

Telemetry
---------
``output.capped``  Outcome.executed — output was redirected to blob store
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.blob_store import BlobStore, InMemoryBlobStore
from agent.telemetry import Outcome, emit as _emit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default caps (chars)
# ---------------------------------------------------------------------------

DEFAULT_CAPS: dict[str, int] = {
    "file_read": 20_000,
    "search": 10_000,
    "shell": 15_000,
    "default": 15_000,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OutputLimitConfig:
    """
    Per-type output caps and tool→type mapping.

    Attributes
    ----------
    caps
        Maps tool type names to character limits.
        Merged with ``DEFAULT_CAPS`` (explicit entries override defaults).
    tool_types
        Maps tool *names* to tool *types*.  A tool not listed here uses
        the ``"default"`` type.
    """

    caps: dict[str, int] = field(default_factory=dict)
    tool_types: dict[str, str] = field(default_factory=dict)

    def effective_caps(self) -> dict[str, int]:
        """Return default caps merged with any overrides."""
        merged = dict(DEFAULT_CAPS)
        merged.update(self.caps)
        return merged

    def cap_for_tool(self, tool_name: str) -> int:
        """Return the character cap for *tool_name*."""
        effective = self.effective_caps()
        tool_type = self.tool_types.get(tool_name, "default")
        return effective.get(tool_type, effective["default"])


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

class OutputLimiter:
    """
    Enforces output caps and redirects oversized outputs to blob storage.

    Integrate into ``ToolRegistry`` by calling ``limit()`` on every
    ``ExecutionResult.output`` before returning it to the engine.
    """

    def __init__(
        self,
        config: OutputLimitConfig | None = None,
        blob_store: BlobStore | None = None,
    ) -> None:
        self.config = config or OutputLimitConfig()
        self.blob_store = blob_store or InMemoryBlobStore()

    def limit(
        self,
        tool_name: str,
        output: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Apply the output cap for *tool_name*.

        If *output* is within the cap, returns it unchanged.
        If it exceeds the cap, stores the full output in the blob store
        and returns a reference stub.
        """
        cap = self.config.cap_for_tool(tool_name)
        if len(output) <= cap:
            return output

        # Store oversized output
        blob_metadata = {
            "tool_name": tool_name,
            "original_length": len(output),
            "cap": cap,
            **(metadata or {}),
        }
        if session_id:
            blob_metadata["session_id"] = session_id

        ref = self.blob_store.put(output, metadata=blob_metadata)

        log.info(
            "Output for tool %r capped: %d chars exceeds %d cap → %s",
            tool_name, len(output), cap, ref,
        )
        _emit(
            "output.capped", Outcome.executed,
            session_id=session_id,
            tool_name=tool_name,
            original_length=len(output),
            cap=cap,
            blob_ref=ref,
        )

        stub = (
            f"[OUTPUT TRUNCATED — {len(output)} chars exceeded {cap} cap]\n"
            f"Full output stored as: {ref}\n"
            f'Retrieve with: blob_store.get("{ref}")'
        )
        return stub
