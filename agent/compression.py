"""Two-layer context compression for long agent sessions.

Layer 1 — Micro-compression (``micro_threshold``)
    Truncates old tool-result content to a short stub.  Preserves message
    structure so the conversation still parses correctly.

Layer 2 — Auto-summarization (``summarization_threshold``)
    Collapses the bulk of the conversation into a structured
    ``ConversationSummary`` and replaces older messages with a single
    re-injection message.

Both layers always preserve
    - the last ``recent_messages_to_preserve`` messages (recent context)
    - the task objective   (from config or auto-detected from first user message)
    - the system identity  (from config)

Telemetry
---------
``compression.micro``      Outcome.executed — layer 1 ran and changed messages
``compression.summarize``  Outcome.executed — layer 2 ran successfully
``compression.summarize``  Outcome.error    — summarizer raised
When neither threshold is crossed no event is emitted (absence = no compression).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.telemetry import Outcome, emit as _emit
from agent.types import Message, MessageRole, ToolResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    """
    Thresholds and identity strings for the two-layer compressor.

    Attributes
    ----------
    context_window
        Estimated token budget for the model.
        Default: 128 000 (fits GPT-4o / Claude-3 / most large models).
    micro_threshold
        Fraction of ``context_window`` at which Layer 1 triggers.
        Default 0.50 → triggers when usage > 50 %.
    summarization_threshold
        Fraction at which Layer 2 triggers.  Must be ≥ ``micro_threshold``.
        Default 0.65 → triggers when usage > 65 %.
    recent_messages_to_preserve
        Both layers keep the last N messages verbatim.  Default 6.
    micro_tool_result_max_chars
        Layer 1 truncates old tool-result content to at most this many
        characters before appending a ``…[truncated]`` marker.  Default 200.
    system_identity
        Short description of the agent's role / persona.  Re-injected at
        the top of the message list after any compression.
    task_objective
        The main goal of the current session.  If empty the compressor
        auto-detects it from the first user message.
    """

    context_window: int = 128_000
    micro_threshold: float = 0.50
    summarization_threshold: float = 0.65
    recent_messages_to_preserve: int = 6
    micro_tool_result_max_chars: int = 200
    system_identity: str = ""
    task_objective: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.micro_threshold <= 1.0):
            raise ValueError("micro_threshold must be in (0, 1]")
        if not (0.0 < self.summarization_threshold <= 1.0):
            raise ValueError("summarization_threshold must be in (0, 1]")
        if self.summarization_threshold < self.micro_threshold:
            raise ValueError(
                "summarization_threshold must be >= micro_threshold"
            )
        if self.recent_messages_to_preserve < 0:
            raise ValueError("recent_messages_to_preserve must be >= 0")
        if self.micro_tool_result_max_chars < 10:
            raise ValueError("micro_tool_result_max_chars must be >= 10")


# ---------------------------------------------------------------------------
# Summary schema
# ---------------------------------------------------------------------------

@dataclass
class ConversationSummary:
    """
    Structured summary of the messages that were compressed away.

    Fields
    ------
    system_identity
        The agent's role/persona (from ``CompressionConfig``).
    task_objective
        The main goal of the session (from config or auto-detected).
    completed_steps
        Ordered list of completed actions extracted from the conversation.
    key_facts
        Important facts discovered during the session.
    message_count_before
        How many messages existed before compression.
    compression_level
        ``"micro"`` (layer 1 only) or ``"summarized"`` (layer 2).
    created_at
        When the summary was produced.
    """

    system_identity: str
    task_objective: str
    completed_steps: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    message_count_before: int = 0
    compression_level: str = "micro"
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list[Message]) -> int:
    """
    Estimate the token count for a message list.

    Uses ``len(text) // 4`` as a simple approximation (reasonable for
    English text with BPE tokenizers).  Adds a small per-message overhead
    for role / separator tokens.

    This is intentionally conservative — it is better to compress slightly
    early than to overflow the context window.
    """
    total = 0
    for m in messages:
        total += 10  # per-message overhead (role token, separators)
        if m.content:
            total += max(1, len(m.content) // 4)
        for tc in m.tool_calls:
            total += 10 + max(1, len(str(tc.arguments)) // 4)
        for tr in m.tool_results:
            total += 10 + max(1, len(tr.content) // 4)
    return total


# ---------------------------------------------------------------------------
# Summarizer interface + implementations
# ---------------------------------------------------------------------------

class Summarizer(ABC):
    """Abstract base class for summarizer implementations."""

    @abstractmethod
    def summarize(
        self,
        messages: list[Message],
        config: CompressionConfig,
    ) -> ConversationSummary:
        """
        Produce a ``ConversationSummary`` from *messages*.

        Must not modify *messages*.
        Must not raise — implementations should catch and handle errors
        internally, possibly returning a minimal summary.
        """
        ...


class NoOpSummarizer(Summarizer):
    """
    Minimal summarizer that does not call an LLM.

    Extracts the objective from the first user message and records message
    counts only.  Suitable for tests, micro-compression-only setups, or
    environments without LLM access.
    """

    def summarize(
        self,
        messages: list[Message],
        config: CompressionConfig,
    ) -> ConversationSummary:
        objective = config.task_objective or _extract_objective(messages)
        return ConversationSummary(
            system_identity=config.system_identity,
            task_objective=objective,
            completed_steps=[],
            key_facts=[],
            message_count_before=len(messages),
            compression_level="summarized",
        )


class LLMSummarizer(Summarizer):
    """
    LLM-backed summarizer.

    Sends a structured prompt to *provider* and parses the free-text
    response into a ``ConversationSummary``.  Falls back to
    ``NoOpSummarizer`` if the LLM call fails, ensuring the caller always
    gets a usable summary.

    Parameters
    ----------
    provider
        Any ``LLMProvider``-compatible object; the summarizer uses a
        single user-message prompt with no tool schemas.
    """

    def __init__(self, provider: Any) -> None:  # Any avoids circular import with llm.base
        self._provider = provider
        self._fallback = NoOpSummarizer()

    def summarize(
        self,
        messages: list[Message],
        config: CompressionConfig,
    ) -> ConversationSummary:
        try:
            return self._call_llm(messages, config)
        except Exception as exc:
            log.warning("LLMSummarizer failed (%s); falling back to NoOpSummarizer", exc)
            summary = self._fallback.summarize(messages, config)
            summary.key_facts.append(f"[Summarizer error: {exc}]")
            return summary

    def _call_llm(
        self,
        messages: list[Message],
        config: CompressionConfig,
    ) -> ConversationSummary:
        objective = config.task_objective or _extract_objective(messages)
        conv_text = _messages_to_text(messages)

        prompt = (
            "You are a conversation summarizer.  Analyze the following agent "
            "conversation and produce a concise, structured summary.\n\n"
            f"Original task objective: {objective}\n\n"
            "Conversation:\n"
            f"{conv_text}\n\n"
            "Respond in EXACTLY this format (include all section headers):\n"
            "OBJECTIVE: <one-sentence objective>\n"
            "STEPS:\n"
            "- <completed step>\n"
            "FACTS:\n"
            "- <key fact discovered>\n"
        )

        response = self._provider.complete([Message.user(prompt)], [])
        return _parse_llm_response(
            text=response.content,
            config=config,
            message_count=len(messages),
        )


# ---------------------------------------------------------------------------
# Context compressor — orchestrates both layers
# ---------------------------------------------------------------------------

class ContextCompressor:
    """
    Two-layer context compressor.

    Call ``compress(messages, session_id)`` at the start of each loop
    iteration (before the LLM call).  If compression ran, the return value
    is a *new* list; if nothing triggered, the *same* list object is
    returned so callers can use ``is`` to detect a no-op.

    Example::

        config = CompressionConfig(
            context_window=32_000,
            system_identity="You are a helpful coding assistant.",
        )
        compressor = ContextCompressor(config, summarizer=LLMSummarizer(provider))
        engine = AgentEngine(store, llm, tools, compressor=compressor)

    Layers
    ------
    Layer 2 (summarization) is checked first; if it triggers, layer 1 is
    skipped.  If layer 2 fails, layer 1 runs as a safety net.
    """

    def __init__(
        self,
        config: CompressionConfig,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.config = config
        self._summarizer = summarizer

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: list[Message],
        session_id: str | None = None,
    ) -> list[Message]:
        """
        Apply compression if token usage exceeds configured thresholds.

        Returns the same list object (``is``-identical) when nothing changed.
        """
        if not messages:
            return messages

        token_estimate = estimate_tokens(messages)
        budget = self.config.context_window

        # --- Layer 2: summarization -----------------------------------------
        summ_limit = int(budget * self.config.summarization_threshold)
        if self._summarizer is not None and token_estimate > summ_limit:
            try:
                compressed = self._apply_summarization(messages)
                _emit(
                    "compression.summarize", Outcome.executed,
                    session_id=session_id,
                    token_estimate=token_estimate,
                    context_window=budget,
                    messages_before=len(messages),
                    messages_after=len(compressed),
                )
                return compressed
            except Exception as exc:
                log.error("Summarization failed: %s", exc)
                _emit(
                    "compression.summarize", Outcome.error,
                    session_id=session_id,
                    token_estimate=token_estimate,
                    error=str(exc),
                )
                # Fall through to layer 1 as a safety net

        # --- Layer 1: micro-compression -------------------------------------
        micro_limit = int(budget * self.config.micro_threshold)
        if token_estimate > micro_limit:
            compressed = self._apply_micro(messages)
            if compressed is not messages:
                _emit(
                    "compression.micro", Outcome.executed,
                    session_id=session_id,
                    token_estimate=token_estimate,
                    context_window=budget,
                    messages_before=len(messages),
                    messages_after=len(compressed),
                )
                return compressed

        # --- Below all thresholds: nothing to do ----------------------------
        return messages

    # ------------------------------------------------------------------
    # Layer 1 — micro-compression
    # ------------------------------------------------------------------

    def _apply_micro(self, messages: list[Message]) -> list[Message]:
        """
        Truncate old tool-result content, preserving structure.

        Strategy
        --------
        - ``messages[0]``  → always kept verbatim (task objective)
        - ``messages[-n:]`` → kept verbatim (recent context)
        - Middle messages  → tool-result content truncated to
          ``micro_tool_result_max_chars``; all other message types
          are passed through unchanged

        Returns the same list object when nothing needed truncation.
        """
        n = self.config.recent_messages_to_preserve
        max_chars = self.config.micro_tool_result_max_chars

        # Need at least: head + 1 middle message + recent tail to do anything
        min_length = 1 + 1 + n
        if len(messages) < min_length:
            return messages

        head = messages[:1]                          # first user message
        recent = messages[-n:] if n > 0 else []
        middle = messages[1: len(messages) - n if n > 0 else len(messages)]

        compacted: list[Message] = []
        changed = False

        for msg in middle:
            if not msg.tool_results:
                compacted.append(msg)
                continue

            new_results: list[ToolResult] = []
            for tr in msg.tool_results:
                if len(tr.content) > max_chars:
                    stub = tr.content[:max_chars] + " …[truncated]"
                    new_results.append(ToolResult(
                        tool_call_id=tr.tool_call_id,
                        content=stub,
                        error=tr.error,
                    ))
                    changed = True
                else:
                    new_results.append(tr)

            compacted.append(Message(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_results=new_results,
            ))

        if not changed:
            return messages

        return head + compacted + recent

    # ------------------------------------------------------------------
    # Layer 2 — summarization
    # ------------------------------------------------------------------

    def _apply_summarization(self, messages: list[Message]) -> list[Message]:
        """
        Replace the bulk of the conversation with a summary + last N messages.

        The result is:
            [summary_message] + messages[-recent_messages_to_preserve:]
        """
        n = self.config.recent_messages_to_preserve
        recent = messages[-n:] if n > 0 else []

        summary = self._summarizer.summarize(messages, self.config)
        summary_msg = _build_summary_message(summary)
        return [summary_msg] + recent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_objective(messages: list[Message]) -> str:
    """Return the content of the first user message, truncated to 500 chars."""
    for m in messages:
        if m.role == MessageRole.user and m.content:
            return m.content[:500]
    return "(objective not found)"


def _build_summary_message(summary: ConversationSummary) -> Message:
    """
    Build the synthetic user message that re-injects identity and objective.

    The message is formatted so the LLM can clearly identify all re-injected
    context, and so tests can assert on the presence of specific strings.
    """
    lines: list[str] = [
        f"[CONTEXT SUMMARY — {summary.message_count_before} messages compressed]",
    ]
    if summary.system_identity:
        lines.append(f"System: {summary.system_identity}")
    lines.append(f"Objective: {summary.task_objective}")

    if summary.completed_steps:
        lines.append("\nSteps completed:")
        for step in summary.completed_steps:
            lines.append(f"  - {step}")

    if summary.key_facts:
        lines.append("\nKey facts:")
        for fact in summary.key_facts:
            lines.append(f"  - {fact}")

    lines.append("[END SUMMARY]")
    return Message.user("\n".join(lines))


def _messages_to_text(messages: list[Message], max_content: int = 800) -> str:
    """
    Render messages as a readable text block for the summarization prompt.
    Truncates individual messages to ``max_content`` chars to keep the
    prompt itself from overflowing.
    """
    parts: list[str] = []
    for m in messages:
        if m.role == MessageRole.user:
            if m.content:
                parts.append(f"[User]: {m.content[:max_content]}")
        elif m.role == MessageRole.assistant:
            if m.content:
                parts.append(f"[Assistant]: {m.content[:max_content]}")
            for tc in m.tool_calls:
                parts.append(f"[Tool call: {tc.name}] {str(tc.arguments)[:300]}")
        elif m.role == MessageRole.tool_result:
            for tr in m.tool_results:
                status = "error" if tr.error else "ok"
                parts.append(f"[Tool result ({status})]: {tr.content[:300]}")
    return "\n".join(parts)


def _parse_llm_response(
    text: str,
    config: CompressionConfig,
    message_count: int,
) -> ConversationSummary:
    """
    Parse the structured LLM response into a ConversationSummary.

    Expected format::

        OBJECTIVE: <text>
        STEPS:
        - <step>
        FACTS:
        - <fact>

    Missing sections produce empty lists rather than errors.
    """
    objective = config.task_objective
    steps: list[str] = []
    facts: list[str] = []
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("OBJECTIVE:"):
            objective = line[len("OBJECTIVE:"):].strip() or objective
            current_section = None
        elif upper.startswith("STEPS:") or upper == "STEPS":
            current_section = "steps"
        elif upper.startswith("FACTS:") or upper == "FACTS":
            current_section = "facts"
        elif line.startswith("- ") or line.startswith("* "):
            item = line[2:].strip()
            if item:
                if current_section == "steps":
                    steps.append(item)
                elif current_section == "facts":
                    facts.append(item)

    return ConversationSummary(
        system_identity=config.system_identity,
        task_objective=objective or _extract_objective([]),
        completed_steps=steps,
        key_facts=facts,
        message_count_before=message_count,
        compression_level="summarized",
    )
