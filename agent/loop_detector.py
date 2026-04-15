"""Loop detection and safe intervention message handling.

Loop detection
--------------
``LoopDetector`` examines recent tool-call history for repeated patterns.
A *pattern* is the tuple ``(tool_name, sorted_arguments_repr)``.  When the
same pattern appears ``repeat_threshold`` or more times within the last
``window_size`` assistant messages, the detector signals a loop.

Intervention queue
------------------
``InterventionQueue`` holds system-generated guidance messages that must
not be injected mid-protocol.  The LLM protocol requires that every
assistant message containing tool_calls is immediately followed by a
tool_result message.  Interventions are therefore *queued* during tool
execution and only *flushed* after the tool_result message has been
appended to the conversation.

Telemetry
---------
``loop.detected``   Outcome.blocked    — a repetitive pattern was found
``loop.not_detected`` Outcome.condition_unsatisfied — checked, no loop
``loop.intervention_flushed`` Outcome.executed — queued messages injected
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.telemetry import Outcome, emit as _emit
from agent.types import Message, ToolCall

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoopDetectorConfig:
    """
    Parameters for the loop detector.

    Attributes
    ----------
    window_size
        Number of most-recent assistant messages to examine.  Default 10.
    repeat_threshold
        How many times the *same* tool-call pattern must appear within
        the window to be considered a loop.  Default 3.
    enabled
        Master switch.  When False, ``check()`` always returns no-loop.
    """

    window_size: int = 10
    repeat_threshold: int = 3
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold must be >= 2")


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopDetection:
    """Result of a single loop check."""

    detected: bool
    pattern: str = ""
    count: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Intervention queue
# ---------------------------------------------------------------------------

class InterventionQueue:
    """
    Holds intervention messages until they can be safely injected.

    Protocol correctness rule: an assistant message with tool_calls must
    be immediately followed by a tool_result message.  Intervention
    messages (which are user-role) must therefore wait until *after* the
    tool_result message is appended.

    Usage in the engine loop::

        # ... tool execution produces results ...
        session.messages.append(Message.tool_result_msg(results))

        # NOW it is safe to flush
        for msg in intervention_queue.flush():
            session.messages.append(msg)
    """

    def __init__(self) -> None:
        self._queue: list[Message] = []

    def enqueue(self, message: Message) -> None:
        """Add a message to the queue."""
        self._queue.append(message)

    def flush(self) -> list[Message]:
        """Return and clear all queued messages."""
        msgs = list(self._queue)
        self._queue.clear()
        return msgs

    @property
    def pending(self) -> int:
        """Number of messages waiting to be flushed."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------

def _tool_call_pattern(tc: ToolCall) -> str:
    """Produce a canonical string key for a tool call."""
    # Sort argument keys for deterministic comparison
    args_repr = str(sorted(tc.arguments.items()))
    return f"{tc.name}:{args_repr}"


class LoopDetector:
    """
    Detects repeated tool-call patterns in recent conversation history.

    Call ``check(messages, session_id)`` after each iteration's tool
    results are appended.  If a loop is detected, the caller should
    generate an intervention message and enqueue it.

    The detector is stateless — it examines the message list directly
    each time.  This makes it safe across checkpoints and recovery.
    """

    def __init__(self, config: LoopDetectorConfig | None = None) -> None:
        self.config = config or LoopDetectorConfig()

    def check(
        self,
        messages: list[Message],
        session_id: str | None = None,
        iteration: int | None = None,
    ) -> LoopDetection:
        """
        Examine recent messages for repeated tool-call patterns.

        Returns a ``LoopDetection`` indicating whether a loop was found.
        Emits telemetry regardless of the outcome.
        """
        if not self.config.enabled:
            return LoopDetection(detected=False)

        # Collect tool-call patterns from the last N assistant messages
        assistant_msgs = [
            m for m in messages if m.tool_calls
        ]
        window = assistant_msgs[-self.config.window_size:]

        if not window:
            _emit(
                "loop.not_detected", Outcome.condition_unsatisfied,
                session_id=session_id, iteration=iteration,
                reason="no_tool_calls_in_window",
            )
            return LoopDetection(detected=False)

        # Count each pattern
        counts: dict[str, int] = {}
        for msg in window:
            # Build a composite key from ALL tool calls in one assistant msg
            pattern = "|".join(_tool_call_pattern(tc) for tc in msg.tool_calls)
            counts[pattern] = counts.get(pattern, 0) + 1

        # Find the most repeated pattern
        worst_pattern = max(counts, key=counts.get)  # type: ignore[arg-type]
        worst_count = counts[worst_pattern]

        if worst_count >= self.config.repeat_threshold:
            reason = (
                f"Pattern repeated {worst_count} times in last "
                f"{len(window)} assistant messages (threshold: "
                f"{self.config.repeat_threshold})"
            )
            log.info(
                "Loop detected in session %s: %s",
                session_id, reason,
            )
            _emit(
                "loop.detected", Outcome.blocked,
                session_id=session_id, iteration=iteration,
                pattern=worst_pattern,
                count=worst_count,
                threshold=self.config.repeat_threshold,
            )
            return LoopDetection(
                detected=True,
                pattern=worst_pattern,
                count=worst_count,
                reason=reason,
            )

        _emit(
            "loop.not_detected", Outcome.condition_unsatisfied,
            session_id=session_id, iteration=iteration,
            max_repeat=worst_count,
            threshold=self.config.repeat_threshold,
        )
        return LoopDetection(detected=False)

    @staticmethod
    def make_intervention_message(detection: LoopDetection) -> Message:
        """
        Build the intervention user message for an detected loop.

        The message tells the model what happened and asks it to change
        strategy.  It does NOT prescribe a specific fix — that would be
        a business-specific rule.
        """
        text = (
            "[LOOP DETECTED] You have been repeating the same tool call "
            f"pattern {detection.count} times: {detection.pattern}\n"
            "This is not making progress. Please try a different approach "
            "or provide a final answer."
        )
        return Message.user(text)
