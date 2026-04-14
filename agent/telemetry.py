"""Lightweight telemetry / observability layer.

Architecture
------------
Every decision point in the runtime emits a structured ``Event`` to a
swappable ``TelemetryBackend``.  The default backend discards events (zero
overhead).  Swap it in tests or production with ``set_backend()``.

Quick start
-----------
Emit an event from anywhere::

    from agent.telemetry import emit, Outcome

    emit("tool.execute", Outcome.executed, tool_name="calculator")

Capture events in tests::

    from agent.telemetry import InMemoryBackend, set_backend

    backend = InMemoryBackend()
    set_backend(backend)

    # ... run the system under test ...

    execs = backend.by_name("tool.execute")
    assert any(e.outcome == Outcome.executed for e in execs)

Replace the backend in production::

    from agent.telemetry import LoggingBackend, set_backend
    set_backend(LoggingBackend("my.telemetry"))

    # Or wire up OpenTelemetry / Datadog / etc. via a custom backend:
    class OTelBackend:
        def emit(self, event: Event) -> None:
            tracer.start_as_current_span(event.name)
            ...
    set_backend(OTelBackend())

Outcome vocabulary
------------------
``executed``              — the action was taken successfully.
``condition_unsatisfied`` — the decision point was evaluated; the condition
                            was not met; the action was skipped as expected
                            (e.g. loop exits cleanly because LLM had no tool
                            calls).  Maps to "not_executed_condition_unsatisfied"
                            in external tracing conventions.
``blocked``               — a policy gate or ownership lock intercepted the
                            action before it could run.
``error``                 — an error was encountered.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    executed             = "executed"
    condition_unsatisfied = "condition_unsatisfied"  # expected skip — condition not met
    blocked              = "blocked"                 # intercepted by gate / lock / policy
    error                = "error"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Immutable record of one decision point being evaluated."""
    name: str
    outcome: Outcome
    ts: datetime
    session_id: str | None = None
    iteration: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend protocol + built-in implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class TelemetryBackend(Protocol):
    def emit(self, event: Event) -> None: ...


class NullBackend:
    """Default: discard every event.  Zero allocation, zero I/O."""
    def emit(self, event: Event) -> None:
        pass


class LoggingBackend:
    """Forward events to Python's ``logging`` system."""

    def __init__(self, logger_name: str = "agent.telemetry") -> None:
        self._log = logging.getLogger(logger_name)

    def emit(self, event: Event) -> None:
        self._log.info(
            "event=%s outcome=%s session=%s iter=%s %s",
            event.name,
            event.outcome.value,
            event.session_id,
            event.iteration,
            " ".join(f"{k}={v!r}" for k, v in event.data.items()),
        )


class InMemoryBackend:
    """
    Collect events in an in-process list.

    Intended for unit tests and local debugging.  Thread-safe.

    Usage::

        backend = InMemoryBackend()
        set_backend(backend)
        # ... exercise code ...
        events = backend.by_name("tool.execute")
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def emit(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    def by_name(self, name: str) -> list[Event]:
        """Return all events with the given name."""
        with self._lock:
            return [e for e in self._events if e.name == name]

    def by_outcome(self, outcome: Outcome) -> list[Event]:
        """Return all events with the given outcome."""
        with self._lock:
            return [e for e in self._events if e.outcome == outcome]

    def first(self, name: str) -> Event | None:
        """Return the first event with the given name, or None."""
        with self._lock:
            return next((e for e in self._events if e.name == name), None)

    def names(self) -> list[str]:
        """Return all distinct event names seen so far (in order of first appearance)."""
        with self._lock:
            seen: list[str] = []
            for e in self._events:
                if e.name not in seen:
                    seen.append(e.name)
            return seen

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


# ---------------------------------------------------------------------------
# Telemetry wrapper (swappable backend)
# ---------------------------------------------------------------------------

class Telemetry:
    """
    Thread-safe wrapper around a ``TelemetryBackend``.

    Holds the backend swap lock only long enough to read the pointer —
    never while the backend's ``emit`` runs — so slow backends don't
    block unrelated threads.
    """

    def __init__(self, backend: TelemetryBackend | None = None) -> None:
        self._backend: TelemetryBackend = backend or NullBackend()
        self._swap_lock = threading.Lock()

    def set_backend(self, backend: TelemetryBackend) -> None:
        with self._swap_lock:
            self._backend = backend

    def emit(
        self,
        name: str,
        outcome: Outcome,
        *,
        session_id: str | None = None,
        iteration: int | None = None,
        **data: Any,
    ) -> None:
        event = Event(
            name=name,
            outcome=outcome,
            ts=datetime.utcnow(),
            session_id=session_id,
            iteration=iteration,
            data=data,
        )
        with self._swap_lock:
            backend = self._backend
        backend.emit(event)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience helpers
# ---------------------------------------------------------------------------

_telemetry = Telemetry()


def get_telemetry() -> Telemetry:
    """Return the process-wide Telemetry instance."""
    return _telemetry


def set_backend(backend: TelemetryBackend) -> None:
    """Swap the active backend (thread-safe, takes effect immediately)."""
    _telemetry.set_backend(backend)


def emit(
    name: str,
    outcome: Outcome,
    *,
    session_id: str | None = None,
    iteration: int | None = None,
    **data: Any,
) -> None:
    """
    Emit a single telemetry event to the current backend.

    Parameters
    ----------
    name:       dot-separated event name, e.g. ``"tool.execute"``
    outcome:    one of the ``Outcome`` enum values
    session_id: optional session context
    iteration:  optional loop-iteration context
    **data:     arbitrary key/value payload (must be JSON-serialisable)
    """
    _telemetry.emit(name, outcome, session_id=session_id, iteration=iteration, **data)
