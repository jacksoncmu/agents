"""Example tools that can be registered with ToolRegistry."""
from __future__ import annotations

import datetime

from agent.tools import ToolDefinition, ToolParam


def make_calculator() -> ToolDefinition:
    """Simple calculator: evaluates safe arithmetic expressions."""

    def handler(expression: str) -> str:
        # Restrict to safe arithmetic to avoid arbitrary code execution.
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            raise ValueError(f"Unsafe characters in expression: {expression!r}")
        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
        except Exception as exc:
            raise ValueError(f"Could not evaluate {expression!r}: {exc}") from exc
        return str(result)

    return ToolDefinition(
        name="calculator",
        description="Evaluate a simple arithmetic expression and return the result.",
        params=[
            ToolParam(
                name="expression",
                type="string",
                description="Arithmetic expression to evaluate, e.g. '2 + 3 * 4'",
            )
        ],
        handler=handler,
    )


def make_get_current_time() -> ToolDefinition:
    """Returns the current UTC time."""

    def handler() -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return ToolDefinition(
        name="get_current_time",
        description="Return the current UTC date and time.",
        params=[],
        handler=handler,
    )
