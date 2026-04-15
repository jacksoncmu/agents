"""Tool registry and execution.

Design
------
ToolDefinition  — schema + handler + optional flags (requires_confirmation)
ToolRegistry    — register/unregister/execute; all_schemas() is always fresh
ExecutionResult — structured result returned by every execute(); never raises
ToolContext     — injected into handlers that declare a ``ctx`` parameter;
                  lets tools emit structured log lines without side-effects
ValidationError — one per broken constraint; collected before any execution
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.output_limiter import OutputLimiter
from agent.telemetry import Outcome, emit as _emit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolContext  — optional log collector for handlers
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """
    Passed to handlers that declare a ``ctx: ToolContext`` parameter.
    Lets handlers emit structured log lines accessible to the engine.
    """
    _logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self._logs.append(message)

    @property
    def logs(self) -> list[str]:
        return list(self._logs)


# ---------------------------------------------------------------------------
# Parameter + schema types
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    name: str
    type: str          # JSON Schema primitive: string | integer | number | boolean | array | object
    description: str
    required: bool = True
    enum: list[Any] | None = None   # restrict to specific values


@dataclass
class ValidationError:
    param: str
    message: str

    def __str__(self) -> str:
        return f"{self.param}: {self.message}"


# Type-check lambdas keyed on JSON Schema primitive names.
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string":  lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array":   lambda v: isinstance(v, list),
    "object":  lambda v: isinstance(v, dict),
}


# ---------------------------------------------------------------------------
# ExecutionResult — what execute() always returns
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    output: str
    error: bool = False
    logs: list[str] = field(default_factory=list)
    requires_confirmation: bool = False


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    name: str
    description: str
    params: list[ToolParam]
    handler: Callable[..., str]
    requires_confirmation: bool = False

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-schema-compatible dict for passing to the LLM."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum is not None:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def validate(self, arguments: dict[str, Any]) -> list[ValidationError]:
        """
        Validate *arguments* against this tool's parameter schema.
        Returns a (possibly empty) list of ValidationError; does not raise.
        """
        errors: list[ValidationError] = []

        # 1. Required params must be present
        for p in self.params:
            if p.required and p.name not in arguments:
                errors.append(ValidationError(p.name, "required parameter is missing"))

        # 2. Per-param type and enum checks
        for p in self.params:
            if p.name not in arguments:
                continue
            value = arguments[p.name]

            type_check = _TYPE_CHECKS.get(p.type)
            if type_check and not type_check(value):
                errors.append(
                    ValidationError(p.name, f"expected type {p.type!r}, got {type(value).__name__!r}")
                )
                continue  # skip enum check if type is wrong

            if p.enum is not None and value not in p.enum:
                errors.append(
                    ValidationError(p.name, f"value {value!r} not in allowed values {p.enum}")
                )

        return errors


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, output_limiter: OutputLimiter | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._output_limiter = output_limiter

    # -- registration -------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            log.debug("Replacing existing tool %r", tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # -- schema snapshot (always fresh) ------------------------------------

    def all_schemas(self) -> list[dict[str, Any]]:
        """
        Return a fresh schema list.  Called at the start of every loop
        iteration so tools registered after a turn are visible next turn.
        """
        return [t.to_schema() for t in self._tools.values()]

    # -- execution ---------------------------------------------------------

    def execute(self, name: str, arguments: dict[str, Any]) -> ExecutionResult:
        """
        Validate and execute a tool.  Never raises — all outcomes are
        expressed through ExecutionResult fields.

        Callers should check:
          result.error               — validation or runtime failure
          result.requires_confirmation — pause for human approval before acting
          result.logs                — structured log lines from the handler
        """
        tool = self.get(name)
        if tool is None:
            log.warning("Tool %r not found in registry", name)
            _emit("tool.execute", Outcome.error, tool_name=name, reason="unknown_tool")
            return ExecutionResult(
                output=f"Unknown tool: {name!r}",
                error=True,
            )

        # Validate inputs before touching the handler
        errors = tool.validate(arguments)
        if errors:
            msg = "; ".join(str(e) for e in errors)
            log.warning("Validation failed for tool %r: %s", name, msg)
            _emit(
                "tool.validate", Outcome.error,
                tool_name=name,
                errors=[{"param": e.param, "message": e.message} for e in errors],
            )
            return ExecutionResult(
                output=f"Validation failed: {msg}",
                error=True,
            )
        _emit("tool.validate", Outcome.executed, tool_name=name)

        # Confirmation gate — return early; engine decides what to do
        if tool.requires_confirmation:
            log.debug("Tool %r requires confirmation, deferring execution", name)
            _emit("tool.confirmation_gate", Outcome.blocked, tool_name=name)
            return ExecutionResult(output="", requires_confirmation=True)

        return self._invoke(tool, arguments)

    def execute_confirmed(self, name: str, arguments: dict[str, Any]) -> ExecutionResult:
        """
        Execute a tool that has ``requires_confirmation=True`` after the
        human has approved.  Bypasses the confirmation gate; validation
        still runs.
        """
        tool = self.get(name)
        if tool is None:
            _emit("tool.execute", Outcome.error, tool_name=name, reason="unknown_tool", confirmed=True)
            return ExecutionResult(output=f"Unknown tool: {name!r}", error=True)

        errors = tool.validate(arguments)
        if errors:
            msg = "; ".join(str(e) for e in errors)
            _emit(
                "tool.validate", Outcome.error,
                tool_name=name,
                errors=[{"param": e.param, "message": e.message} for e in errors],
                confirmed=True,
            )
            return ExecutionResult(output=f"Validation failed: {msg}", error=True)

        _emit("tool.validate", Outcome.executed, tool_name=name, confirmed=True)
        return self._invoke(tool, arguments, confirmed=True)

    # -- internal ----------------------------------------------------------

    def _invoke(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        confirmed: bool = False,
    ) -> ExecutionResult:
        ctx = ToolContext()
        try:
            sig = inspect.signature(tool.handler)
            if "ctx" in sig.parameters:
                raw = tool.handler(**arguments, ctx=ctx)
            else:
                raw = tool.handler(**arguments)
        except Exception as exc:
            log.exception("Tool %r raised during execution: %s", tool.name, exc)
            _emit(
                "tool.execute", Outcome.error,
                tool_name=tool.name,
                error=str(exc),
                confirmed=confirmed,
            )
            return ExecutionResult(
                output=f"Tool execution failed: {exc}",
                error=True,
                logs=ctx.logs,
            )
        _emit("tool.execute", Outcome.executed, tool_name=tool.name, confirmed=confirmed)
        output = str(raw)
        if self._output_limiter is not None:
            output = self._output_limiter.limit(tool.name, output)
        return ExecutionResult(output=output, logs=ctx.logs)
