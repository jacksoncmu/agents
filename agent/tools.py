"""Tool registry and execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class ToolDefinition:
    name: str
    description: str
    params: list[ToolParam]
    handler: Callable[..., str]

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-schema-compatible dict for passing to the LLM."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            properties[p.name] = {"type": p.type, "description": p.description}
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


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}")
        return tool.handler(**arguments)


def tool(
    name: str,
    description: str,
    params: list[ToolParam] | None = None,
) -> Callable[[Callable[..., str]], ToolDefinition]:
    """Decorator helper for defining tools inline."""

    def decorator(fn: Callable[..., str]) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            params=params or [],
            handler=fn,
        )

    return decorator
