"""Shared types for the agent runtime."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    waiting_for_user = "waiting_for_user"
    running = "running"
    waiting_for_confirmation = "waiting_for_confirmation"
    finished = "finished"
    error = "error"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    tool_result = "tool_result"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def new(name: str, arguments: dict[str, Any]) -> "ToolCall":
        return ToolCall(id=str(uuid.uuid4()), name=name, arguments=arguments)


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    error: bool = False


@dataclass
class Message:
    role: MessageRole
    # For user/assistant: plain text content (may be empty if tool_calls present)
    content: str = ""
    # Only on assistant messages
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Only on tool_result messages
    tool_results: list[ToolResult] = field(default_factory=list)

    @staticmethod
    def user(content: str) -> "Message":
        return Message(role=MessageRole.user, content=content)

    @staticmethod
    def assistant(content: str, tool_calls: list[ToolCall] | None = None) -> "Message":
        return Message(
            role=MessageRole.assistant,
            content=content,
            tool_calls=tool_calls or [],
        )

    @staticmethod
    def tool_result_msg(results: list[ToolResult]) -> "Message":
        return Message(role=MessageRole.tool_result, tool_results=results)


@dataclass
class Session:
    id: str
    state: SessionState = SessionState.waiting_for_user
    messages: list[Message] = field(default_factory=list)
    cancelled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    # Tool calls deferred pending human confirmation.  Populated when the
    # engine transitions to waiting_for_confirmation; cleared on resume.
    pending_confirmation: list[ToolCall] = field(default_factory=list)

    @staticmethod
    def new() -> "Session":
        return Session(id=str(uuid.uuid4()))
