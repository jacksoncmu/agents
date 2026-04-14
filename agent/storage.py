"""Storage layer for sessions, messages, checkpoints, and run status.

Public surface
--------------
Checkpoint      — immutable snapshot taken at a safe loop boundary
RunStatus       — live status view of one session (no messages)
SessionSummary  — lightweight listing row (no full message list)
SessionStore    — abstract interface; implement for any backend
InMemoryStore   — default, single-process, zero-config
FileStore       — JSON/JSONL files; survives process restarts

Checkpoint labels written by the engine
----------------------------------------
LABEL_POST_LLM    after appending the assistant message
LABEL_POST_TOOLS  after appending all tool results (safest resume point)
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.types import (
    Message,
    MessageRole,
    Session,
    SessionState,
    ToolCall,
    ToolResult,
)

# Checkpoint label constants used by the engine
LABEL_POST_LLM   = "post_llm"
LABEL_POST_TOOLS = "post_tools"


# ---------------------------------------------------------------------------
# Storage types
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """
    Immutable snapshot of session state at a known-safe loop boundary.

    message_count   — len(session.messages) at this point; used to truncate
                      the message list when restoring.
    label           — one of LABEL_POST_LLM / LABEL_POST_TOOLS (or custom).
    iteration       — which engine loop iteration produced this checkpoint.
    """
    id: str
    session_id: str
    iteration: int
    message_count: int
    state: SessionState
    label: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def new(
        session_id: str,
        iteration: int,
        message_count: int,
        state: SessionState,
        label: str,
    ) -> "Checkpoint":
        return Checkpoint(
            id=str(uuid.uuid4()),
            session_id=session_id,
            iteration=iteration,
            message_count=message_count,
            state=state,
            label=label,
        )


@dataclass
class RunStatus:
    """
    Live status of a session — no messages, cheap to fetch.

    is_running        — True while the engine loop is executing.
    cancel_requested  — True if cancel() has been called.
    current_state     — the SessionState enum value.
    """
    session_id: str
    current_state: SessionState
    is_running: bool
    cancel_requested: bool
    error_message: str
    updated_at: datetime


@dataclass
class SessionSummary:
    """
    Lightweight listing row — enough for a session-picker UI or CLI.

    last_messages contains the final min(n, total) messages so callers can
    show context without loading the full message list.
    """
    session_id: str
    current_state: SessionState
    is_running: bool
    cancel_requested: bool
    message_count: int
    last_messages: list[Message]
    error_message: str
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# In-process run-ownership tracker (shared by all store implementations)
# ---------------------------------------------------------------------------

class _RunOwnership:
    """
    Tracks which sessions currently have an active engine run.

    Uses a single ``threading.Lock`` to protect the ownership set, but holds
    that lock for *microseconds* only — long enough to perform an atomic
    check-and-set.  The actual work (LLM calls, tool execution) runs entirely
    outside this lock.

    This is process-local: ownership is not persisted.  After a process
    restart every session is implicitly unowned; ``continue_run`` handles
    crash-recovery via checkpoints.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._owned: set[str] = set()

    def try_acquire(self, session_id: str) -> bool:
        """Atomically claim ownership.  Returns True if claimed, False if already owned."""
        with self._guard:
            if session_id in self._owned:
                return False
            self._owned.add(session_id)
            return True

    def release(self, session_id: str) -> None:
        """Release ownership unconditionally (safe to call even if not owned)."""
        with self._guard:
            self._owned.discard(session_id)

    def is_owned(self, session_id: str) -> bool:
        with self._guard:
            return session_id in self._owned


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class SessionStore(ABC):
    """
    Backend-agnostic interface for all persistence operations.

    Session CRUD
    ────────────
    create / get / save / delete / list_ids  — keep existing engine call-sites
    list_sessions                            — richer listing for UIs / CLIs
    set_cancellation                         — atomic flag update (no full save)
    get_run_status                           — lightweight status without messages

    Checkpoints
    ───────────
    save_checkpoint / latest_checkpoint / list_checkpoints / delete_checkpoints
    """

    # -- Session CRUD -------------------------------------------------------

    @abstractmethod
    def create(self) -> Session: ...

    @abstractmethod
    def get(self, session_id: str) -> Session | None: ...

    @abstractmethod
    def save(self, session: Session) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def list_ids(self) -> list[str]: ...

    # -- Richer queries ------------------------------------------------------

    @abstractmethod
    def list_sessions(self, last_n_messages: int = 3) -> list[SessionSummary]: ...

    @abstractmethod
    def get_run_status(self, session_id: str) -> RunStatus | None: ...

    # -- Targeted mutation ---------------------------------------------------

    @abstractmethod
    def set_cancellation(self, session_id: str, value: bool) -> None:
        """Atomically set the cancellation flag without a full session round-trip."""
        ...

    # -- Run ownership (in-process, not persisted) ---------------------------

    @abstractmethod
    def try_acquire_run(self, session_id: str) -> bool:
        """
        Atomically claim run ownership for *session_id*.

        Returns True if ownership was acquired (caller may proceed).
        Returns False if another run is already active (caller must reject).

        The implementation must hold its internal mutex for the minimum time
        needed to perform the check-and-set — never while doing I/O or
        long-running work.
        """
        ...

    @abstractmethod
    def release_run(self, session_id: str) -> None:
        """Release run ownership.  Safe to call even when not owned (idempotent)."""
        ...

    @abstractmethod
    def is_run_owned(self, session_id: str) -> bool:
        """Return True if a run currently owns this session."""
        ...

    # -- Checkpoints ---------------------------------------------------------

    @abstractmethod
    def save_checkpoint(self, cp: Checkpoint) -> None: ...

    @abstractmethod
    def latest_checkpoint(self, session_id: str) -> Checkpoint | None: ...

    @abstractmethod
    def list_checkpoints(self, session_id: str) -> list[Checkpoint]: ...

    @abstractmethod
    def delete_checkpoints(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryStore(SessionStore):
    """Thread-unsafe in-memory store — zero setup, sufficient for single-process use."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        # session_id → ordered list of checkpoints (oldest first)
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._ownership = _RunOwnership()

    # -- Session CRUD -------------------------------------------------------

    def create(self) -> Session:
        session = Session.new()
        self._sessions[session.id] = session
        self._checkpoints[session.id] = []
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        session.updated_at = datetime.utcnow()
        self._sessions[session.id] = session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._checkpoints.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())

    # -- Richer queries ------------------------------------------------------

    def list_sessions(self, last_n_messages: int = 3) -> list[SessionSummary]:
        summaries = []
        for s in self._sessions.values():
            summaries.append(SessionSummary(
                session_id=s.id,
                current_state=s.state,
                is_running=(s.state == SessionState.running),
                cancel_requested=s.cancelled,
                message_count=len(s.messages),
                last_messages=s.messages[-last_n_messages:] if s.messages else [],
                error_message=s.error_message,
                created_at=s.created_at,
                updated_at=s.updated_at,
            ))
        return summaries

    def get_run_status(self, session_id: str) -> RunStatus | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return RunStatus(
            session_id=s.id,
            current_state=s.state,
            is_running=(s.state == SessionState.running),
            cancel_requested=s.cancelled,
            error_message=s.error_message,
            updated_at=s.updated_at,
        )

    # -- Targeted mutation ---------------------------------------------------

    def set_cancellation(self, session_id: str, value: bool) -> None:
        s = self._sessions.get(session_id)
        if s is not None:
            s.cancelled = value
            s.updated_at = datetime.utcnow()

    # -- Run ownership -------------------------------------------------------

    def try_acquire_run(self, session_id: str) -> bool:
        return self._ownership.try_acquire(session_id)

    def release_run(self, session_id: str) -> None:
        self._ownership.release(session_id)

    def is_run_owned(self, session_id: str) -> bool:
        return self._ownership.is_owned(session_id)

    # -- Checkpoints ---------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        self._checkpoints.setdefault(cp.session_id, []).append(cp)

    def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        cps = self._checkpoints.get(session_id)
        return cps[-1] if cps else None

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        return list(self._checkpoints.get(session_id, []))

    def delete_checkpoints(self, session_id: str) -> None:
        self._checkpoints[session_id] = []


# ---------------------------------------------------------------------------
# File-backed implementation
# ---------------------------------------------------------------------------

class FileStore(SessionStore):
    """
    JSON / JSONL file-backed store.  Survives process restarts.

    Layout::

        {base_dir}/
          sessions/
            {session_id}.json          full session (messages + metadata)
          checkpoints/
            {session_id}.jsonl         one checkpoint JSON per line (append-only)

    Writes are atomic (write to .tmp then os.replace) so a crash mid-write
    leaves the previous version intact.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._ownership = _RunOwnership()

    @property
    def _sessions_dir(self) -> Path:
        return self._base / "sessions"

    @property
    def _checkpoints_dir(self) -> Path:
        return self._base / "checkpoints"

    def _session_path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"

    def _checkpoint_path(self, session_id: str) -> Path:
        return self._checkpoints_dir / f"{session_id}.jsonl"

    # -- Session CRUD -------------------------------------------------------

    def create(self) -> Session:
        session = Session.new()
        self.save(session)
        return session

    def get(self, session_id: str) -> Session | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _session_from_dict(data)
        except Exception:
            return None

    def save(self, session: Session) -> None:
        session.updated_at = datetime.utcnow()
        data = _session_to_dict(session)
        _atomic_write(self._session_path(session.id), json.dumps(data, indent=2))

    def delete(self, session_id: str) -> None:
        self._session_path(session_id).unlink(missing_ok=True)
        self._checkpoint_path(session_id).unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        return [p.stem for p in self._sessions_dir.glob("*.json")]

    # -- Richer queries ------------------------------------------------------

    def list_sessions(self, last_n_messages: int = 3) -> list[SessionSummary]:
        summaries = []
        for sid in self.list_ids():
            s = self.get(sid)
            if s is None:
                continue
            summaries.append(SessionSummary(
                session_id=s.id,
                current_state=s.state,
                is_running=(s.state == SessionState.running),
                cancel_requested=s.cancelled,
                message_count=len(s.messages),
                last_messages=s.messages[-last_n_messages:] if s.messages else [],
                error_message=s.error_message,
                created_at=s.created_at,
                updated_at=s.updated_at,
            ))
        return summaries

    def get_run_status(self, session_id: str) -> RunStatus | None:
        s = self.get(session_id)
        if s is None:
            return None
        return RunStatus(
            session_id=s.id,
            current_state=s.state,
            is_running=(s.state == SessionState.running),
            cancel_requested=s.cancelled,
            error_message=s.error_message,
            updated_at=s.updated_at,
        )

    # -- Targeted mutation ---------------------------------------------------

    def set_cancellation(self, session_id: str, value: bool) -> None:
        s = self.get(session_id)
        if s is not None:
            s.cancelled = value
            self.save(s)

    # -- Run ownership -------------------------------------------------------

    def try_acquire_run(self, session_id: str) -> bool:
        return self._ownership.try_acquire(session_id)

    def release_run(self, session_id: str) -> None:
        self._ownership.release(session_id)

    def is_run_owned(self, session_id: str) -> bool:
        return self._ownership.is_owned(session_id)

    # -- Checkpoints ---------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        path = self._checkpoint_path(cp.session_id)
        line = json.dumps(_checkpoint_to_dict(cp)) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        cps = self.list_checkpoints(session_id)
        return cps[-1] if cps else None

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        path = self._checkpoint_path(session_id)
        if not path.exists():
            return []
        checkpoints = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    checkpoints.append(_checkpoint_from_dict(json.loads(line)))
                except Exception:
                    pass
        return checkpoints

    def delete_checkpoints(self, session_id: str) -> None:
        self._checkpoint_path(session_id).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Serialization helpers (private)
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a sibling temp file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _dt_from_str(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def _tool_call_from_dict(d: dict[str, Any]) -> ToolCall:
    return ToolCall(id=d["id"], name=d["name"], arguments=d["arguments"])


def _tool_result_to_dict(tr: ToolResult) -> dict[str, Any]:
    return {"tool_call_id": tr.tool_call_id, "content": tr.content, "error": tr.error}


def _tool_result_from_dict(d: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=d["tool_call_id"],
        content=d["content"],
        error=d.get("error", False),
    )


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {
        "role": m.role.value,
        "content": m.content,
        "tool_calls": [_tool_call_to_dict(tc) for tc in m.tool_calls],
        "tool_results": [_tool_result_to_dict(tr) for tr in m.tool_results],
    }


def _message_from_dict(d: dict[str, Any]) -> Message:
    return Message(
        role=MessageRole(d["role"]),
        content=d.get("content", ""),
        tool_calls=[_tool_call_from_dict(tc) for tc in d.get("tool_calls", [])],
        tool_results=[_tool_result_from_dict(tr) for tr in d.get("tool_results", [])],
    )


def _session_to_dict(s: Session) -> dict[str, Any]:
    return {
        "id": s.id,
        "state": s.state.value,
        "messages": [_message_to_dict(m) for m in s.messages],
        "cancelled": s.cancelled,
        "metadata": s.metadata,
        "error_message": s.error_message,
        "pending_confirmation": [_tool_call_to_dict(tc) for tc in s.pending_confirmation],
        "created_at": _dt_to_str(s.created_at),
        "updated_at": _dt_to_str(s.updated_at),
    }


def _session_from_dict(d: dict[str, Any]) -> Session:
    return Session(
        id=d["id"],
        state=SessionState(d["state"]),
        messages=[_message_from_dict(m) for m in d.get("messages", [])],
        cancelled=d.get("cancelled", False),
        metadata=d.get("metadata", {}),
        error_message=d.get("error_message", ""),
        pending_confirmation=[_tool_call_from_dict(tc) for tc in d.get("pending_confirmation", [])],
        created_at=_dt_from_str(d["created_at"]) if "created_at" in d else datetime.utcnow(),
        updated_at=_dt_from_str(d["updated_at"]) if "updated_at" in d else datetime.utcnow(),
    )


def _checkpoint_to_dict(cp: Checkpoint) -> dict[str, Any]:
    return {
        "id": cp.id,
        "session_id": cp.session_id,
        "iteration": cp.iteration,
        "message_count": cp.message_count,
        "state": cp.state.value,
        "label": cp.label,
        "created_at": _dt_to_str(cp.created_at),
    }


def _checkpoint_from_dict(d: dict[str, Any]) -> Checkpoint:
    return Checkpoint(
        id=d["id"],
        session_id=d["session_id"],
        iteration=d["iteration"],
        message_count=d["message_count"],
        state=SessionState(d["state"]),
        label=d["label"],
        created_at=_dt_from_str(d["created_at"]) if "created_at" in d else datetime.utcnow(),
    )
