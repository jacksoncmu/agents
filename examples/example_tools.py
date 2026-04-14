"""Example tools that can be registered with ToolRegistry.

Covers four representative patterns:
  calculator          — pure computation, no side-effects
  get_current_time    — no-param tool
  read_file           — file-read-like (sandboxed to a base directory)
  search_notes        — search-like (in-memory corpus, uses ToolContext logs)
  request_confirmation — requires human confirmation before acting
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

from agent.tools import ToolContext, ToolDefinition, ToolParam


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

def make_calculator() -> ToolDefinition:
    """Evaluate a safe arithmetic expression."""

    def handler(expression: str) -> str:
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


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------

def make_get_current_time() -> ToolDefinition:
    """Return the current UTC time — no parameters."""

    def handler() -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return ToolDefinition(
        name="get_current_time",
        description="Return the current UTC date and time.",
        params=[],
        handler=handler,
    )


# ---------------------------------------------------------------------------
# read_file  (file-read-like, sandboxed)
# ---------------------------------------------------------------------------

def make_read_file(base_dir: str | Path = ".") -> ToolDefinition:
    """
    Read a text file from within *base_dir*.

    Path traversal is prevented: any path that would escape the sandbox raises
    a validation-time error.  The handler uses ToolContext to log the resolved
    path so callers can audit what was accessed.
    """
    base = Path(base_dir).resolve()

    def handler(path: str, ctx: ToolContext) -> str:
        # Resolve relative to sandbox; reject traversal attempts
        target = (base / path).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Path {path!r} escapes the sandbox ({base})"
            )

        ctx.log(f"resolved path: {target}")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {path!r}")
        if not target.is_file():
            raise IsADirectoryError(f"Path is a directory: {path!r}")

        content = target.read_text(encoding="utf-8")
        ctx.log(f"read {len(content)} bytes")
        return content

    return ToolDefinition(
        name="read_file",
        description=(
            "Read the contents of a text file. "
            "Paths are relative to the configured sandbox directory."
        ),
        params=[
            ToolParam(
                name="path",
                type="string",
                description="Relative path to the file within the sandbox.",
            )
        ],
        handler=handler,
    )


# ---------------------------------------------------------------------------
# search_notes  (search-like, in-memory corpus)
# ---------------------------------------------------------------------------

def make_search_notes(notes: list[dict[str, Any]] | None = None) -> ToolDefinition:
    """
    Search an in-memory list of notes by keyword.

    Each note is a dict with at least a ``"text"`` key.  Optional ``"title"``
    and ``"tags"`` keys are also searched.  The handler logs the number of
    candidates checked via ToolContext.

    *notes* defaults to a small built-in corpus if not provided.
    """
    corpus: list[dict[str, Any]] = notes if notes is not None else _DEFAULT_NOTES

    def handler(query: str, ctx: ToolContext) -> str:
        ctx.log(f"searching {len(corpus)} notes for {query!r}")
        q = query.lower()
        hits: list[str] = []
        for note in corpus:
            haystack = " ".join(str(v) for v in note.values()).lower()
            if q in haystack:
                title = note.get("title", "(untitled)")
                hits.append(f"[{title}] {note.get('text', '')}")

        ctx.log(f"found {len(hits)} matches")
        if not hits:
            return f"No notes found matching {query!r}."
        return "\n".join(hits)

    return ToolDefinition(
        name="search_notes",
        description="Search notes by keyword. Returns matching note titles and text.",
        params=[
            ToolParam(
                name="query",
                type="string",
                description="Keyword or phrase to search for.",
            )
        ],
        handler=handler,
    )


_DEFAULT_NOTES: list[dict[str, Any]] = [
    {"title": "Shopping list", "text": "Milk, eggs, bread, butter", "tags": "groceries"},
    {"title": "Meeting notes", "text": "Discussed Q3 roadmap and tool runtime design", "tags": "work"},
    {"title": "Book recommendation", "text": "Read 'The Pragmatic Programmer'", "tags": "learning"},
    {"title": "Project idea", "text": "Build an agent runtime with modular tool registry", "tags": "work ideas"},
]


# ---------------------------------------------------------------------------
# request_confirmation  (confirmation-requiring)
# ---------------------------------------------------------------------------

def make_request_confirmation() -> ToolDefinition:
    """
    A placeholder for any action that must be explicitly approved before
    executing.  The engine detects ``requires_confirmation=True`` and
    transitions to ``waiting_for_confirmation``; the handler is never called
    until ``engine.resume(approved=True)`` is invoked.

    In this example the "action" is deleting a file by name.  The handler
    itself is deliberately simple — the confirmation gate is the feature.
    """

    def handler(filename: str, ctx: ToolContext) -> str:
        ctx.log(f"confirmed deletion of {filename!r}")
        # In a real implementation this would perform the destructive action.
        return f"File {filename!r} has been deleted."

    return ToolDefinition(
        name="delete_file",
        description=(
            "Permanently delete a file. Requires explicit user confirmation "
            "before the deletion is carried out."
        ),
        params=[
            ToolParam(
                name="filename",
                type="string",
                description="Name of the file to delete.",
            )
        ],
        handler=handler,
        requires_confirmation=True,
    )
