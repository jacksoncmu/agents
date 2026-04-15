"""Blob storage abstraction for oversized tool outputs.

When a tool produces output that exceeds the per-type character cap, the
full payload is written to the blob store and a short reference ID is
returned in its place.  This keeps context windows clean while still
making the full output retrievable.

Usage::

    store = InMemoryBlobStore()
    ref = store.put(big_string, metadata={"tool": "shell", "session": "s1"})
    # ref looks like "blob:abc123"

    content = store.get(ref)
    assert content == big_string
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


BLOB_REF_PREFIX = "blob:"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlobRecord:
    """Metadata + content for one stored blob."""
    ref: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BlobStore(ABC):
    """Backend-agnostic blob storage."""

    @abstractmethod
    def put(self, content: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Store *content* and return a reference ID (``blob:<id>``)."""
        ...

    @abstractmethod
    def get(self, ref: str) -> str | None:
        """Retrieve content by reference.  Returns None if not found."""
        ...

    @abstractmethod
    def get_record(self, ref: str) -> BlobRecord | None:
        """Retrieve full record (content + metadata).  None if missing."""
        ...

    @abstractmethod
    def delete(self, ref: str) -> bool:
        """Delete a blob.  Returns True if it existed."""
        ...

    @abstractmethod
    def list_refs(self) -> list[str]:
        """List all stored reference IDs."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryBlobStore(BlobStore):
    """Simple dict-backed blob store for testing and single-process use."""

    def __init__(self) -> None:
        self._blobs: dict[str, BlobRecord] = {}

    def put(self, content: str, *, metadata: dict[str, Any] | None = None) -> str:
        ref = f"{BLOB_REF_PREFIX}{uuid.uuid4().hex[:12]}"
        self._blobs[ref] = BlobRecord(
            ref=ref,
            content=content,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
        )
        return ref

    def get(self, ref: str) -> str | None:
        record = self._blobs.get(ref)
        return record.content if record else None

    def get_record(self, ref: str) -> BlobRecord | None:
        return self._blobs.get(ref)

    def delete(self, ref: str) -> bool:
        return self._blobs.pop(ref, None) is not None

    def list_refs(self) -> list[str]:
        return list(self._blobs.keys())

    @property
    def count(self) -> int:
        return len(self._blobs)
