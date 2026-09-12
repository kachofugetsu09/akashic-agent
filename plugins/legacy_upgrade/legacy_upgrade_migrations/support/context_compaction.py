"""Stable IDs used by the historical compaction migration."""
from __future__ import annotations


def compaction_scope_id(session_key: str) -> str:
    return f"session:{session_key}:compaction"


def compaction_source_ref(session_key: str, generation: int) -> str:
    return f"{compaction_scope_id(session_key)}:{generation}"
