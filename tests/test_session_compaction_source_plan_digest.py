from __future__ import annotations

import pytest
from typing import Any, TypedDict

from session.store import SessionStore


class _CompactionKwargs(TypedDict):
    session_key: str
    trigger: str
    summary: str
    source_ref: str
    source_plan_digest: str
    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: list[str]
    retained_tail: list[dict[str, Any]]
    model_runtime_id: str
    model: str
    context_window: int
    threshold_tokens: int
    hard_input_tokens: int
    keep_recent_tokens: int
    tokens_before: int
    tokens_after: int
    summary_usage: dict[str, Any]
    generation: int
    parent_generation: int


def _persist_kwargs(session_key: str, digest: str) -> _CompactionKwargs:
    return {
        "session_key": session_key,
        "trigger": "soft_limit",
        "summary": "summary",
        "source_ref": "source:1",
        "source_plan_digest": digest,
        "source_from_seq": 0,
        "consolidated_through_seq": 0,
        "source_message_ids": ["session:digest:0"],
        "retained_tail": [
            {
                "id": "session:digest:0",
                "seq": 0,
                "unit_ref": "unit:1",
                "message": {"role": "user", "content": "source"},
            }
        ],
        "model_runtime_id": "runtime",
        "model": "model",
        "context_window": 100,
        "threshold_tokens": 74,
        "hard_input_tokens": 90,
        "keep_recent_tokens": 20,
        "tokens_before": 80,
        "tokens_after": 40,
        "summary_usage": {},
        "generation": 1,
        "parent_generation": 0,
    }


def _store(tmp_path) -> SessionStore:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="session:digest")
    store.insert_message(
        "session:digest",
        role="user",
        content="source",
        ts="2026-08-08T00:00:00+00:00",
        seq=0,
    )
    return store


def test_store_persists_and_reloads_canonical_digest(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        digest = "a" * 64
        row = store.persist_compaction(**_persist_kwargs("session:digest", digest))
        assert row.source_plan_digest == digest
        reopened = SessionStore(tmp_path / "sessions.db")
        try:
            loaded = reopened.get_compaction("session:digest", 1)
            assert loaded is not None
            assert loaded.source_plan_digest == digest
        finally:
            reopened.close()
    finally:
        store.close()


@pytest.mark.parametrize("digest", ("a" * 63, "g" * 64, ""))
def test_store_rejects_noncanonical_digest_at_write_boundary(tmp_path, digest: str) -> None:
    store = _store(tmp_path)
    try:
        with pytest.raises(ValueError, match="source_plan_digest"):
            store.persist_compaction(**_persist_kwargs("session:digest", digest))
        assert store.list_compactions("session:digest") == []
        assert store.get_compaction_head("session:digest").parent_generation == 0
    finally:
        store.close()
