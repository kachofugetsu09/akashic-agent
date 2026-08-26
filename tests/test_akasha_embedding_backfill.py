from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from agent.migrations import akasha_embedding_backfill as migration
from plugins.akasha.infrastructure.sparse_index import (
    BuildConfig,
    audit_source_embeddings,
)


class _Embedder:
    MAX_BATCH = 10
    calls: list[list[str]] = []

    def __init__(self, **_: object) -> None:
        pass

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    async def aclose(self) -> None:
        return None


def _write_config(path: Path) -> None:
    path.write_text(
        """
[llm]
main = "test_main"

[llm.runtimes.test_main]
provider = "openai"
model = "chat-model"
api_key = "chat-key"
base_url = "https://chat.invalid/v1"
input_modalities = ["text"]

[memory]
enabled = true
engine = "default"

[memory.embedding]
model = "embedding-model"
api_key = "embedding-key"
base_url = "https://embedding.invalid/v1"
output_dimensionality = 2
""".strip() + "\n",
        encoding="utf-8",
    )


def _create_sessions(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE sessions (key TEXT PRIMARY KEY, created_at TEXT, "
            "updated_at TEXT, last_consolidated INTEGER, metadata TEXT)"
        )
        connection.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, session_key TEXT NOT NULL, "
            "seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT, tool_chain TEXT, "
            "extra TEXT, ts TEXT NOT NULL, UNIQUE(session_key, seq))"
        )
        connection.executemany(
            "INSERT INTO sessions VALUES (?, '2026-08-01', '2026-08-01', 0, NULL)",
            [("chat:one",), ("scheduler:job",), ("chat:interrupted",)],
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, NULL, ?, '2026-08-01')",
            [
                ("u1", "chat:one", 0, "user", "remember me", None),
                ("a1", "chat:one", 1, "assistant", "I will", None),
                (
                    "su",
                    "scheduler:job",
                    0,
                    "user",
                    "background",
                    '{"effects":{"post_commit":"suppress"}}',
                ),
                ("sa", "scheduler:job", 1, "assistant", "done", None),
                ("iu", "chat:interrupted", 0, "user", "unfinished", None),
                ("ia", "chat:interrupted", 1, "assistant", "[interrupted]", None),
            ],
        )


def test_enabled_legacy_history_backfills_only_eligible_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    _write_config(config)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    legacy = workspace / "memory/memory2.db"
    legacy.parent.mkdir()
    legacy.write_bytes(b"retired-classic-database")
    _Embedder.calls = []
    monkeypatch.setattr(migration, "Embedder", _Embedder)

    async def run_with_fake_resources(*, sessions_path: Path, host: object) -> int:
        return await migration._backfill(  # noqa: SLF001
            sessions_path=sessions_path,
            host=cast(Any, host),
            requester=cast(Any, object()),
        )

    monkeypatch.setattr(migration, "_backfill_with_resources", run_with_fake_resources)

    result = migration.backfill_akasha_message_embeddings(
        config_path=config,
        migrated_config=config.read_bytes().replace(b'engine = "default"\n', b""),
        workspace=workspace,
    )

    assert result.eligible_messages == 2
    assert result.embedded_messages == 2
    assert _Embedder.calls == [["remember me", "I will"]]
    assert result.backup_path is not None and result.backup_path.is_file()
    assert legacy.read_bytes() == b"retired-classic-database"
    audit = audit_source_embeddings(
        sessions,
        BuildConfig(embedding_model="embedding-model", embedding_dimension=2),
    )
    assert audit.complete
    assert audit.excluded_interrupted_turns == 1
    assert audit.excluded_memory_turns == 1


def test_complete_history_is_idempotent_without_backup_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    _write_config(config)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    first = workspace / "backups/preexisting/sessions.db"
    first.parent.mkdir(parents=True)
    first.write_bytes(hashlib.sha256(b"evidence").digest())
    _Embedder.calls = []
    monkeypatch.setattr(migration, "Embedder", _Embedder)

    async def run_with_fake_resources(*, sessions_path: Path, host: object) -> int:
        return await migration._backfill(  # noqa: SLF001
            sessions_path=sessions_path,
            host=cast(Any, host),
            requester=cast(Any, object()),
        )

    monkeypatch.setattr(migration, "_backfill_with_resources", run_with_fake_resources)
    _ = migration.backfill_akasha_message_embeddings(
        config_path=config,
        migrated_config=config.read_bytes().replace(b'engine = "default"\n', b""),
        workspace=workspace,
    )
    _Embedder.calls = []

    result = migration.backfill_akasha_message_embeddings(
        config_path=config,
        migrated_config=config.read_bytes().replace(b'engine = "default"\n', b""),
        workspace=workspace,
    )

    assert result.embedded_messages == 0
    assert result.backup_path is None
    assert _Embedder.calls == []
    roots = list((workspace / "backups/backfill-akasha-message-embeddings").iterdir())
    assert len(roots) == 1
