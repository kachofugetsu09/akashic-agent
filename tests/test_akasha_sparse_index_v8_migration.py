# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

import pytest

import migrations.akasha_sparse_index_v8.migration as migration
from migrations.akasha_sparse_index_v8.migration import (
    MigrationContext,
    _apply,
    _assess,
    _revert,
    _verify,
)
from plugins.akasha.infrastructure.persistence import sha256_file

_ACTIVE_CONFIG = """\
[llm]
main = "test_main"

[llm.runtimes.test_main]
provider = "openai"
model = "test-model"
api_key = "test-key"
context_window = 64000

[agent]
system_prompt = "test"

[memory]
enabled = true
engine = "akasha"

[memory.embedding]
model = "embedding-model"
output_dimensionality = 2
"""

_DEFAULT_CONFIG = _ACTIVE_CONFIG.replace(
    'engine = "akasha"',
    'engine = "default"',
)


def _context(tmp_path: Path, config: str = _ACTIVE_CONFIG) -> MigrationContext:
    config_path = tmp_path / "config.toml"
    config_path.write_text(config, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return MigrationContext(
        config_path=config_path,
        workspace=workspace,
        migration_commit="a" * 40,
        backup_dir=tmp_path / "backups" / "akasha-v8",
    )


def _create_sessions(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""
            CREATE TABLE sessions (
                key               TEXT PRIMARY KEY,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                last_consolidated INTEGER NOT NULL DEFAULT 0,
                metadata          TEXT
            )
            """)
        connection.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_chain TEXT,
                extra TEXT,
                ts TEXT NOT NULL,
                UNIQUE(session_key, seq)
            )
            """)
        connection.execute("""
            CREATE TABLE message_embeddings (
                message_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(message_id, model)
            )
            """)


def _append_reinforced_turn(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
            VALUES ('test:one', '2026-07-30T00:00:00+00:00', '2026-07-30T00:00:00+00:00', 0, NULL)
            """
        )
    messages = (
        (
            "message:user",
            0,
            "user",
            "请记住昨晚睡得很好",
            json.dumps({"akasha_reinforce": {"boost": 2.5}}),
            (1.0, 0.0),
        ),
        (
            "message:assistant",
            1,
            "assistant",
            "好的，我记住了。",
            None,
            (0.0, 1.0),
        ),
    )
    with closing(sqlite3.connect(path)) as connection, connection:
        for message_id, seq, role, content, extra, vector in messages:
            connection.execute(
                """
                INSERT INTO messages
                VALUES (?, 'test:one', ?, ?, ?, NULL, ?, ?)
                """,
                (
                    message_id,
                    seq,
                    role,
                    content,
                    extra,
                    f"2026-07-30T00:00:0{seq}+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO message_embeddings
                VALUES (?, ?, 'embedding-model', ?, 2, ?, ?)
                """,
                (
                    message_id,
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    struct.pack("<ff", *vector),
                    "2026-07-30T00:00:00+00:00",
                    "2026-07-30T00:00:00+00:00",
                ),
            )


def _create_legacy_sidecar(path: Path, **metadata: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.execute("CREATE TABLE legacy_payload (value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO legacy_payload VALUES ('preserve-before-rebuild')"
        )


def _metadata(path: Path) -> dict[str, str]:
    with closing(sqlite3.connect(path)) as connection:
        return dict(connection.execute("SELECT key, value FROM metadata"))


def test_v7_rebuilds_index_and_graph_from_sessions_once(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    sessions = context.workspace / "sessions.db"
    index = context.workspace / "memory" / "akasha-v2-index.db"
    memory = context.workspace / "memory" / "akasha.db"
    _create_sessions(sessions)
    _append_reinforced_turn(sessions)
    _create_legacy_sidecar(index, index_version="7")
    _create_legacy_sidecar(
        memory,
        sparse_index_index_version="7",
        turn_count="1",
    )
    sessions_before = sha256_file(sessions)

    assert _assess(context) == {"status": "needed"}
    _apply(context)
    _verify(context)

    assert sha256_file(sessions) == sessions_before
    assert _metadata(index)["index_version"] == "8"
    graph_metadata = _metadata(memory)
    assert graph_metadata["sparse_index_index_version"] == "8"
    assert graph_metadata["source_index_sha256"] == sha256_file(index)
    assert graph_metadata["turn_count"] == "1"
    with closing(sqlite3.connect(memory)) as connection:
        feedback = connection.execute("""
            SELECT event_id, action, target_turn_node_id, boost
            FROM feedback_events
            """).fetchall()
    assert feedback == [(0, "remember", 0, 2.5)]
    manifest = json.loads(
        (context.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidate"]["turnCount"] == 1
    assert (context.backup_dir / "index-before.db").is_file()
    assert (context.backup_dir / "memory-before.db").is_file()

    identities = (
        sha256_file(index),
        sha256_file(memory),
        index.stat().st_mtime_ns,
        memory.stat().st_mtime_ns,
    )
    assert _assess(context) == {"status": "satisfied"}
    _verify(context)
    assert identities == (
        sha256_file(index),
        sha256_file(memory),
        index.stat().st_mtime_ns,
        memory.stat().st_mtime_ns,
    )


def test_revert_restores_both_legacy_sidecars(tmp_path: Path) -> None:
    context = _context(tmp_path)
    sessions = context.workspace / "sessions.db"
    index = context.workspace / "memory" / "akasha-v2-index.db"
    memory = context.workspace / "memory" / "akasha.db"
    _create_sessions(sessions)
    _append_reinforced_turn(sessions)
    _create_legacy_sidecar(index, index_version="7")
    _create_legacy_sidecar(
        memory,
        sparse_index_index_version="7",
        turn_count="1",
    )

    _apply(context)
    _revert(context)

    assert _metadata(index)["index_version"] == "7"
    assert _metadata(memory)["sparse_index_index_version"] == "7"
    with closing(sqlite3.connect(index)) as connection:
        assert connection.execute("SELECT value FROM legacy_payload").fetchone() == (
            "preserve-before-rebuild",
        )
    with closing(sqlite3.connect(memory)) as connection:
        assert connection.execute("SELECT value FROM legacy_payload").fetchone() == (
            "preserve-before-rebuild",
        )


def test_non_akasha_does_not_read_or_touch_sidecars(tmp_path: Path) -> None:
    context = _context(tmp_path, _DEFAULT_CONFIG)
    index = context.workspace / "memory" / "akasha-v2-index.db"
    memory = context.workspace / "memory" / "akasha.db"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"not sqlite")
    memory.write_bytes(b"also not sqlite")

    assert _assess(context) == {"status": "satisfied"}
    _verify(context)

    assert index.read_bytes() == b"not sqlite"
    assert memory.read_bytes() == b"also not sqlite"
    assert not context.backup_dir.exists()


def test_unknown_index_version_blocks_downgrade(tmp_path: Path) -> None:
    context = _context(tmp_path)
    sessions = context.workspace / "sessions.db"
    index = context.workspace / "memory" / "akasha-v2-index.db"
    _create_sessions(sessions)
    _create_legacy_sidecar(index, index_version="9")

    assessment = _assess(context)

    assert assessment["status"] == "blocked"
    assert "版本: 9" in assessment["reason"]


def test_v8_index_with_unpaired_graph_is_rebuilt(tmp_path: Path) -> None:
    context = _context(tmp_path)
    sessions = context.workspace / "sessions.db"
    index = context.workspace / "memory" / "akasha-v2-index.db"
    memory = context.workspace / "memory" / "akasha.db"
    _create_sessions(sessions)
    _append_reinforced_turn(sessions)
    _create_legacy_sidecar(index, index_version="7")
    _create_legacy_sidecar(memory, sparse_index_index_version="7")
    _apply(context)
    with closing(sqlite3.connect(memory)) as connection, connection:
        connection.execute("""
            UPDATE metadata
            SET value = '7'
            WHERE key = 'sparse_index_index_version'
            """)

    assert _assess(context) == {"status": "needed"}


def test_publish_failure_restores_both_legacy_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    sessions = context.workspace / "sessions.db"
    index = context.workspace / "memory" / "akasha-v2-index.db"
    memory = context.workspace / "memory" / "akasha.db"
    _create_sessions(sessions)
    _append_reinforced_turn(sessions)
    _create_legacy_sidecar(index, index_version="7")
    _create_legacy_sidecar(memory, sparse_index_index_version="7")
    real_replace = migration.os.replace

    def fail_index_publish(source: Path, target: Path) -> None:
        if target == index and ".candidate" in str(source):
            raise OSError("injected index publication failure")
        real_replace(source, target)

    monkeypatch.setattr(migration.os, "replace", fail_index_publish)

    with pytest.raises(OSError, match="publication failure"):
        _apply(context)

    assert _metadata(index)["index_version"] == "7"
    assert _metadata(memory)["sparse_index_index_version"] == "7"
