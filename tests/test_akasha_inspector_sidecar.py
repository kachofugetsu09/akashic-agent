from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent.tools.recall_memory import render_memory_unavailable
from plugins.akasha.config import AkashaConfig
from plugins.akasha.infrastructure.sparse_index.builder import (
    AppendOnlyViolation,
    BuildConfig,
    build_sparse_index,
)
from plugins.akasha.infrastructure.sparse_index.schema import SCHEMA
from plugins.akasha.inspector import AkashaInspectorReader, _tool_recall_lanes


def _create_source(path: Path, tool_chain: str | None) -> None:
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE sessions (
            key TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_consolidated INTEGER NOT NULL DEFAULT 0,
            metadata TEXT
        );
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
        );
        CREATE TABLE message_embeddings (
            message_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(message_id, model)
        );
        """)
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, 0, NULL)",
        (
            "test:one",
            "2026-08-17T00:00:00+00:00",
            "2026-08-17T00:00:01+00:00",
        ),
    )
    connection.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "user:0",
                "test:one",
                0,
                "user",
                "hello",
                None,
                None,
                "2026-08-17T00:00:00+00:00",
            ),
            (
                "assistant:1",
                "test:one",
                1,
                "assistant",
                "answer",
                tool_chain,
                None,
                "2026-08-17T00:00:01+00:00",
            ),
        ],
    )
    connection.commit()
    connection.close()


def _build_source_sidecar(tmp_path: Path, tool_chain: str | None) -> Path:
    source = tmp_path / "sessions.db"
    index = tmp_path / "index.db"
    _create_source(source, tool_chain)
    build_sparse_index(source, index, BuildConfig())
    return index


def _reader(tmp_path: Path, index: Path) -> AkashaInspectorReader:
    memory = tmp_path / "akasha.db"
    sqlite3.connect(memory).close()
    return AkashaInspectorReader(
        memory_root=tmp_path,
        config=AkashaConfig(db_path="akasha.db", index_path=index.name),
    )


def test_sparse_projection_preserves_tool_chain_without_sessions_attach(
    tmp_path: Path,
) -> None:
    chain = json.dumps(
        [
            {
                "calls": [
                    {
                        "name": "recall_memory",
                        "status": "success",
                        "result": render_memory_unavailable("embedding unavailable"),
                    }
                ]
            }
        ]
    )
    index = _build_source_sidecar(tmp_path, chain)
    (tmp_path / "sessions.db").unlink()
    reader = _reader(tmp_path, index)

    with closing(reader._connect()) as connection:  # noqa: SLF001
        databases = {str(row[1]) for row in connection.execute("PRAGMA database_list")}
        projected = connection.execute(
            "SELECT assistant_tool_chain_json FROM sparse.sparse_turns"
        ).fetchone()[0]

    assert databases == {"main", "sparse"}
    assert projected == chain
    assert _tool_recall_lanes(projected) == ([], [])


def test_source_tool_chain_change_is_not_accepted_as_incremental_append(
    tmp_path: Path,
) -> None:
    index = _build_source_sidecar(tmp_path, "[]")
    source = tmp_path / "sessions.db"
    connection = sqlite3.connect(source)
    connection.execute(
        "UPDATE messages SET tool_chain = ? WHERE id = 'assistant:1'",
        (json.dumps([{"calls": [{"name": "shell"}]}]),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(AppendOnlyViolation, match="indexed turn changed"):
        build_sparse_index(source, index, BuildConfig())


def test_old_sparse_schema_fails_loud_with_rebuild_instruction(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.db"
    connection = sqlite3.connect(index)
    connection.executescript(
        SCHEMA.replace("    assistant_tool_chain_json TEXT,\n", "")
    )
    connection.execute("INSERT INTO metadata(key, value) VALUES ('index_version', '9')")
    connection.commit()
    connection.close()
    reader = _reader(tmp_path, index)

    with pytest.raises(ValueError, match="explicit rebuild is required"):
        with closing(reader._connect()):  # noqa: SLF001
            pass


def test_inspector_rejects_sidecars_outside_declared_memory_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="必须位于 memory root"):
        AkashaInspectorReader(
            memory_root=tmp_path / "memory",
            config=AkashaConfig(
                db_path="../sessions.db",
                index_path="memory/akasha-v2-index.db",
            ),
        )


def test_engine_and_inspector_share_declared_memory_path_contract(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "memory"

    direct = AkashaInspectorReader(
        memory_root=memory_root,
        config=AkashaConfig(
            db_path="akasha.db",
            index_path="akasha-v2-index.db",
        ),
    )
    historical = AkashaInspectorReader(
        memory_root=memory_root,
        config=AkashaConfig(),
    )

    assert direct.paths == historical.paths
    assert direct.paths.memory == memory_root / "akasha.db"
    assert direct.paths.index == memory_root / "akasha-v2-index.db"
    with pytest.raises(ValueError, match="必须位于 memory root"):
        AkashaInspectorReader(
            memory_root=memory_root,
            config=AkashaConfig(
                db_path="custom/akasha.db",
                index_path="akasha-v2-index.db",
            ),
        )


def test_inspector_rejects_symlinked_memory_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    memory_root = tmp_path / "workspace" / "memory"
    memory_root.parent.mkdir()
    memory_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="memory root 不能是符号链接"):
        AkashaInspectorReader(
            memory_root=memory_root,
            config=AkashaConfig(),
        )

    assert list(outside.iterdir()) == []
