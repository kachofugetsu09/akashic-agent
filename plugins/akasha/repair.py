"""Explicit, recoverable Akasha embedding-space repair."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agent.plugin_composition import EmbeddingSpaceDescriptor, Embeddings
from session.embedding_store import MessageEmbeddingStore

from .application.rebuild import rebuild_memory
from .config import AkashaConfig, resolve_memory_path
from .infrastructure.loader import load_turns
from .infrastructure.sparse_index import (
    BuildConfig,
    audit_source_embeddings,
    build_sparse_index,
    sparse_index_state_sha256,
)
from .infrastructure.persistence import load_memory_state, sha256_file

_BATCH_SIZE = 100


@dataclass(frozen=True, slots=True)
class ReindexRequest:
    embedding_identity: str
    model_id: str
    dimensions: int
    requested_at: str


@dataclass(frozen=True, slots=True)
class ReindexResult:
    embedded_messages: int
    eligible_messages: int
    backup_dir: Path


def request_path(data_root: Path) -> Path:
    return data_root / "reindex-request.json"


def save_request(data_root: Path, descriptor: EmbeddingSpaceDescriptor) -> ReindexRequest:
    """Atomically record explicit intent without touching Session or memory data."""

    request = ReindexRequest(
        embedding_identity=descriptor.identity,
        model_id=descriptor.model_id,
        dimensions=descriptor.dimensions,
        requested_at=datetime.now(UTC).isoformat(),
    )
    path = request_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(asdict(request), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return request


def load_request(data_root: Path) -> ReindexRequest | None:
    path = request_path(data_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Akasha reindex request 必须是对象")
    return ReindexRequest(
        embedding_identity=_text(payload.get("embedding_identity"), "embedding_identity"),
        model_id=_text(payload.get("model_id"), "model_id"),
        dimensions=_positive_int(payload.get("dimensions"), "dimensions"),
        requested_at=_text(payload.get("requested_at"), "requested_at"),
    )


def finish_request(data_root: Path) -> None:
    path = request_path(data_root)
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


async def reindex(
    *,
    embeddings: Embeddings,
    descriptor: EmbeddingSpaceDescriptor,
    request: ReindexRequest,
    workspace: Path,
    data_root: Path,
    config: AkashaConfig,
    runtime_scope: Callable[[], AbstractAsyncContextManager[None]],
) -> ReindexResult:
    """Back up inputs, fill one new space, then publish verified sidecars."""

    if (
        request.embedding_identity != descriptor.identity
        or request.model_id != descriptor.model_id
        or request.dimensions != descriptor.dimensions
    ):
        raise RuntimeError("Akasha reindex 请求与当前默认 embedding 不一致，请重新确认")
    sessions = workspace / "sessions.db"
    if not sessions.is_file():
        raise RuntimeError(f"Akasha reindex 缺少 sessions.db: {sessions}")
    backup_dir = data_root / "backups" / "reindex" / uuid4().hex
    backup_dir.mkdir(parents=True, mode=0o700)
    os.chmod(backup_dir, 0o700)
    _backup_database(sessions, backup_dir / "sessions-before.db")

    build_config = BuildConfig(
        embedding_model=descriptor.identity,
        embedding_dimension=descriptor.dimensions,
    )
    embedded = await _fill_embeddings(
        embeddings=embeddings,
        descriptor=descriptor,
        sessions=sessions,
        build_config=build_config,
        runtime_scope=runtime_scope,
    )
    audit = audit_source_embeddings(sessions, build_config)
    if not audit.complete:
        raise RuntimeError(
            "Akasha reindex embedding 审计未通过: "
            f"eligible={audit.eligible_messages} valid={audit.valid_messages} "
            f"issues={len(audit.issues)}"
        )
    await _rebuild_sidecars(
        sessions=sessions,
        workspace=workspace,
        backup_dir=backup_dir,
        config=config,
        build_config=build_config,
    )
    manifest = {
        "schema_version": 1,
        "embedding_identity": descriptor.identity,
        "model_id": descriptor.model_id,
        "dimensions": descriptor.dimensions,
        "eligible_messages": audit.eligible_messages,
        "embedded_messages": embedded,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(backup_dir / "manifest.json", manifest)
    return ReindexResult(embedded, audit.eligible_messages, backup_dir)


async def _fill_embeddings(
    *,
    embeddings: Embeddings,
    descriptor: EmbeddingSpaceDescriptor,
    sessions: Path,
    build_config: BuildConfig,
    runtime_scope: Callable[[], AbstractAsyncContextManager[None]],
) -> int:
    audit = audit_source_embeddings(sessions, build_config)
    messages = _issue_messages(sessions, tuple(item.message_id for item in audit.issues))
    if not messages:
        return 0
    store = MessageEmbeddingStore(sessions)
    embedded = 0
    try:
        for offset in range(0, len(messages), _BATCH_SIZE):
            batch = messages[offset : offset + _BATCH_SIZE]
            async with runtime_scope():
                async with embeddings.bind(model_id=descriptor.model_id) as bound:
                    if bound.descriptor.identity != descriptor.identity:
                        raise RuntimeError("Akasha reindex 期间 embedding 空间已变化")
                    result = await bound.embed(
                        tuple(content for _, content in batch)
                    )
            if len(result.vectors) != len(batch):
                raise RuntimeError("Akasha reindex embedding 数量与输入不一致")
            for (message_id, content), vector in zip(
                batch, result.vectors, strict=True
            ):
                store.upsert(
                    message_id=message_id,
                    content=content,
                    model=descriptor.identity,
                    embedding=list(vector),
                )
                embedded += 1
    finally:
        store.close()
    return embedded


async def _rebuild_sidecars(
    *,
    sessions: Path,
    workspace: Path,
    backup_dir: Path,
    config: AkashaConfig,
    build_config: BuildConfig,
) -> None:
    memory_root = workspace / "memory"
    index = resolve_memory_path(memory_root, config.index_path)
    memory = resolve_memory_path(memory_root, config.db_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    memory.parent.mkdir(parents=True, exist_ok=True)
    existed: dict[str, bool] = {}
    for name, source in (("index", index), ("memory", memory)):
        existed[name] = source.exists()
        if source.exists():
            _backup_database(source, backup_dir / f"{name}-before.db")

    candidate_index = index.with_name(f".{index.name}.reindex-{uuid4().hex}.candidate")
    candidate_memory = memory.with_name(f".{memory.name}.reindex-{uuid4().hex}.candidate")
    try:
        _ = build_sparse_index(sessions, candidate_index, build_config)
        turns = load_turns(candidate_index)
        if turns:
            _ = rebuild_memory(
                candidate_index,
                candidate_memory,
                config=config.memory_config(),
                target_sequences=(),
            )
        _validate_sidecars(candidate_index, candidate_memory, config=config)
        try:
            if turns:
                os.replace(candidate_memory, memory)
            else:
                memory.unlink(missing_ok=True)
            _fsync_directory(memory.parent)
            os.replace(candidate_index, index)
            _fsync_directory(index.parent)
            _validate_sidecars(index, memory, config=config)
        except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError):
            _restore_target(
                memory,
                backup_dir / "memory-before.db",
                existed=existed["memory"],
            )
            _restore_target(
                index,
                backup_dir / "index-before.db",
                existed=existed["index"],
            )
            if existed["index"] and existed["memory"]:
                _bind_memory_to_index(memory, index)
                _validate_sidecars(index, memory, config=config)
            raise
    finally:
        candidate_index.unlink(missing_ok=True)
        candidate_memory.unlink(missing_ok=True)
        candidate_memory.with_suffix(candidate_memory.suffix + ".tmp").unlink(
            missing_ok=True
        )


def _issue_messages(sessions: Path, message_ids: tuple[str, ...]) -> list[tuple[str, str]]:
    if not message_ids:
        return []
    placeholders = ",".join("?" for _ in message_ids)
    with closing(sqlite3.connect(f"file:{sessions}?mode=ro", uri=True)) as db:
        rows = db.execute(
            f"SELECT id, content FROM messages WHERE id IN ({placeholders})",
            message_ids,
        ).fetchall()
    content = {str(row[0]): str(row[1] or "") for row in rows}
    missing = [message_id for message_id in message_ids if message_id not in content]
    if missing:
        raise RuntimeError(f"Akasha reindex 消息不存在: {missing[:5]}")
    return [(message_id, content[message_id]) for message_id in message_ids]


def _backup_database(source: Path, destination: Path) -> None:
    _sqlite_integrity(source)
    with (
        closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as source_db,
        closing(sqlite3.connect(destination)) as destination_db,
    ):
        source_db.backup(destination_db)
    os.chmod(destination, 0o600)
    _sqlite_integrity(destination)


def _restore_target(target: Path, backup: Path, *, existed: bool) -> None:
    if not existed:
        target.unlink(missing_ok=True)
        _fsync_directory(target.parent)
        return
    temporary = target.with_name(f".{target.name}.restore-{uuid4().hex}.tmp")
    _backup_database(backup, temporary)
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def _sqlite_integrity(path: Path) -> None:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        row = db.execute("PRAGMA integrity_check").fetchone()
    if row != ("ok",):
        raise RuntimeError(f"SQLite integrity_check 失败: {path}: {row}")


def _validate_sidecars(index: Path, memory: Path, *, config: AkashaConfig) -> None:
    """Open the exact reader path before accepting one published pair."""

    _sqlite_integrity(index)
    turns = load_turns(index)
    if not turns:
        if memory.exists():
            raise RuntimeError("空 Akasha index 不应带有 memory sidecar")
        return
    _sqlite_integrity(memory)
    _ = load_memory_state(
        memory,
        turns=turns,
        config=config.memory_config(),
        source_index_sha256=sha256_file(index),
        source_index_state_sha256=sparse_index_state_sha256(index),
    )


def _bind_memory_to_index(memory: Path, index: Path) -> None:
    """Repair pair identity after SQLite backup changes the restored file hash."""

    with closing(sqlite3.connect(memory)) as db, db:
        values = {
            "source_index_sha256": sha256_file(index),
            "source_index_state_sha256": sparse_index_state_sha256(index),
        }
        for key, value in values.items():
            changed = db.execute(
                "UPDATE metadata SET value = ? WHERE key = ?",
                (value, key),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"Akasha memory 缺少配对身份: {key}")
    _sqlite_integrity(memory)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"Akasha reindex {field} 无效")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Akasha reindex {field} 必须是正整数")
    return value
