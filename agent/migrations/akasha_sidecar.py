from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import tomllib
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agent.model_runtime.store import ModelRegistryStore
from agent.plugins.manifest import builtin_plugin_data_dir
from plugins.akasha.application.rebuild import rebuild_memory
from plugins.akasha.config import AkashaConfig, load_akasha_config, resolve_memory_path
from plugins.akasha.infrastructure.loader import load_turns
from plugins.akasha.infrastructure.persistence import load_memory_state
from plugins.akasha.infrastructure.sparse_index import (
    BuildConfig,
    audit_source_embeddings,
    build_sparse_index,
)
from plugins.akasha.infrastructure.sparse_index.schema import INDEX_VERSION


@dataclass(frozen=True)
class AkashaSidecars:
    sessions: Path
    index: Path
    memory: Path


def rebuild_akasha_sidecars(
    *,
    config_path: Path,
    workspace: Path,
    backup_dir: Path,
    accepted_versions: set[str],
    force: bool = False,
) -> bool:
    """Build, verify, and atomically publish the current Akasha sidecars."""

    # 1. Resolve the same paths and model identity used by the runtime.
    plugin = load_akasha_config(
        builtin_plugin_data_dir("akasha", workspace) / "config.local.toml"
    )
    paths = AkashaSidecars(
        sessions=workspace / "sessions.db",
        index=resolve_memory_path(workspace / "memory", plugin.index_path),
        memory=resolve_memory_path(workspace / "memory", plugin.db_path),
    )
    if not paths.sessions.is_file():
        raise RuntimeError(f"Akasha 重建缺少权威源: {paths.sessions}")
    if not paths.index.exists() and not paths.memory.exists():
        return False
    if paths.index.exists():
        version = _index_version(paths.index)
        if not force and version == INDEX_VERSION and _pair_is_valid(paths, plugin):
            return False
        if version not in accepted_versions | {INDEX_VERSION}:
            raise RuntimeError(f"不支持 Akasha sparse index 版本: {version}")

    # 2. Audit the immutable source before creating any replacement candidate.
    embedding_model, embedding_dimension = _legacy_embedding_config(
        config_path, workspace
    )
    build_config = BuildConfig(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
    )
    audit = audit_source_embeddings(paths.sessions, build_config)
    if not audit.complete:
        examples: list[dict[str, object]] = [
            {
                "messageId": issue.message_id,
                "sessionKey": issue.session_key,
                "seq": issue.seq,
                "reason": issue.reason,
            }
            for issue in audit.issues[:5]
        ]
        raise RuntimeError(
            "Akasha 重建前 embedding 审计失败: "
            + json.dumps(
                {
                    "eligibleMessages": audit.eligible_messages,
                    "validMessages": audit.valid_messages,
                    "issueCount": len(audit.issues),
                    "examples": examples,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    # 3. Preserve the current pair before building and publishing replacements.
    records = _backup_pair(paths, backup_dir)
    candidate_index = paths.index.with_name(
        f".{paths.index.name}.migration-{uuid4().hex}.candidate"
    )
    candidate_memory = paths.memory.with_name(
        f".{paths.memory.name}.migration-{uuid4().hex}.candidate"
    )
    try:
        _ = build_sparse_index(paths.sessions, candidate_index, build_config)
        turns = load_turns(candidate_index)
        if turns:
            _ = rebuild_memory(
                candidate_index,
                candidate_memory,
                config=plugin.memory_config(),
                target_sequences=(),
            )
            _verify_pair(candidate_index, candidate_memory, plugin)
        else:
            _verify_pair(candidate_index, None, plugin)
        _write_manifest(
            backup_dir,
            paths,
            records,
            candidate_index,
            candidate_memory if turns else None,
            len(turns),
        )

        try:
            if turns:
                os.replace(candidate_memory, paths.memory)
            else:
                paths.memory.unlink(missing_ok=True)
            _fsync_directory(paths.memory.parent)
            os.replace(candidate_index, paths.index)
            _fsync_directory(paths.index.parent)
            _verify_pair(paths.index, paths.memory if turns else None, plugin)
        except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError):
            _restore_pair(paths, records, backup_dir)
            raise
    finally:
        candidate_index.unlink(missing_ok=True)
        candidate_memory.unlink(missing_ok=True)
        candidate_memory.with_suffix(candidate_memory.suffix + ".tmp").unlink(
            missing_ok=True
        )
    return True


def _legacy_embedding_config(config_path: Path, workspace: Path) -> tuple[str, int]:
    """Read only the retired embedding formats needed to rebuild old sidecars."""

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    memory = raw.get("memory")
    embedding = memory.get("embedding") if isinstance(memory, dict) else None
    if not isinstance(embedding, dict):
        raise RuntimeError("Akasha 重建缺少历史 embedding 配置")
    model = str(embedding.get("model") or "").strip()
    dimension = embedding.get("output_dimensionality")
    if model and isinstance(dimension, int) and not isinstance(dimension, bool):
        return model, dimension
    model_ref = str(embedding.get("model_ref") or "").strip()
    stored = ModelRegistryStore.for_workspace(workspace).get_embedding_model(model_ref)
    if stored is None:
        raise RuntimeError(f"Akasha 重建找不到历史 embedding 模型: {model_ref}")
    return stored.model, stored.dimensions


def _pair_is_valid(paths: AkashaSidecars, plugin: AkashaConfig) -> bool:
    try:
        _verify_pair(
            paths.index, paths.memory if paths.memory.exists() else None, plugin
        )
    except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError):
        return False
    return True


def _verify_pair(index: Path, memory: Path | None, plugin: AkashaConfig) -> None:
    """Exercise the complete read path for one candidate pair."""

    _sqlite_integrity(index)
    turns = load_turns(index)
    if not turns:
        if memory is not None and memory.exists():
            raise ValueError("空 Akasha index 不应保留 graph snapshot")
        return
    if memory is None or not memory.exists():
        raise ValueError("非空 Akasha index 缺少 graph snapshot")
    _sqlite_integrity(memory)
    config = plugin.memory_config()
    _ = load_memory_state(
        memory,
        turns=turns,
        config=config,
        source_index_sha256=_sha256_file(index),
    )


def _backup_pair(
    paths: AkashaSidecars,
    backup_dir: Path,
) -> dict[str, tuple[bool, str | None]]:
    backup_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_dir, 0o700)
    records: dict[str, tuple[bool, str | None]] = {}
    for name, source in (("index", paths.index), ("memory", paths.memory)):
        if not source.exists():
            records[name] = (False, None)
            continue
        destination = backup_dir / f"{name}-before.db"
        _backup_database(source, destination)
        records[name] = (True, destination.name)
    return records


def _backup_database(source: Path, destination: Path) -> None:
    _sqlite_integrity(source)
    with (
        closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as source_db,
        closing(sqlite3.connect(destination)) as destination_db,
    ):
        source_db.backup(destination_db)
    os.chmod(destination, 0o600)
    _sqlite_integrity(destination)


def _restore_pair(
    paths: AkashaSidecars,
    records: dict[str, tuple[bool, str | None]],
    backup_dir: Path,
) -> None:
    for name, target in (("memory", paths.memory), ("index", paths.index)):
        existed, backup_name = records[name]
        if not existed:
            target.unlink(missing_ok=True)
            _fsync_directory(target.parent)
            continue
        if backup_name is None:
            raise RuntimeError(f"Akasha backup record 缺少文件: {name}")
        backup = backup_dir / backup_name
        temporary = target.with_name(f".{target.name}.restore-{uuid4().hex}.tmp")
        _backup_database(backup, temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)


def _write_manifest(
    backup_dir: Path,
    paths: AkashaSidecars,
    records: dict[str, tuple[bool, str | None]],
    candidate_index: Path,
    candidate_memory: Path | None,
    turn_count: int,
) -> None:
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "indexVersion": INDEX_VERSION,
        "workspace": str(paths.sessions.parent),
        "turnCount": turn_count,
        "candidateIndexSha256": _sha256_file(candidate_index),
        "candidateMemorySha256": (
            _sha256_file(candidate_memory) if candidate_memory is not None else None
        ),
        "targets": {
            name: {
                "path": str(getattr(paths, name)),
                "existed": existed,
                "backup": backup,
                "backupSha256": (
                    _sha256_file(backup_dir / backup) if backup is not None else None
                ),
            }
            for name, (existed, backup) in records.items()
        },
    }
    _atomic_write_json(backup_dir / "manifest.json", payload)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _ = stream.write(
                (
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                ).encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _index_version(path: Path) -> str:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'index_version'"
        ).fetchone()
    if row is None:
        raise ValueError("Akasha sparse index 缺少 index_version")
    return str(row[0])


def _sqlite_integrity(path: Path) -> None:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    if rows != [("ok",)]:
        raise sqlite3.IntegrityError(
            f"SQLite integrity_check 失败: path={path} rows={rows[:3]}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
