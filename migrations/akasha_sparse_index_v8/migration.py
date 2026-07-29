from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.config import Config
from agent.plugins.manifest import builtin_plugin_data_dir
from plugins.akasha.application.rebuild import rebuild_memory
from plugins.akasha.config import (
    AkashaConfig,
    load_akasha_config,
    resolve_workspace_path,
)
from plugins.akasha.infrastructure.loader import load_turns
from plugins.akasha.infrastructure.persistence import (
    load_memory_state,
    logical_state_sha256,
    sha256_file,
)
from plugins.akasha.infrastructure.sparse_index import (
    BuildConfig,
    audit_source_embeddings,
    build_sparse_index,
)
from plugins.akasha.infrastructure.sparse_index.schema import INDEX_VERSION

_LEGACY_INDEX_VERSION = "7"
_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class MigrationContext:
    config_path: Path
    workspace: Path
    migration_commit: str
    backup_dir: Path | None


@dataclass(frozen=True)
class AkashaPaths:
    sessions: Path
    index: Path
    memory: Path


@dataclass(frozen=True)
class ActiveAkasha:
    paths: AkashaPaths
    plugin_config: AkashaConfig
    build_config: BuildConfig


@dataclass(frozen=True)
class CandidateState:
    index: Path
    memory: Path | None
    turn_count: int
    logical_state_sha256: str | None


def _parse_args() -> tuple[str, MigrationContext]:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "action",
        choices=("assess", "apply", "verify", "revert"),
    )
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--workspace", type=Path, required=True)
    _ = parser.add_argument("--migration-commit", required=True)
    _ = parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    return str(args.action), MigrationContext(
        config_path=Path(args.config).expanduser().resolve(),
        workspace=Path(args.workspace).expanduser().resolve(),
        migration_commit=str(args.migration_commit),
        backup_dir=Path(args.backup_dir).resolve() if args.backup_dir else None,
    )


def _memory_engine_selection(config_path: Path) -> tuple[bool, str]:
    """Read only the host-owned fields that decide memory-engine ownership."""

    if not config_path.exists():
        return False, ""
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_memory_value = payload.get("memory", {})
    if not isinstance(raw_memory_value, dict):
        raise ValueError("memory 配置必须是 table")
    raw_memory = cast(dict[str, object], raw_memory_value)
    enabled = raw_memory.get("enabled", False)
    engine = raw_memory.get("engine", "")
    if not isinstance(enabled, bool):
        raise ValueError("memory.enabled 必须是 bool")
    if not isinstance(engine, str):
        raise ValueError("memory.engine 必须是 string")
    return enabled, engine.strip()


def _active_akasha(context: MigrationContext) -> ActiveAkasha | None:
    """Resolve the exact runtime configuration only for an active Akasha engine."""

    # 1. Non-Akasha installations must remain completely opaque to this bundle.
    enabled, engine = _memory_engine_selection(context.config_path)
    if not enabled or engine != "akasha":
        return None

    # 2. Reuse the runtime loaders so rebuild parameters cannot drift.
    host_config = Config.load(context.config_path, workspace=context.workspace)
    plugin_path = (
        builtin_plugin_data_dir("akasha", context.workspace) / "config.local.toml"
    )
    plugin_config = load_akasha_config(plugin_path)
    paths = AkashaPaths(
        sessions=context.workspace / "sessions.db",
        index=resolve_workspace_path(
            context.workspace,
            plugin_config.index_path,
        ),
        memory=resolve_workspace_path(
            context.workspace,
            plugin_config.db_path,
        ),
    )
    return ActiveAkasha(
        paths=paths,
        plugin_config=plugin_config,
        build_config=BuildConfig(
            embedding_model=host_config.memory.embedding.model,
            embedding_dimension=(host_config.memory.embedding.output_dimensionality),
        ),
    )


def _sqlite_metadata(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT key, value FROM metadata ORDER BY key"
        ).fetchall()
    finally:
        connection.close()
    return {str(key): str(value) for key, value in rows}


def _index_version(path: Path) -> str:
    value = _sqlite_metadata(path).get("index_version")
    if value is None:
        raise ValueError("Akasha sparse index 缺少 index_version")
    return value


def _sqlite_integrity(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise sqlite3.IntegrityError(
            f"SQLite integrity_check 失败: path={path} rows={rows[:3]}"
        )


def _verify_pair(
    active: ActiveAkasha,
    *,
    index_path: Path,
    memory_path: Path | None,
) -> dict[str, str | int | None]:
    """Validate one complete index/graph pair through the runtime read path."""

    # 1. Validate the current index schema and materialize every canonical turn.
    _sqlite_integrity(index_path)
    turns = load_turns(index_path)
    source_hash = sha256_file(index_path)
    if not turns:
        if memory_path is not None and memory_path.exists():
            raise ValueError("空 Akasha index 不应保留 graph snapshot")
        return {
            "turnCount": 0,
            "sourceIndexSha256": source_hash,
            "logicalStateSha256": None,
        }

    # 2. Validate graph identity, feedback bindings, and learned-state schema.
    if memory_path is None or not memory_path.exists():
        raise ValueError("非空 Akasha index 缺少 graph snapshot")
    _sqlite_integrity(memory_path)
    metadata = _sqlite_metadata(memory_path)
    if metadata.get("sparse_index_index_version") != INDEX_VERSION:
        raise ValueError("Akasha graph 未绑定 sparse index v8")
    _ = load_memory_state(
        memory_path,
        turns=turns,
        config=active.plugin_config.memory_config(),
        source_index_sha256=source_hash,
    )
    return {
        "turnCount": len(turns),
        "sourceIndexSha256": source_hash,
        "logicalStateSha256": logical_state_sha256(memory_path),
    }


def _current_pair_is_valid(active: ActiveAkasha) -> bool:
    paths = active.paths
    try:
        _ = _verify_pair(
            active,
            index_path=paths.index,
            memory_path=paths.memory if paths.memory.exists() else None,
        )
    except (OSError, sqlite3.DatabaseError, ValueError):
        return False
    return True


def _assess(context: MigrationContext) -> dict[str, str]:
    """Classify this installation without mutating source or derived state."""

    try:
        # 1. Scope the migration to the active Akasha engine and existing state.
        active = _active_akasha(context)
        if active is None:
            return {"status": "satisfied"}
        paths = active.paths
        if not paths.index.exists() and not paths.memory.exists():
            return {"status": "satisfied"}
        if not paths.sessions.is_file():
            return {
                "status": "blocked",
                "reason": f"Akasha 重建缺少权威源: {paths.sessions}",
            }
        if not paths.index.exists():
            return {"status": "needed"}

        # 2. Rebuild known v7 state, reject unknown versions, and validate v8.
        version = _index_version(paths.index)
        if version == _LEGACY_INDEX_VERSION:
            return {"status": "needed"}
        if version != INDEX_VERSION:
            return {
                "status": "blocked",
                "reason": f"不支持 Akasha sparse index 版本: {version}",
            }
        return {"status": ("satisfied" if _current_pair_is_valid(active) else "needed")}
    except (OSError, sqlite3.DatabaseError, ValueError) as error:
        return {"status": "blocked", "reason": str(error)}


def _audit_source(active: ActiveAkasha) -> None:
    audit = audit_source_embeddings(
        active.paths.sessions,
        active.build_config,
    )
    if audit.complete:
        return
    examples: list[dict[str, str | int]] = [
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


def _candidate_path(target: Path, token: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f".{target.name}.migration-{token}.candidate")


def _build_candidates(active: ActiveAkasha) -> CandidateState:
    """Build and fully validate new derived state without touching live files."""

    token = uuid4().hex
    index = _candidate_path(active.paths.index, token)
    memory = _candidate_path(active.paths.memory, token)
    try:
        # 1. Rebuild the sparse index from immutable source rows.
        _ = build_sparse_index(
            active.paths.sessions,
            index,
            active.build_config,
        )
        turns = load_turns(index)

        # 2. Replay all historical turns so v8 feedback affects graph state.
        if turns:
            _ = rebuild_memory(
                index,
                memory,
                config=active.plugin_config.memory_config(),
                target_sequences=(),
            )
            identity = _verify_pair(
                active,
                index_path=index,
                memory_path=memory,
            )
            return CandidateState(
                index=index,
                memory=memory,
                turn_count=len(turns),
                logical_state_sha256=cast(
                    str,
                    identity["logicalStateSha256"],
                ),
            )
        identity = _verify_pair(
            active,
            index_path=index,
            memory_path=None,
        )
        return CandidateState(
            index=index,
            memory=None,
            turn_count=cast(int, identity["turnCount"]),
            logical_state_sha256=None,
        )
    except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError):
        index.unlink(missing_ok=True)
        memory.unlink(missing_ok=True)
        memory.with_suffix(memory.suffix + ".tmp").unlink(missing_ok=True)
        raise


def _backup_database(source: Path, destination: Path) -> dict[str, object]:
    """Create and validate a standalone SQLite backup."""

    # 1. Reject damaged old state before claiming it is recoverable.
    _sqlite_integrity(source)
    source_connection = sqlite3.connect(
        f"file:{source}?mode=ro",
        uri=True,
    )
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()

    # 2. Make the backup private and prove it can be opened independently.
    os.chmod(destination, 0o600)
    _sqlite_integrity(destination)
    return {
        "path": str(source),
        "existed": True,
        "backup": destination.name,
        "backupSha256": sha256_file(destination),
    }


def _backup_targets(
    context: MigrationContext,
    paths: AkashaPaths,
) -> dict[str, dict[str, object]]:
    """Back up every existing sidecar and record intentionally absent targets."""

    if context.backup_dir is None:
        raise RuntimeError("apply 缺少 --backup-dir")
    context.backup_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(context.backup_dir, 0o700)
    records: dict[str, dict[str, object]] = {}

    # 1. Preserve both sides of the published pair before replacing either.
    for name, source in (("index", paths.index), ("memory", paths.memory)):
        if source.exists():
            records[name] = _backup_database(
                source,
                context.backup_dir / f"{name}-before.db",
            )
        else:
            records[name] = {
                "path": str(source),
                "existed": False,
                "backup": None,
                "backupSha256": None,
            }
    return records


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(
    context: MigrationContext,
    records: dict[str, dict[str, object]],
    candidate: CandidateState,
) -> None:
    if context.backup_dir is None:
        raise RuntimeError("apply 缺少 --backup-dir")
    _atomic_write_json(
        context.backup_dir / "manifest.json",
        {
            "schemaVersion": _MANIFEST_VERSION,
            "migrationCommit": context.migration_commit,
            "workspace": str(context.workspace),
            "targets": records,
            "candidate": {
                "indexVersion": INDEX_VERSION,
                "turnCount": candidate.turn_count,
                "indexSha256": sha256_file(candidate.index),
                "logicalStateSha256": candidate.logical_state_sha256,
            },
        },
    )


def _publish(candidate: CandidateState, paths: AkashaPaths) -> None:
    """Publish a validated pair while the migration runner owns the workspace."""

    # 1. Publish graph first; a crash leaves an intentionally detectable mismatch.
    if candidate.memory is None:
        paths.memory.unlink(missing_ok=True)
    else:
        os.replace(candidate.memory, paths.memory)
    _fsync_directory(paths.memory.parent)

    # 2. Publish the index identity last, then make the directory entry durable.
    os.replace(candidate.index, paths.index)
    _fsync_directory(paths.index.parent)


def _validated_manifest(
    context: MigrationContext,
) -> dict[str, dict[str, object]]:
    """Validate the private restore manifest at its serialization boundary."""

    # 1. Validate bundle identity before trusting any recorded filesystem path.
    if context.backup_dir is None or not context.backup_dir.is_dir():
        raise RuntimeError("revert 需要有效的 --backup-dir")
    loaded: object = json.loads(
        (context.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(loaded, dict):
        raise ValueError("Akasha 迁移 manifest 必须是对象")
    raw = cast(dict[str, object], loaded)
    if raw.get("schemaVersion") != _MANIFEST_VERSION:
        raise ValueError("Akasha 迁移 manifest 版本不受支持")
    if raw.get("migrationCommit") != context.migration_commit:
        raise ValueError("Akasha 迁移 manifest commit 不匹配")
    if raw.get("workspace") != str(context.workspace):
        raise ValueError("Akasha 迁移 manifest workspace 不匹配")
    targets_value = raw.get("targets")
    if not isinstance(targets_value, dict):
        raise ValueError("Akasha 迁移 manifest targets 无效")
    targets = cast(dict[str, object], targets_value)
    if set(targets) != {"index", "memory"}:
        raise ValueError("Akasha 迁移 manifest targets 无效")

    # 2. Establish the target-record invariant once for the restore path.
    records: dict[str, dict[str, object]] = {}
    for name in ("index", "memory"):
        record = targets[name]
        if not isinstance(record, dict):
            raise ValueError(f"Akasha 迁移 manifest target 无效: {name}")
        records[name] = cast(dict[str, object], record)
    return records


def _restore_database(backup: Path, target: Path) -> None:
    """Restore one verified SQLite backup through an adjacent temporary file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.restore-{uuid4().hex}.tmp")
    source = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    destination = sqlite3.connect(temporary)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    os.chmod(temporary, 0o600)
    _sqlite_integrity(temporary)
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def _restore_from_manifest(context: MigrationContext) -> None:
    """Restore both pre-migration targets from a validated private manifest."""

    # 1. Validate all serialized structure before touching current sidecars.
    records = _validated_manifest(context)
    assert context.backup_dir is not None

    # 2. Restore the old graph/index pair, including originally absent files.
    for name in ("memory", "index"):
        record = records[name]
        target = Path(str(record.get("path", ""))).resolve()
        if not target.is_relative_to(context.workspace):
            raise ValueError(f"Akasha 迁移恢复路径越界: {target}")
        existed = record.get("existed")
        if not isinstance(existed, bool):
            raise ValueError(f"Akasha 迁移 manifest existed 无效: {name}")
        if not existed:
            target.unlink(missing_ok=True)
            _fsync_directory(target.parent)
            continue
        backup_name = record.get("backup")
        expected_hash = record.get("backupSha256")
        if not isinstance(backup_name, str) or not isinstance(
            expected_hash,
            str,
        ):
            raise ValueError(f"Akasha 迁移 manifest backup 无效: {name}")
        backup = (context.backup_dir / backup_name).resolve()
        if not backup.is_relative_to(context.backup_dir):
            raise ValueError(f"Akasha 迁移 backup 路径越界: {backup}")
        if sha256_file(backup) != expected_hash:
            raise ValueError(f"Akasha 迁移 backup hash 不匹配: {name}")
        _sqlite_integrity(backup)
        _restore_database(backup, target)


def _apply(context: MigrationContext) -> None:
    """Rebuild and publish Akasha v8 derived state with recoverable backups."""

    active = _active_akasha(context)
    if active is None:
        raise RuntimeError("apply 只能用于启用中的 Akasha engine")
    if not active.paths.sessions.is_file():
        raise RuntimeError(f"Akasha 重建缺少权威源: {active.paths.sessions}")
    version = (
        _index_version(active.paths.index) if active.paths.index.exists() else None
    )
    if version not in {None, _LEGACY_INDEX_VERSION, INDEX_VERSION}:
        raise RuntimeError(f"不支持 Akasha sparse index 版本: {version}")

    # 1. Prove the immutable source is complete and preserve old sidecars.
    _audit_source(active)
    records = _backup_targets(context, active.paths)

    # 2. Build and validate both candidates before publishing either one.
    candidate = _build_candidates(active)
    _write_manifest(context, records, candidate)
    try:
        # 3. Publish under the runner's workspace lock and verify runtime parity.
        _publish(candidate, active.paths)
        _ = _verify_pair(
            active,
            index_path=active.paths.index,
            memory_path=(active.paths.memory if active.paths.memory.exists() else None),
        )
    except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError):
        _restore_from_manifest(context)
        raise
    finally:
        candidate.index.unlink(missing_ok=True)
        if candidate.memory is not None:
            candidate.memory.unlink(missing_ok=True)


def _verify(context: MigrationContext) -> None:
    """Prove that applicable state is absent or a complete runtime-readable v8 pair."""

    # 1. Non-Akasha and fresh Akasha installations require no derived state.
    active = _active_akasha(context)
    if active is None:
        return
    paths = active.paths
    if not paths.index.exists() and not paths.memory.exists():
        return
    if not paths.index.exists():
        raise RuntimeError("Akasha v8 迁移验证缺少 sparse index")

    # 2. Reject legacy identity and exercise the complete runtime restore path.
    version = _index_version(paths.index)
    if version != INDEX_VERSION:
        raise RuntimeError(f"Akasha v8 迁移验证发现版本 {version}")
    _ = _verify_pair(
        active,
        index_path=paths.index,
        memory_path=paths.memory if paths.memory.exists() else None,
    )


def _revert(context: MigrationContext) -> None:
    _restore_from_manifest(context)


def main() -> None:
    action, context = _parse_args()
    if action == "assess":
        print(json.dumps(_assess(context), ensure_ascii=False))
        return
    if action == "apply":
        _apply(context)
        return
    if action == "verify":
        _verify(context)
        return
    _revert(context)


if __name__ == "__main__":
    main()
