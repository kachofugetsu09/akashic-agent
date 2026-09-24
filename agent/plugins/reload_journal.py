from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from agent.plugins import update_rollback
from agent.plugins.artifacts import ArtifactPointer, ArtifactPointers

ReloadPhase = Literal[
    "preparing",
    "prepared",
    "validating",
    "commit_started",
    "latest_ready",
    "discarding",
    "promoting",
    "committed",
    "draining",
    "cleanup_failed",
    "degraded",
    "complete",
    "aborted",
    "recovered",
]
RecoveryActionName = Literal[
    "discard_candidate",
    "restore_candidate",
    "restore_committed",
    "retry_generation_cleanup",
    "retry_runtime_recovery",
]
RecoveryTarget = Literal["base", "candidate"]
_TERMINAL_PHASES = frozenset({"complete", "aborted", "recovered"})
_FAILURE_PHASES = frozenset({"cleanup_failed", "degraded"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "preparing": frozenset({"prepared", "aborted", "cleanup_failed", "degraded"}),
    "prepared": frozenset({"validating", "aborted", "cleanup_failed", "degraded"}),
    "validating": frozenset({"commit_started", "aborted", "cleanup_failed", "degraded"}),
    "commit_started": frozenset(
        {"latest_ready", "committed", "aborted", "recovered", "cleanup_failed", "degraded"}
    ),
    "latest_ready": frozenset(
        {"discarding", "promoting", "aborted", "recovered", "cleanup_failed", "degraded"}
    ),
    "discarding": frozenset({"aborted", "cleanup_failed", "degraded"}),
    "promoting": frozenset(
        {"discarding", "committed", "aborted", "recovered", "cleanup_failed", "degraded"}
    ),
    "committed": frozenset({"draining", "complete", "recovered", "cleanup_failed", "degraded"}),
    "draining": frozenset({"complete", "recovered", "cleanup_failed", "degraded"}),
    "cleanup_failed": frozenset({"cleanup_failed", "degraded"}),
    "degraded": frozenset({"degraded"}),
}


@dataclass(frozen=True)
class ReloadTransactionRecord:
    tx_id: str
    plugin_id: str
    base_snapshot_id: str | None
    candidate_snapshot_id: str | None
    generation_id: str
    source_revision: str
    config_revision: str
    phase: ReloadPhase
    started_at: str
    updated_at: str
    error: str
    base_generation_id: str | None = None
    formal_effects: tuple[str, ...] = ()
    failure_resource: str | None = None
    recovery_action: RecoveryActionName | None = None
    attempt_count: int = 0
    runtime_owner_boot_id: str | None = None
    base_artifact_pointer: str | None = None
    candidate_artifact_pointer: str | None = None
    recovery_target: RecoveryTarget | None = None

    @property
    def old_snapshot_id(self) -> str | None:
        """Return the stable snapshot that the attempt started from."""

        return self.base_snapshot_id

    @property
    def new_snapshot_id(self) -> str | None:
        """Return the candidate snapshot produced by the attempt."""

        return self.candidate_snapshot_id

    @property
    def old_generation_id(self) -> str | None:
        """Return the stable generation that the attempt started from."""

        return self.base_generation_id

    @property
    def attempt_generation_id(self) -> str:
        """Return the generation being prepared by this attempt."""

        return self.generation_id

    @property
    def resource(self) -> str | None:
        """Return the retained failed resource owner, if any."""

        return self.failure_resource

    @property
    def attempt(self) -> int:
        """Return the durable retry attempt count."""

        return self.attempt_count


@dataclass(frozen=True)
class ReloadJournalEvent:
    sequence: int
    phase: ReloadPhase
    details: dict[str, object]
    created_at: str


@dataclass(frozen=True)
class ReloadRecoveryAction:
    tx_id: str
    plugin_id: str
    generation_id: str
    source_revision: str
    phase: ReloadPhase
    action: RecoveryActionName
    base_snapshot_id: str | None = None
    candidate_snapshot_id: str | None = None
    base_generation_id: str | None = None
    formal_effects: tuple[str, ...] = ()
    failure_resource: str | None = None
    error: str = ""
    attempt_count: int = 0
    runtime_owner_boot_id: str | None = None
    base_artifact_pointer: str | None = None
    candidate_artifact_pointer: str | None = None
    recovery_target: RecoveryTarget | None = None


@dataclass(frozen=True)
class JournalPreflight:
    """Existing journal facts and the one WAL-aware backup source."""

    pending_recovery: tuple[ReloadRecoveryAction, ...]
    armed_updates: tuple[update_rollback.UpdateRollback, ...]
    _copy: sqlite3.Connection

    def update(self, update_id: str) -> update_rollback.UpdateRollback:
        """Read one exact row from the checked, read-only journal snapshot."""
        return update_rollback.read(self._copy, update_id)

    def backup_to(self, path: Path) -> None:
        """Save the checked snapshot without opening the live journal in SQLite."""
        saved = sqlite3.connect(path)
        try:
            self._copy.backup(saved)
            if saved.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("reload journal 备份完整性检查失败")
        finally:
            saved.close()


def _journal_bytes(path: Path) -> tuple[bytes, tuple[int, int, int]]:
    """Read an existing regular file without following a replacement link."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"journal 路径必须是普通文件: {path}")
        with os.fdopen(os.dup(fd), "rb") as stream:
            content = stream.read()
        if len(content) != info.st_size:
            raise RuntimeError(f"journal 在读取期间变化: {path}")
        return content, (info.st_dev, info.st_ino, info.st_size)
    finally:
        os.close(fd)


def _check_journal_source(
    paths: tuple[Path, ...], original: dict[Path, tuple[bytes, tuple[int, int, int]]],
) -> None:
    """Reject a changed source or sidecar set while its snapshot is in use."""
    if {path for path in paths if path.exists() or path.is_symlink()} != set(original):
        raise RuntimeError("journal sidecar 在读取期间变化")
    for path, (content, identity) in original.items():
        observed, current_identity = _journal_bytes(path)
        if current_identity != identity or hashlib.sha256(observed).digest() != hashlib.sha256(content).digest():
            raise RuntimeError(f"journal 在读取期间变化: {path}")

class ReloadJournal:
    """Persist plugin reload phases and expose deterministic crash recovery work."""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / "runtime" / "plugin-reloads.sqlite3"
        new = not self.path.exists()
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        else:
            self._check_existing_schema()

    @classmethod
    @contextmanager
    def inspect_existing(cls, workspace: Path) -> Iterator[JournalPreflight]:
        """Read one existing DB/WAL/SHM snapshot without SQLite opening the source."""
        if workspace.is_symlink() or not workspace.is_dir():
            raise ValueError(f"journal workspace 目录无效: {workspace}")
        runtime = workspace / "runtime"
        if runtime.is_symlink() or not runtime.is_dir():
            raise ValueError(f"journal runtime 目录无效: {runtime}")
        source = runtime / "plugin-reloads.sqlite3"
        rollback_sidecar = runtime / "plugin-reloads.sqlite3-journal"
        if rollback_sidecar.exists() or rollback_sidecar.is_symlink():
            raise RuntimeError("journal 存在未结算 rollback sidecar")
        sources = (source, Path(f"{source}-wal"), Path(f"{source}-shm"))
        original: dict[Path, tuple[bytes, tuple[int, int, int]]] = {}
        for path in sources:
            if path == source or path.exists() or path.is_symlink():
                original[path] = _journal_bytes(path)
        with tempfile.TemporaryDirectory(prefix="akashic-journal-preflight-") as directory:
            copy = Path(directory) / source.name
            if Path(directory).resolve().is_relative_to(workspace.resolve()):
                raise ValueError("journal 临时副本不能位于 workspace 内")
            for path, (content, _) in original.items():
                (Path(directory) / path.name).write_bytes(content)
            _check_journal_source(sources, original)
            conn = sqlite3.connect(copy)
            try:
                conn.execute("PRAGMA query_only=ON")
                cls._check_schema(conn)
                if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise RuntimeError("reload journal 内容损坏")
                pending = cls._pending_recovery(conn)
                armed = tuple(
                    update_rollback.read(conn, str(row[0]))
                    for row in conn.execute(
                        "SELECT update_id FROM plugin_updates WHERE phase='armed' ORDER BY update_id"
                    )
                )
                yield JournalPreflight(pending, armed, conn)
            finally:
                conn.close()
                _check_journal_source(sources, original)

    def arm_update(
        self, *, update_id: str, plugin_id: str, plugin_base: Path,
        previous: ArtifactPointers | None, candidate: ArtifactPointer,
        previous_enabled: bool | None,
    ) -> None:
        with self._connect() as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            update_rollback.arm(conn, update_id=update_id, plugin_id=plugin_id, plugin_base=plugin_base,
                previous=previous, candidate=candidate, previous_enabled=previous_enabled, now=_now())

    def update(self, update_id: str) -> update_rollback.UpdateRollback:
        with self._connect() as conn:
            return update_rollback.read(conn, update_id)

    def set_input_ref(self, update_id: str, input_ref: str) -> None:
        """Persist the fixed archive input before selection CAS."""
        with self._connect() as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            update_rollback.set_input_ref(conn, update_id=update_id, input_ref=input_ref)

    def update_for_reload(self, tx_id: str) -> update_rollback.UpdateRollback | None:
        """有更新恢复点时，完整旧指针对只由该记录恢复。"""
        with self._connect() as conn:
            if not update_rollback.check_schema(conn):
                return None
            row = conn.execute("SELECT update_id FROM plugin_updates WHERE reload_tx_id=?", (tx_id,)).fetchone()
            return None if row is None else update_rollback.read(conn, row[0])

    def record_update_error(self, update_id: str, error: str) -> None:
        """保存实际失败原因，不把诊断写入伪装成发布或回退。"""
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE plugin_updates SET error=?,updated_at=? WHERE update_id=?",
                (error, _now(), update_id),
            )
            if changed.rowcount != 1:
                raise KeyError(f"插件更新不存在: {update_id}")

    def commit_update(self, update_id: str) -> None:
        """离线安装没有 runtime reload；安装 owner 核验完成后提交恢复点。"""
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE plugin_updates SET phase='committed',updated_at=? WHERE update_id=? AND phase='armed' AND reload_tx_id IS NULL",
                (_now(), update_id),
            )
            if changed.rowcount != 1:
                raise RuntimeError("插件安装恢复点不能提交")

    def rollback_updates(self, plugins_home: Path, *, update_id: str | None = None, error: str = "update interrupted") -> None:
        """启动前或安装失败后恢复旧指针；已有提交不参加自动回退。"""
        with self._connect() as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            if not update_rollback.check_schema(conn):
                return
            query = "SELECT update_id FROM plugin_updates WHERE phase='armed'"
            values: tuple[str, ...] = ()
            if update_id is not None:
                query += " AND update_id=?"
                values = (update_id,)
            for row in conn.execute(query, values).fetchall():
                update_rollback.rollback(conn, update_rollback.read(conn, row[0]), plugins_home, now=_now(), error=error)

    def rollback_install_update(
        self, plugins_home: Path, *, expected: update_rollback.UpdateRollback, error: str,
    ) -> None:
        """Roll back one unchanged, unlinked install row under the caller's offline locks."""
        with self._connect() as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            current = update_rollback.read(conn, expected.update_id)
            if current != expected:
                raise RuntimeError("插件安装恢复点在预检后改变")
            if current.phase != "armed" or current.reload_tx_id is not None or current.input_ref is not None:
                raise RuntimeError("指定记录不是孤立 armed 安装")
            update_rollback.rollback(conn, current, plugins_home, now=_now(), error=error)

    def begin(
        self,
        *,
        plugin_id: str,
        base_snapshot_id: str | None,
        base_generation_id: str | None = None,
        generation_id: str,
        source_revision: str,
        config_revision: str,
        base_artifact_pointer: str | None = None,
        candidate_artifact_pointer: str | None = None,
        details: dict[str, object] | None = None,
    ) -> str:
        now = _now()
        tx_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reload_transactions (
                    tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                    base_generation_id, generation_id, source_revision,
                    config_revision, phase, started_at, updated_at, error,
                    formal_effects_json, attempt_count, base_artifact_pointer,
                    candidate_artifact_pointer
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, 'preparing', ?, ?, '', '[]', 0, ?, ?)
                """,
                (
                    tx_id,
                    plugin_id,
                    base_snapshot_id,
                    base_generation_id,
                    generation_id,
                    source_revision,
                    config_revision,
                    now,
                    now,
                    base_artifact_pointer,
                    candidate_artifact_pointer,
                ),
            )
            self._append_event(conn, tx_id, "preparing", details or {}, now)
            update_rollback.link(conn, tx_id=tx_id, plugin_id=plugin_id, candidate_pointer=candidate_artifact_pointer)
        return tx_id

    def mark_runtime_owner(self, tx_id: str, boot_id: str) -> None:
        """Persist the boot owner before starting any candidate or formal runtime."""

        if not boot_id.strip():
            raise ValueError("ReloadTransaction runtime boot id 不能为空")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT runtime_owner_boot_id, phase FROM reload_transactions WHERE tx_id = ?",
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            existing = _optional_string(row[0])
            if existing not in {None, boot_id}:
                raise RuntimeError(
                    "ReloadTransaction runtime boot owner 不可覆盖: "
                    f"{existing} -> {boot_id}"
                )
            now = _now()
            conn.execute(
                """
                UPDATE reload_transactions
                SET runtime_owner_boot_id = COALESCE(runtime_owner_boot_id, ?),
                    updated_at = ?
                WHERE tx_id = ?
                """,
                (boot_id, now, tx_id),
            )
            self._append_event(
                conn,
                tx_id,
                cast(ReloadPhase, str(row[1])),
                {"runtime_owner_boot_id": boot_id},
                now,
            )

    def advance(
        self,
        tx_id: str,
        phase: ReloadPhase,
        *,
        candidate_snapshot_id: str | None = None,
        details: dict[str, object] | None = None,
        error: str | None = None,
        resource: str | None = None,
        formal_effects: tuple[str, ...] | None = None,
        recovery_action: RecoveryActionName | None = None,
        attempt_count: int | None = None,
        recovery_target: RecoveryTarget | None = None,
    ) -> None:
        """Advance one reload transaction and append its durable evidence."""

        with self._connect() as conn:
            # 1. 读取当前状态并验证单向 phase contract。
            row = conn.execute(
                """
                SELECT phase, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, formal_effects_json,
                       failure_resource, recovery_action, attempt_count, error,
                       runtime_owner_boot_id, base_artifact_pointer,
                       candidate_artifact_pointer, recovery_target
                FROM reload_transactions
                WHERE tx_id = ?
                """,
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            current = str(row[0])
            if phase not in _TRANSITIONS.get(current, frozenset()):
                raise RuntimeError(
                    f"ReloadTransaction 状态跳转无效: {current} -> {phase}"
                )
            # 2. 规范化 snapshot、resource、formal effect 与 recovery evidence。
            details_for_event = dict(details or {})
            current_candidate = _optional_string(row[2])
            current_base_generation = _optional_string(row[3])
            current_effects = _decode_effects(row[5])
            current_resource = _optional_string(row[6])
            current_action = _optional_action(row[7])
            current_attempt_count = int(row[8])
            next_candidate = candidate_snapshot_id
            if next_candidate is None:
                next_candidate = _detail_string(
                    details_for_event,
                    "new_snapshot_id",
                    "candidate_snapshot_id",
                )
            if next_candidate is None:
                next_candidate = current_candidate
            next_base_generation = current_base_generation
            detail_base_generation = _detail_string(
                details_for_event,
                "old_generation_id",
                "base_generation_id",
                "old_generation",
            )
            if detail_base_generation is not None:
                next_base_generation = detail_base_generation
            next_resource = resource
            if next_resource is None:
                next_resource = _detail_string(
                    details_for_event,
                    "resource",
                    "failure_resource",
                )
            if next_resource is None:
                next_resource = current_resource
            elif current in _FAILURE_PHASES and current_resource is not None:
                next_resource = _merge_resources(current_resource, next_resource)
            next_effects = current_effects
            if formal_effects is not None:
                next_effects = _merge_effects(
                    current_effects,
                    _validate_effects(formal_effects),
                )
            elif "formal_effects" in details_for_event:
                next_effects = _merge_effects(
                    current_effects,
                    _validate_effects(details_for_event["formal_effects"]),
                )
            next_action = recovery_action
            if next_action is None:
                next_action = _optional_action(details_for_event.get("recovery_action"))
            expected_action = _recovery_action(phase)
            if next_action is None and phase in _FAILURE_PHASES:
                next_action = expected_action
            elif next_action is None and current in _FAILURE_PHASES:
                next_action = current_action
            if next_action is None:
                next_action = expected_action
            if expected_action is not None:
                if next_action is None:
                    next_action = expected_action
                elif next_action != expected_action:
                    raise RuntimeError(
                        f"ReloadTransaction 恢复 action 与状态不一致: {phase} -> {next_action}"
                    )
            next_attempt_count = current_attempt_count
            if attempt_count is None and "attempt" in details_for_event:
                raw_attempt = details_for_event["attempt"]
                if isinstance(raw_attempt, int) and not isinstance(raw_attempt, bool):
                    attempt_count = raw_attempt
            if phase in _FAILURE_PHASES:
                if attempt_count is None:
                    next_attempt_count += 1
                else:
                    if attempt_count < current_attempt_count:
                        raise ValueError("ReloadTransaction attempt_count 不能减少")
                    next_attempt_count = attempt_count
            elif attempt_count is not None:
                if attempt_count < current_attempt_count:
                    raise ValueError("ReloadTransaction attempt_count 不能减少")
                next_attempt_count = attempt_count
            next_error = error
            if next_error is None:
                next_error = _detail_string(details_for_event, "error")
            if next_error is None:
                next_error = str(row[9])
            if phase in _FAILURE_PHASES:
                stored_target = _optional_recovery_target(row[13])
                if (
                    stored_target is not None
                    and recovery_target is not None
                    and recovery_target != stored_target
                ):
                    raise RuntimeError(
                        "ReloadTransaction recovery target 不可覆盖: "
                        f"{stored_target} -> {recovery_target}"
                    )
                next_recovery_target = (
                    recovery_target
                    if recovery_target is not None
                    else stored_target
                )
                if next_recovery_target is None:
                    raise ValueError(
                        "ReloadTransaction failure phase 必须保存 recovery target"
                    )
                if not next_resource:
                    raise ValueError(
                        "ReloadTransaction failure phase 必须保存 resource identity"
                    )
                if not next_error:
                    raise ValueError(
                        "ReloadTransaction failure phase 必须保存 error evidence"
                    )
                _add_failure_evidence(
                    details_for_event,
                    base_snapshot_id=_optional_string(row[1]),
                    candidate_snapshot_id=next_candidate,
                    base_generation_id=next_base_generation,
                    generation_id=str(row[4]),
                    formal_effects=next_effects,
                    resource=next_resource,
                    error=next_error,
                    action=next_action,
                    attempt_count=next_attempt_count,
                    runtime_owner_boot_id=_optional_string(row[10]),
                    base_artifact_pointer=_optional_string(row[11]),
                    candidate_artifact_pointer=_optional_string(row[12]),
                    recovery_target=next_recovery_target,
                )
            else:
                next_recovery_target = _optional_recovery_target(row[13])
            # 3. 在同一 SQLite transaction 内更新状态并追加事件。
            now = _now()
            conn.execute(
                """
                UPDATE reload_transactions
                SET phase = ?,
                    candidate_snapshot_id = COALESCE(?, candidate_snapshot_id),
                    base_generation_id = COALESCE(?, base_generation_id),
                    updated_at = ?,
                    error = ?,
                    formal_effects_json = ?,
                    failure_resource = ?,
                    recovery_action = ?,
                    attempt_count = ?,
                    recovery_target = COALESCE(?, recovery_target)
                WHERE tx_id = ?
                """,
                (
                    phase,
                    next_candidate,
                    next_base_generation,
                    now,
                    next_error,
                    json.dumps(next_effects, ensure_ascii=False),
                    next_resource,
                    next_action,
                    next_attempt_count,
                    next_recovery_target,
                    tx_id,
                ),
            )
            self._append_event(conn, tx_id, phase, details_for_event, now)
            if phase == "committed":
                update_rollback.commit(conn, tx_id, now)
            # 安装回退属于安装 owner；运行候选失败不能回写 manifest/pointers。

    def get(self, tx_id: str) -> ReloadTransactionRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, source_revision,
                       config_revision, phase, started_at, updated_at, error,
                       formal_effects_json, failure_resource, recovery_action,
                       attempt_count, runtime_owner_boot_id,
                       base_artifact_pointer, candidate_artifact_pointer,
                       recovery_target
                FROM reload_transactions
                WHERE tx_id = ?
                """,
                (tx_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
        return _record(row)

    def latest(
        self,
        *,
        plugin_id: str | None = None,
    ) -> ReloadTransactionRecord | None:
        """返回指定插件最后发生状态变化的 reload transaction。"""
        where = "" if plugin_id is None else "WHERE plugin_id = ?"
        values: tuple[object, ...] = () if plugin_id is None else (plugin_id,)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, source_revision,
                       config_revision, phase, started_at, updated_at, error,
                       formal_effects_json, failure_resource, recovery_action,
                       attempt_count, runtime_owner_boot_id,
                       base_artifact_pointer, candidate_artifact_pointer,
                       recovery_target
                FROM reload_transactions
                {where}
                ORDER BY updated_at DESC, rowid DESC
                LIMIT 1
                """,
                values,
            ).fetchone()
        return None if row is None else _record(row)

    def events(self, tx_id: str) -> tuple[ReloadJournalEvent, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sequence, phase, details_json, created_at
                FROM reload_events
                WHERE tx_id = ?
                ORDER BY sequence
                """,
                (tx_id,),
            ).fetchall()
        return tuple(
            ReloadJournalEvent(
                sequence=int(row[0]),
                phase=cast(ReloadPhase, str(row[1])),
                details=cast(dict[str, object], json.loads(str(row[2]))),
                created_at=str(row[3]),
            )
            for row in rows
        )

    def runtime_generation_ids(self, tx_id: str) -> tuple[str, ...]:
        """从实际取得资源的事件读取身份，候选 ID 不代替正式 owner。"""
        record = self.get(tx_id)
        identities = {record.generation_id}
        if record.base_generation_id is not None:
            identities.add(record.base_generation_id)
        for event in self.events(tx_id):
            owner = event.details.get("runtime_generation_id")
            if isinstance(owner, str):
                identities.add(owner)
            owners = event.details.get("runtime_generations")
            if isinstance(owners, dict):
                identities.update(cast(dict[str, str], owners).values())
        return tuple(sorted(identities))

    def annotate(self, tx_id: str, details: dict[str, object]) -> None:
        """Append evidence without inventing another public rollout phase."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT phase, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, formal_effects_json,
                       failure_resource, recovery_action, attempt_count, error
                FROM reload_transactions
                WHERE tx_id = ?
                """,
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            now = _now()
            phase = cast(ReloadPhase, str(row[0]))
            details_for_event = dict(details)
            candidate_snapshot_id = _detail_string(
                details_for_event,
                "new_snapshot_id",
                "candidate_snapshot_id",
            )
            base_generation_id = _detail_string(
                details_for_event,
                "old_generation_id",
                "base_generation_id",
                "old_generation",
            )
            resource = _detail_string(
                details_for_event,
                "resource",
                "failure_resource",
            )
            effects = _decode_effects(row[5])
            if "formal_effects" in details_for_event:
                effects = _merge_effects(
                    effects,
                    _validate_effects(details_for_event["formal_effects"]),
                )
            action = _optional_action(details_for_event.get("recovery_action"))
            expected_action = _recovery_action(phase)
            if expected_action is not None and action not in {None, expected_action}:
                raise RuntimeError(
                    f"ReloadTransaction 恢复 action 与状态不一致: {phase} -> {action}"
                )
            if action is None:
                action = expected_action or _optional_action(row[7])
            attempt_count = int(row[8])
            if "attempt_count" in details_for_event:
                attempt_count = _validate_attempt_count(
                    details_for_event["attempt_count"],
                    minimum=attempt_count,
                )
            error = _detail_string(details_for_event, "error")
            conn.execute(
                """
                UPDATE reload_transactions
                SET candidate_snapshot_id = COALESCE(?, candidate_snapshot_id),
                    base_generation_id = COALESCE(?, base_generation_id),
                    updated_at = ?,
                    error = COALESCE(?, error),
                    formal_effects_json = ?,
                    failure_resource = COALESCE(?, failure_resource),
                    recovery_action = COALESCE(?, recovery_action),
                    attempt_count = ?
                WHERE tx_id = ?
                """,
                (
                    candidate_snapshot_id,
                    base_generation_id,
                    now,
                    error,
                    json.dumps(effects, ensure_ascii=False),
                    resource,
                    action,
                    attempt_count,
                    tx_id,
                ),
            )
            self._append_event(conn, tx_id, phase, details_for_event, now)

    def pending_recovery(self) -> tuple[ReloadRecoveryAction, ...]:
        with self._connect() as conn:
            return self._pending_recovery(conn)

    def orphaned_armed_updates(self) -> tuple[update_rollback.UpdateRollback, ...]:
        """Read installs with no runtime transaction to settle at boot."""
        with self._connect() as conn:
            return tuple(
                update_rollback.read(conn, str(row[0]))
                for row in conn.execute(
                    "SELECT update_id FROM plugin_updates "
                    "WHERE phase='armed' AND reload_tx_id IS NULL ORDER BY update_id"
                )
            )

    def settle_generation_cleanup(
        self, *, tx_id: str, plugin_id: str, generation_id: str, receipt: str,
    ) -> None:
        """Close the exact failed cleanup only after its owner releases resources."""
        if not receipt:
            raise ValueError("generation cleanup 缺少清理回执")
        with self._connect() as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT plugin_id,generation_id,phase,recovery_action,failure_resource "
                "FROM reload_transactions WHERE tx_id=?", (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            if tuple(row) != (
                plugin_id, generation_id, "cleanup_failed", "retry_generation_cleanup",
                f"generation-cleanup:{generation_id}",
            ):
                raise RuntimeError("generation cleanup 回执与原失败 owner 不匹配")
            now = _now()
            changed = conn.execute(
                "UPDATE reload_transactions SET phase='recovered',updated_at=? "
                "WHERE tx_id=? AND plugin_id=? AND generation_id=? "
                "AND phase='cleanup_failed' AND recovery_action='retry_generation_cleanup'",
                (now, tx_id, plugin_id, generation_id),
            )
            if changed.rowcount != 1:
                raise RuntimeError("generation cleanup 结算事实已变化")
            self._append_event(conn, tx_id, "recovered", {
                "event": "generation_cleanup_settled",
                "plugin_id": plugin_id,
                "generation_id": generation_id,
                "cleanup_receipt": receipt,
            }, now)

    @staticmethod
    def _pending_recovery(conn: sqlite3.Connection) -> tuple[ReloadRecoveryAction, ...]:
        placeholders = ", ".join("?" for _ in _TERMINAL_PHASES)
        rows = conn.execute(
                f"""
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, source_revision, phase,
                       formal_effects_json, failure_resource, recovery_action,
                       error, attempt_count, runtime_owner_boot_id,
                       base_artifact_pointer, candidate_artifact_pointer,
                       recovery_target
                FROM reload_transactions
                WHERE phase NOT IN ({placeholders})
                ORDER BY started_at, tx_id
                """,
                tuple(sorted(_TERMINAL_PHASES)),
            ).fetchall()
        actions: list[ReloadRecoveryAction] = []
        for row in rows:
            phase = cast(ReloadPhase, str(row[7]))
            action = _recovery_action(phase, _optional_action(row[10]))
            if action is None:
                raise RuntimeError(f"ReloadTransaction 无法恢复状态: {phase}")
            target = _optional_recovery_target(row[16])
            actions.append(
                ReloadRecoveryAction(
                    tx_id=str(row[0]),
                    plugin_id=str(row[1]),
                    generation_id=str(row[5]),
                    source_revision=str(row[6]),
                    phase=phase,
                    action=action,
                    base_snapshot_id=_optional_string(row[2]),
                    candidate_snapshot_id=_optional_string(row[3]),
                    base_generation_id=_optional_string(row[4]),
                    formal_effects=_decode_effects(row[8]),
                    failure_resource=_optional_string(row[9]),
                    error=str(row[11]),
                    attempt_count=int(row[12]),
                    runtime_owner_boot_id=_optional_string(row[13]),
                    base_artifact_pointer=_optional_string(row[14]),
                    candidate_artifact_pointer=_optional_string(row[15]),
                    recovery_target=target,
                )
            )
        return tuple(
            sorted(
                actions,
                key=lambda item: (
                    _RECOVERY_ACTION_ORDER[item.action],
                    item.tx_id,
                ),
            )
        )

    def selection_candidate(self, tx_id: str) -> tuple[str | None, tuple[str, ...]] | None:
        """读取一次候选的完整转换证据；历史记录不猜成新格式。"""
        events = self.events(tx_id)
        candidates = [event.details for event in events if event.details.get("event") == "selection_candidate"]
        if not candidates:
            return None
        if len(candidates) != 1 or not events or "base_selection_ref" not in events[0].details:
            raise RuntimeError("候选 selection 证据不完整或重复")
        base = events[0].details["base_selection_ref"]
        components = candidates[0]["components"]
        if (base is not None and not isinstance(base, str)) or not isinstance(components, list):
            raise RuntimeError("候选 selection 证据格式损坏")
        if any(not isinstance(ref, str) for ref in components):
            raise RuntimeError("候选 component ref 格式损坏")
        return base, tuple(components)

    def settle_boot(
        self, action: ReloadRecoveryAction, *, committed: bool | None,
        cleanup_receipt: str | None,
    ) -> None:
        """旧进程 owner 清理后只结算有证据的转换；未知保留原状态。"""
        if (action.runtime_owner_boot_id is not None or action.action in {
            "retry_generation_cleanup", "retry_runtime_recovery",
        }) and not cleanup_receipt:
            raise RuntimeError("旧 runtime owner 缺少清理回执")
        now = _now()
        phase = action.phase if committed is None else ("recovered" if committed else "aborted")
        detail = {"event": "boot_selection_observed", "selection_committed": committed,
                  "cleanup_receipt": cleanup_receipt, "candidate_resumed": False}
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reload_transactions SET phase=?,updated_at=? WHERE tx_id=? AND phase=? AND attempt_count=?",
                (phase, now, action.tx_id, action.phase, action.attempt_count),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("旧 boot recovery 证据已失效")
            self._append_event(conn, action.tx_id, cast(ReloadPhase, phase), detail, now)
            if committed is True:
                update_rollback.commit(conn, action.tx_id, now)
            elif update_rollback.check_schema(conn):
                # armed 是尚待安装 owner 结算，不谎称已经回退安装文件。
                conn.execute(
                    "UPDATE plugin_updates SET updated_at=?,error=? WHERE reload_tx_id=? AND phase='armed'",
                    (now, "runtime selection not committed; installation needs explicit settlement"
                     if committed is False else "runtime selection evidence unknown; explicit settlement required",
                     action.tx_id),
                )

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reload_transactions (
                    tx_id TEXT PRIMARY KEY,
                    plugin_id TEXT NOT NULL,
                    base_snapshot_id TEXT,
                    candidate_snapshot_id TEXT,
                    base_generation_id TEXT,
                    generation_id TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    config_revision TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT NOT NULL,
                    formal_effects_json TEXT NOT NULL DEFAULT '[]',
                    failure_resource TEXT,
                    recovery_action TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    runtime_owner_boot_id TEXT,
                    base_artifact_pointer TEXT,
                    candidate_artifact_pointer TEXT,
                    recovery_target TEXT
                );
                CREATE TABLE IF NOT EXISTS reload_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_id TEXT NOT NULL REFERENCES reload_transactions(tx_id),
                    phase TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reload_transactions_phase
                ON reload_transactions(phase);
                CREATE INDEX IF NOT EXISTS idx_reload_events_tx
                ON reload_events(tx_id, sequence);
                """)

            for statement in update_rollback.SCHEMA.values():
                _ = conn.execute(statement)

    def _check_existing_schema(self) -> None:
        """Read an existing journal without creating or altering any object."""
        uri = f"file:{self.path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            self._check_schema(conn)
        finally:
            conn.close()

    @staticmethod
    def _check_schema(conn: sqlite3.Connection) -> None:
        """Check the current schema on a caller-owned connection."""
        try:
            shape = update_rollback.plugin_update_schema_state(conn)
        except ValueError as error:
            raise RuntimeError(f"runtime/plugin-reloads.sqlite3 schema 无法识别: {error}") from error
        if shape == "old":
            raise RuntimeError(
                "runtime/plugin-reloads.sqlite3 使用旧 plugin_updates schema；"
                "请先执行 Core migration，不会由普通启动自动迁移"
            )
        if shape == "missing":
            raise RuntimeError(
                "runtime/plugin-reloads.sqlite3 缺少 plugin_updates；"
                "不会由普通启动补造历史表"
            )
        required = {
            "reload_transactions": {
                "tx_id", "plugin_id", "base_snapshot_id", "candidate_snapshot_id",
                "base_generation_id", "generation_id", "source_revision", "config_revision",
                "phase", "started_at", "updated_at", "error", "formal_effects_json",
                "failure_resource", "recovery_action", "attempt_count",
                "runtime_owner_boot_id", "base_artifact_pointer", "candidate_artifact_pointer",
                "recovery_target",
            },
            "reload_events": {"sequence", "tx_id", "phase", "details_json", "created_at"},
        }
        for table, columns in required.items():
            actual = {
                str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if actual != columns:
                raise RuntimeError(
                    f"runtime/plugin-reloads.sqlite3 缺少或包含未知 {table} 列；"
                    "请先执行对应 Core migration"
                )
        indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(reload_transactions)")
        }
        event_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(reload_events)")
        }
        if "idx_reload_transactions_phase" not in indexes or "idx_reload_events_tx" not in event_indexes:
            raise RuntimeError(
                "runtime/plugin-reloads.sqlite3 缺少当前索引；请先执行 Core migration"
            )
        valid_phases = tuple(sorted(_TRANSITIONS.keys() | _TERMINAL_PHASES))
        if conn.execute(
            "SELECT 1 FROM reload_transactions WHERE phase NOT IN ("
            + ",".join("?" for _ in valid_phases) + ") LIMIT 1",
            valid_phases,
        ).fetchone() is not None:
            raise RuntimeError("runtime/plugin-reloads.sqlite3 含未知 reload phase")
        if conn.execute(
            "SELECT 1 FROM plugin_updates WHERE phase NOT IN ('armed','committed','rolled_back') LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeError("runtime/plugin-reloads.sqlite3 含未知 install phase")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        tx_id: str,
        phase: ReloadPhase,
        details: dict[str, object],
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO reload_events (tx_id, phase, details_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                tx_id,
                phase,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
                created_at,
            ),
        )


def _record(row: sqlite3.Row | tuple[object, ...]) -> ReloadTransactionRecord:
    return ReloadTransactionRecord(
        tx_id=str(row[0]),
        plugin_id=str(row[1]),
        base_snapshot_id=None if row[2] is None else str(row[2]),
        candidate_snapshot_id=None if row[3] is None else str(row[3]),
        base_generation_id=_optional_string(row[4]),
        generation_id=str(row[5]),
        source_revision=str(row[6]),
        config_revision=str(row[7]),
        phase=cast(ReloadPhase, str(row[8])),
        started_at=str(row[9]),
        updated_at=str(row[10]),
        error=str(row[11]),
        formal_effects=_decode_effects(row[12]),
        failure_resource=_optional_string(row[13]),
        recovery_action=_optional_action(row[14]),
        attempt_count=_stored_attempt_count(row[15]),
        runtime_owner_boot_id=_optional_string(row[16]),
        base_artifact_pointer=_optional_string(row[17]),
        candidate_artifact_pointer=_optional_string(row[18]),
        recovery_target=_optional_recovery_target(row[19]),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_RECOVERY_ACTION_ORDER: dict[RecoveryActionName, int] = {
    "restore_committed": 0,
    "retry_runtime_recovery": 1,
    "retry_generation_cleanup": 2,
    "discard_candidate": 3,
    "restore_candidate": 4,
}
_RECOVERY_ACTIONS = frozenset(_RECOVERY_ACTION_ORDER)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"ReloadTransaction 字段必须是字符串: {value!r}")
    return value


def _detail_string(details: dict[str, object], *names: str) -> str | None:
    for name in names:
        if name in details:
            return _optional_string(details[name])
    return None


def _validate_effects(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError("ReloadTransaction formal_effects 必须是字符串序列")
    effects: list[str] = []
    for item in cast(list[object] | tuple[object, ...], value):
        if not isinstance(item, str):
            raise TypeError("ReloadTransaction formal_effects 必须只包含字符串")
        effects.append(item)
    return tuple(effects)


def _decode_effects(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _validate_effects(json.loads(value))
    return _validate_effects(value)


def _merge_effects(
    current: tuple[str, ...], additions: tuple[str, ...]
) -> tuple[str, ...]:
    merged = list(current)
    seen = set(current)
    for effect in additions:
        if effect not in seen:
            merged.append(effect)
            seen.add(effect)
    return tuple(merged)


def _merge_resources(current: str, additions: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for resource in (*current.split(","), *additions.split(",")):
        item = resource.strip()
        if item and item not in seen:
            merged.append(item)
            seen.add(item)
    return ",".join(merged)


def _optional_action(value: object) -> RecoveryActionName | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _RECOVERY_ACTIONS:
        raise ValueError(f"ReloadTransaction recovery action 无效: {value!r}")
    return cast(RecoveryActionName, value)


def _optional_recovery_target(value: object) -> RecoveryTarget | None:
    if value is None:
        return None
    if value not in {"base", "candidate"}:
        raise ValueError(f"ReloadTransaction recovery target 无效: {value!r}")
    return cast(RecoveryTarget, value)


def _validate_attempt_count(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ReloadTransaction attempt_count 必须是整数")
    if value < minimum or value < 0:
        raise ValueError("ReloadTransaction attempt_count 不能减少或为负数")
    return value


def _stored_attempt_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ReloadTransaction attempt_count 存储值必须是整数")
    if value < 0:
        raise ValueError("ReloadTransaction attempt_count 不能为负数")
    return value


def _add_failure_evidence(
    details: dict[str, object],
    *,
    base_snapshot_id: str | None,
    candidate_snapshot_id: str | None,
    base_generation_id: str | None,
    generation_id: str,
    formal_effects: tuple[str, ...],
    resource: str | None,
    error: str,
    action: RecoveryActionName | None,
    attempt_count: int,
    runtime_owner_boot_id: str | None,
    base_artifact_pointer: str | None,
    candidate_artifact_pointer: str | None,
    recovery_target: RecoveryTarget | None,
) -> None:
    _ = details.setdefault("old_snapshot_id", base_snapshot_id)
    _ = details.setdefault("new_snapshot_id", candidate_snapshot_id)
    _ = details.setdefault("old_generation_id", base_generation_id)
    _ = details.setdefault("attempt_generation_id", generation_id)
    _ = details.setdefault("formal_effects", formal_effects)
    _ = details.setdefault("resource", resource)
    _ = details.setdefault("error", error)
    _ = details.setdefault("recovery_action", action)
    _ = details.setdefault("attempt_count", attempt_count)
    _ = details.setdefault("attempt", attempt_count)
    _ = details.setdefault("runtime_owner_boot_id", runtime_owner_boot_id)
    _ = details.setdefault("base_artifact_pointer", base_artifact_pointer)
    _ = details.setdefault("candidate_artifact_pointer", candidate_artifact_pointer)
    _ = details.setdefault("recovery_target", recovery_target)


def _recovery_action(
    phase: str,
    persisted: RecoveryActionName | None = None,
) -> RecoveryActionName | None:
    if persisted is not None:
        return persisted
    if phase in {"latest_ready", "discarding", "preparing", "prepared", "validating"}:
        return "discard_candidate"
    if phase in {"commit_started", "promoting", "committed", "draining"}:
        return "restore_committed"
    if phase == "cleanup_failed":
        return "retry_generation_cleanup"
    if phase == "degraded":
        return "retry_runtime_recovery"
    return None
