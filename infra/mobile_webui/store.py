from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from uuid import UUID

from infra.mobile_webui.manifest import (
    ManifestError,
    WebUiManifest,
    WebUiTarget,
    canonical_manifest_bytes,
    derive_target_key,
    manifest_digest,
    manifest_from_json,
    validate_manifest,
)


class ReleaseConflictError(RuntimeError):
    """表示不可变 generation/blob 已经存在但内容不一致。"""


class UnknownReleaseError(KeyError):
    """表示 target 或 generation 不存在。"""


class RollbackUnavailableError(UnknownReleaseError):
    """表示目标既未显式 pin，也不在最近 channel 选择窗口内。"""

    code = "rollback_unavailable"


class ReleaseSelectionChangedError(UnknownReleaseError):
    """表示请求期间 ReleaseView selection 已经变化。"""


class TargetResourceNotFoundError(UnknownReleaseError):
    """表示 digest 不是当前 target 的成员。"""


@dataclass(frozen=True, slots=True)
class StoredBlob:
    digest: str
    size_bytes: int
    path: Path
    # MIME 只有在 verify_target_resource 返回时才有值；CAS 自身不拥有 MIME 语义。
    mime: str | None = None


@dataclass(frozen=True, slots=True)
class ReleaseView:
    server_id: str
    release_epoch: str
    sequence: int
    selection_digest: str
    stable: WebUiTarget | None
    preview: WebUiTarget | None

    def target(self, target_key: str) -> WebUiTarget | None:
        for target in (self.stable, self.preview):
            if target is not None and target.target_key == target_key:
                return target
        return None


@dataclass(frozen=True, slots=True)
class GarbageCollectionReport:
    removed_generations: tuple[str, ...]
    removed_blobs: tuple[str, ...]


class MobileWebUiStore:
    """维护 WebUI immutable CAS、单一发布视图和 append-only 审计日志。"""

    def __init__(self, root: Path, *, server_id: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", server_id) is None:
            raise ValueError("mobile WebUI server_id 无效")
        self.root = root
        self.db_path = root / "publication.sqlite3"
        self.blob_root = root / "blobs" / "sha256"
        self.staging_root = root / "staging"
        self.trash_root = root / "trash"
        self.lock_path = root / "publication.lock"
        self.server_id = server_id
        self._recover_restore_marker(root, server_id=server_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.trash_root.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.lock_path.open("a+")
        try:
            self._db = sqlite3.connect(self.db_path, isolation_level=None)
        except BaseException:
            self._lock_file.close()
            raise
        try:
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = FULL")
            self._init_schema()
            self._ensure_server_id()
            self._ensure_release_epoch()
            self._recover_blob_temps()
            self._recover_backup_pending()
            self._recover_trash()
        except BaseException:
            self._db.close()
            self._lock_file.close()
            raise

    def close(self) -> None:
        self._db.close()
        self._lock_file.close()

    @contextmanager
    def _exclusive(self):
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)

    def publish(
        self,
        manifest: WebUiManifest,
        contents: dict[str, bytes],
        *,
        stable: bool = False,
        preview: bool = True,
        actor: str = "publish-mobile-webui",
    ) -> ReleaseView:
        """在跨进程独占锁内完成 CAS 与原子 pointer 提交。"""

        with self._exclusive():
            return self._publish_locked(manifest, contents, stable=stable, preview=preview, actor=actor)

    def _publish_locked(
        self,
        manifest: WebUiManifest,
        contents: dict[str, bytes],
        *,
        stable: bool,
        preview: bool,
        actor: str,
    ) -> ReleaseView:
        """完成 CAS 后原子插入 generation、替换单一 ReleaseView 并写 journal。"""

        # 1. 校验 manifest、稳定发布资格和每个文件的真实摘要
        validate_manifest(manifest)
        if stable and (not manifest.reproducible or manifest.dirty_provenance is not None):
            raise ManifestError("Stable 发布要求 reproducible=true 且 dirty_provenance=null")
        if not stable and not preview:
            raise ValueError("publish 至少要更新 stable 或 preview")
        digest = manifest_digest(manifest)
        if set(contents) != {item.path for item in manifest.files}:
            raise ManifestError("发布内容与 manifest 文件集合不一致")
        manifest_bytes = canonical_manifest_bytes(manifest)
        self._write_blob(digest, manifest_bytes)
        for item in manifest.files:
            data = contents[item.path]
            if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
                raise ManifestError(f"发布内容摘要不匹配: {item.path}")
            self._write_blob(item.sha256, data)

        # 2. 单写者事务只更新 generation 与两个 nullable pointer
        generation_id = manifest.generation_id
        now = _now()
        self._db.execute("BEGIN IMMEDIATE")
        deleted_blobs: set[str] = set()
        moved_blobs: dict[str, Path] = {}
        try:
            existing = self._db.execute(
                "SELECT manifest_digest, manifest_json FROM webui_generations WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            if existing is not None:
                if existing["manifest_digest"] != digest or bytes(existing["manifest_json"]) != manifest_bytes:
                    raise ReleaseConflictError("generation_id 已存在但内容不同")
            else:
                target_key = derive_target_key(self.server_id, generation_id, digest)
                self._db.execute(
                    "INSERT INTO webui_generations(generation_id, target_key, manifest_digest, manifest_json, created_at, source_repository, source_commit, source_tree) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (generation_id, target_key, digest, manifest_bytes, now, manifest.source_repository, manifest.source_commit, manifest.source_tree),
                )
                for item in manifest.files:
                    self._db.execute(
                        "INSERT INTO webui_generation_files(generation_id, path, digest, size_bytes, mime) VALUES (?, ?, ?, ?, ?)",
                        (generation_id, item.path, item.sha256, item.size_bytes, item.mime),
                    )
                self._ensure_blob_row(digest, len(manifest_bytes), now)
                for item in manifest.files:
                    self._ensure_blob_row(item.sha256, item.size_bytes, now)
            state = self._db.execute("SELECT stable_generation_id, preview_generation_id, sequence FROM webui_release_state WHERE singleton = 1").fetchone()
            stable_id = state["stable_generation_id"] if state is not None else None
            preview_id = state["preview_generation_id"] if state is not None else None
            previous_stable_id = stable_id
            previous_preview_id = preview_id
            if stable:
                stable_id = generation_id
            if preview:
                preview_id = generation_id
            sequence = int(state["sequence"]) + 1 if state is not None else 1
            selection_digest = self._selection_digest(stable_id, preview_id)
            release_epoch = self._release_epoch()
            self._db.execute(
                """
                INSERT INTO webui_release_state(singleton, release_epoch, sequence, stable_generation_id, preview_generation_id, selection_digest, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET release_epoch=excluded.release_epoch, sequence=excluded.sequence,
                    stable_generation_id=excluded.stable_generation_id, preview_generation_id=excluded.preview_generation_id,
                    selection_digest=excluded.selection_digest, updated_at=excluded.updated_at
                """,
                (release_epoch, sequence, stable_id, preview_id, selection_digest, now),
            )
            self._db.execute(
                "INSERT INTO webui_publication_journal(sequence, generation_id, operation, release_epoch, stable, preview, actor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sequence, generation_id, "publish", release_epoch, int(stable), int(preview), actor, now),
            )
            if stable_id != previous_stable_id:
                self._record_channel_selection_locked("stable", stable_id)
            if preview_id != previous_preview_id:
                self._record_channel_selection_locked("preview", preview_id)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return self.get_release()

    def promote_preview(self, *, actor: str = "promote-mobile-webui-preview") -> ReleaseView:
        """把当前 Preview 原子提升为 Stable 并清除 Preview 指针。"""

        with self._exclusive():
            return self._update_pointers(stable_from_preview=True, clear_preview=True, actor=actor)

    def clear_preview(self, *, actor: str = "clear-mobile-webui-preview") -> ReleaseView:
        with self._exclusive():
            return self._update_pointers(clear_preview=True, actor=actor)

    def pin_target(self, target_key: str, *, reason: str = "rollback") -> None:
        """固定一个 generation，避免显式 GC 删除可回滚目标。"""

        with self._exclusive():
            row = self._db.execute("SELECT generation_id FROM webui_generations WHERE target_key = ?", (target_key,)).fetchone()
            if row is None:
                raise UnknownReleaseError(target_key)
            self._db.execute(
                "INSERT INTO webui_pins(target_key, generation_id, reason, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(target_key) DO UPDATE SET generation_id=excluded.generation_id, reason=excluded.reason",
                (target_key, row["generation_id"], reason, _now()),
            )

    def unpin_target(self, target_key: str) -> None:
        with self._exclusive():
            self._db.execute("DELETE FROM webui_pins WHERE target_key = ?", (target_key,))

    def rollback(self, target_key: str, *, actor: str = "rollback-mobile-webui") -> ReleaseView:
        """把已 pin 或最近 channel 选择窗口内的 immutable generation 设置为 Stable。"""

        with self._exclusive():
            return self._rollback_locked(target_key, actor=actor)

    def _rollback_locked(self, target_key: str, *, actor: str) -> ReleaseView:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            target = self._db.execute(
                """
                SELECT generation_id FROM webui_pins WHERE target_key = ?
                UNION
                SELECT generation_id FROM webui_channel_selections
                WHERE channel IN ('stable', 'preview') AND target_key = ?
                ORDER BY generation_id LIMIT 1
                """,
                (target_key, target_key),
            ).fetchone()
            if target is None:
                raise RollbackUnavailableError("rollback target 不在 pin 或最近 stable/preview selection 窗口")
            generation_id = str(target["generation_id"])
            manifest = self._manifest_for_generation(generation_id)
            state = self._db.execute("SELECT * FROM webui_release_state WHERE singleton = 1").fetchone()
            if state is None:
                raise UnknownReleaseError("尚未发布任何 WebUI generation")
            sequence = int(state["sequence"]) + 1
            preview_id = state["preview_generation_id"]
            selection = self._selection_digest(generation_id, preview_id)
            now = _now()
            self._db.execute("UPDATE webui_release_state SET sequence = ?, stable_generation_id = ?, selection_digest = ?, updated_at = ? WHERE singleton = 1", (sequence, generation_id, selection, now))
            self._db.execute("INSERT INTO webui_publication_journal(sequence, generation_id, operation, release_epoch, stable, preview, actor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (sequence, generation_id, "rollback", state["release_epoch"], 1, int(preview_id is not None), actor, now))
            if state["stable_generation_id"] != generation_id:
                self._record_channel_selection_locked("stable", generation_id)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return self.get_release()

    def record_channel_selection(self, channel: str, target_key: str) -> None:
        """记录一次成功 Resolve 选择，并保留每个 channel 最近四个 target。"""

        if channel not in {"stable", "preview"}:
            raise ValueError("channel 必须为 stable 或 preview")
        with self._exclusive():
            target = self.get_target(target_key)
            if target is None:
                raise UnknownReleaseError(target_key)
            self._record_channel_selection_locked(channel, target.generation_id)

    def get_release(self, *, verify_integrity: bool = True) -> ReleaseView:
        state = self._db.execute("SELECT * FROM webui_release_state WHERE singleton = 1").fetchone()
        if state is None:
            return ReleaseView(self.server_id, self._release_epoch(), 0, self._selection_digest(None, None), None, None)
        release_epoch_now = self._release_epoch()
        state_epoch = str(state["release_epoch"])
        _require_canonical_uuid4(state_epoch, "release state release_epoch")
        if state_epoch != release_epoch_now:
            raise RuntimeError("release state release_epoch 与当前 release_epoch 不一致")
        stable = self._target_for_generation(state["stable_generation_id"], verify_integrity=verify_integrity)
        preview = self._target_for_generation(state["preview_generation_id"], verify_integrity=verify_integrity)
        expected = self._selection_digest(state["stable_generation_id"], state["preview_generation_id"])
        if expected != state["selection_digest"]:
            raise RuntimeError("release selection_digest 损坏")
        return ReleaseView(self.server_id, str(state["release_epoch"]), int(state["sequence"]), str(state["selection_digest"]), stable, preview)

    def has_stable_publication_for_source(self, source_commit: str) -> bool:
        """判断一个 Core commit 是否曾成功成为 Stable，显式 rollback 不抹除该事实。"""

        row = self._db.execute(
            """
            SELECT 1
            FROM webui_publication_journal AS journal
            JOIN webui_generations AS generation
              ON generation.generation_id = journal.generation_id
            WHERE journal.stable = 1
              AND journal.operation IN ('publish', 'promote_preview')
              AND generation.source_commit = ?
            LIMIT 1
            """,
            (source_commit,),
        ).fetchone()
        return row is not None

    def get_target(self, target_key: str) -> WebUiTarget | None:
        return self.get_release().target(target_key)

    def get_manifest(self, digest: str) -> WebUiManifest:
        row = self._db.execute("SELECT manifest_json, manifest_digest FROM webui_generations WHERE manifest_digest = ?", (digest,)).fetchone()
        if row is None:
            raise UnknownReleaseError(digest)
        manifest = manifest_from_json(_strict_json_loads(bytes(row["manifest_json"])))
        if manifest_digest(manifest) != digest:
            raise RuntimeError("manifest digest 损坏")
        return manifest

    def read_blob(self, digest: str) -> StoredBlob:
        if not _valid_digest(digest):
            raise ValueError("blob digest 无效")
        row = self._db.execute("SELECT digest, size_bytes FROM webui_blobs WHERE digest = ?", (digest,)).fetchone()
        if row is None:
            raise UnknownReleaseError(digest)
        path = self.blob_path(digest)
        if not path.is_file():
            raise RuntimeError(f"CAS blob 缺失: {digest}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"CAS blob 摘要损坏: {digest}")
        return StoredBlob(digest, int(row["size_bytes"]), path)

    def verify_target_resource(self, *, target_key: str, selection_digest: str, resource_digest: str) -> StoredBlob:
        release = self.get_release(verify_integrity=False)
        if release.selection_digest != selection_digest:
            raise ReleaseSelectionChangedError("release selection 已变化")
        target = release.target(target_key)
        if target is None:
            raise TargetResourceNotFoundError("target 不属于当前 stable/preview")
        row = self._db.execute(
            "SELECT digest, size_bytes, mime FROM webui_generation_files WHERE generation_id = ? AND digest = ? ORDER BY path LIMIT 1",
            (target.generation_id, resource_digest),
        ).fetchone()
        if row is None:
            raise TargetResourceNotFoundError("resource 不属于当前 target")
        try:
            blob = self.read_blob(resource_digest)
        except UnknownReleaseError as error:
            raise RuntimeError("target 成员引用的 CAS metadata 缺失") from error
        if blob.size_bytes != int(row["size_bytes"]):
            raise RuntimeError("target 成员 size 与 CAS metadata 不一致")
        return StoredBlob(blob.digest, blob.size_bytes, blob.path, str(row["mime"]))

    def gc(self, *, keep_unreachable: int = 0) -> GarbageCollectionReport:
        """在跨进程独占锁内清理不可达历史。"""

        with self._exclusive():
            return self._gc_locked(keep_unreachable=keep_unreachable)

    def _gc_locked(self, *, keep_unreachable: int = 0) -> GarbageCollectionReport:
        """显式清理不可达历史，Stable/Preview 指针永不被本操作删除。"""

        if keep_unreachable < 0:
            raise ValueError("keep_unreachable 不能为负数")
        self._db.execute("BEGIN IMMEDIATE")
        deleted_blobs: set[str] = set()
        moved_blobs: dict[str, Path] = {}
        try:
            state = self._db.execute("SELECT stable_generation_id, preview_generation_id FROM webui_release_state WHERE singleton = 1").fetchone()
            protected = set()
            if state is not None:
                protected = {str(value) for value in (state["stable_generation_id"], state["preview_generation_id"]) if value is not None}
            protected.update(str(row["generation_id"]) for row in self._db.execute("SELECT generation_id FROM webui_pins").fetchall())
            protected.update(str(row["generation_id"]) for row in self._db.execute("SELECT generation_id FROM webui_channel_selections").fetchall())
            protected.update(str(row["generation_id"]) for row in self._db.execute("SELECT generation_id FROM webui_backup_sets").fetchall())
            rows = self._db.execute("SELECT generation_id FROM webui_generations ORDER BY created_at DESC, generation_id DESC").fetchall()
            candidates = [str(row["generation_id"]) for row in rows if str(row["generation_id"]) not in protected]
            remove_generations = candidates[keep_unreachable:]
            remove_blobs: set[str] = set()
            for generation_id in remove_generations:
                files = self._db.execute("SELECT digest FROM webui_generation_files WHERE generation_id = ?", (generation_id,)).fetchall()
                remove_blobs.update(str(row["digest"]) for row in files)
                row = self._db.execute("SELECT manifest_digest FROM webui_generations WHERE generation_id = ?", (generation_id,)).fetchone()
                if row is not None:
                    remove_blobs.add(str(row["manifest_digest"]))
                self._db.execute("DELETE FROM webui_generations WHERE generation_id = ?", (generation_id,))
            for digest in tuple(remove_blobs):
                ref = self._db.execute("SELECT 1 FROM webui_generations WHERE manifest_digest = ? OR generation_id IN (SELECT generation_id FROM webui_generation_files WHERE digest = ?) LIMIT 1", (digest, digest)).fetchone()
                if ref is None:
                    source = self.blob_path(digest)
                    if source.is_file():
                        trash_path = self.trash_root / digest
                        if trash_path.exists():
                            trash_path.unlink()
                        os.replace(source, trash_path)
                        moved_blobs[digest] = trash_path
                    self._db.execute("DELETE FROM webui_blobs WHERE digest = ?", (digest,))
                    deleted_blobs.add(digest)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            for digest, trash_path in moved_blobs.items():
                target = self.blob_path(digest)
                if trash_path.exists() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(trash_path, target)
            raise
        for digest in tuple(deleted_blobs):
            trash_path = moved_blobs.get(digest)
            if trash_path is not None and trash_path.exists():
                trash_path.unlink()
        known = {str(row["digest"]) for row in self._db.execute("SELECT digest FROM webui_blobs").fetchall()}
        for path in self.blob_root.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            if _valid_digest(path.name) and path.name not in known:
                path.unlink()
                deleted_blobs.add(path.name)
        return GarbageCollectionReport(tuple(remove_generations), tuple(sorted(deleted_blobs)))

    def backup_to(self, destination: Path) -> Path:
        """在跨进程独占锁内发布 SQLite、CAS 和校验清单。"""

        with self._exclusive():
            return self._backup_locked(destination)

    def _backup_locked(self, destination: Path) -> Path:
        """在独立目录发布 SQLite、CAS 和校验清单，供恢复 smoke 使用。"""

        destination = _absolute_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise RuntimeError("WebUI backup destination 不能是符号链接")
        if destination.exists():
            raise FileExistsError(destination)
        backup_id = str(uuid4())
        temporary = destination.parent / f".{destination.name}.{backup_id}.tmp"
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(temporary)
        marker = self._write_backup_pending_marker(backup_id, destination, temporary)
        published = False
        try:
            temporary.mkdir()
            generations = tuple(str(row["generation_id"]) for row in self._db.execute("SELECT generation_id FROM webui_generations").fetchall())
            self._db.execute("BEGIN IMMEDIATE")
            try:
                now = _now()
                for generation_id in generations:
                    self._db.execute("INSERT INTO webui_backup_sets(backup_id, generation_id, destination, created_at) VALUES (?, ?, ?, ?)", (backup_id, generation_id, str(destination), now))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            backup_db = temporary / "publication.sqlite3"
            target = sqlite3.connect(backup_db)
            try:
                self._db.backup(target)
            finally:
                target.close()
            snapshot = sqlite3.connect(backup_db)
            snapshot.row_factory = sqlite3.Row
            try:
                blob_rows = snapshot.execute("SELECT digest FROM webui_blobs").fetchall()
            finally:
                snapshot.close()
            for row in blob_rows:
                digest = str(row["digest"])
                source = self.blob_path(digest)
                if not source.is_file():
                    raise RuntimeError(f"backup 时 CAS blob 缺失: {digest}")
                target_path = temporary / "blobs" / "sha256" / digest[:2] / digest
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target_path)
            descriptor = {"backup_id": backup_id, "server_id": self.server_id, "generation_ids": generations, "sha256": _tree_digest(temporary), "created_at": _now()}
            descriptor_path = temporary / "backup.json"
            with descriptor_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(descriptor, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_tree(temporary)
            if destination.is_symlink() or destination.exists():
                raise FileExistsError(destination)
            os.replace(temporary, destination)
            published = True
            self._fsync_directory(destination.parent)
            self._clear_backup_pending_marker(marker)
            return destination
        except BaseException:
            if published:
                raise
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM webui_backup_sets WHERE backup_id = ?", (backup_id,))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            self._clear_backup_pending_marker(marker)
            raise

    def release_backup(self, backup_id: str) -> None:
        """显式释放 backup source set，之后 GC 才可回收其历史 generation。"""

        with self._exclusive():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM webui_backup_sets WHERE backup_id = ?", (backup_id,))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    @staticmethod
    def restore_backup(destination: Path, target_root: Path, *, server_id: str, pre_restore_backup: Path | None = None) -> Path:
        """验证 backup 后原子替换 target，并为旧 store 留下可恢复副本。"""

        MobileWebUiStore.verify_backup(destination, server_id=server_id)
        target_root = _absolute_path(target_root)
        MobileWebUiStore._recover_restore_marker(target_root, server_id=server_id)
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.is_symlink() or (target_root.exists() and not target_root.is_dir()):
            raise RuntimeError("WebUI restore target 必须是普通目录")
        temporary = Path(tempfile.mkdtemp(prefix=f".{target_root.name}.", dir=target_root.parent))
        old_root: Path | None = None
        restore_marker: Path | None = None
        try:
            shutil.copytree(destination, temporary, dirs_exist_ok=True)
            MobileWebUiStore._append_restore_journal(temporary, server_id=server_id)
            MobileWebUiStore._refresh_backup_descriptor(temporary)
            MobileWebUiStore.verify_backup(temporary, server_id=server_id)
            # 1. 先把候选树的内容和目录项完整落盘，再写 marker 或移动旧 root。
            MobileWebUiStore._fsync_tree(temporary)
            expected_tree_digest = _tree_digest(temporary)
            had_target = target_root.exists()
            recovery_root = target_root.with_name(f"{target_root.name}.recovery-{uuid4()}")
            if had_target:
                backup_path = pre_restore_backup or target_root.with_name(f"{target_root.name}.pre-restore-{uuid4()}")
                if backup_path.exists():
                    raise FileExistsError(backup_path)
                existing = MobileWebUiStore(target_root, server_id=server_id)
                try:
                    existing.backup_to(backup_path)
                    MobileWebUiStore.verify_backup(backup_path, server_id=server_id)
                finally:
                    existing.close()
                old_root = recovery_root
            restore_marker = MobileWebUiStore._write_restore_marker(
                target_root,
                recovery_root,
                temporary_root=temporary,
                server_id=server_id,
                expected_tree_digest=expected_tree_digest,
                had_target=had_target,
            )
            if old_root is not None:
                os.replace(target_root, old_root)
                MobileWebUiStore._fsync_directory(target_root.parent)
            try:
                os.replace(temporary, target_root)
                MobileWebUiStore._fsync_directory(target_root.parent)
            except BaseException:
                if old_root is not None and not target_root.exists():
                    os.replace(old_root, target_root)
                    MobileWebUiStore._fsync_directory(target_root.parent)
                raise
            if restore_marker is not None:
                if old_root is not None:
                    MobileWebUiStore._remove_restore_recovery(old_root)
                MobileWebUiStore._clear_restore_marker(restore_marker)
            return target_root
        except BaseException:
            # 恢复标记存在时，把候选树和旧 root 交给启动恢复；不能提前清掉唯一证据。
            if restore_marker is None:
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def verify_backup(destination: Path, *, server_id: str) -> None:
        """在不连接正式 store 的前提下校验备份 SQLite、manifest 和全部 CAS。"""

        destination = _absolute_path(destination)
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError("WebUI backup destination 必须是普通目录")
        db_path = destination / "publication.sqlite3"
        if not db_path.is_file():
            raise RuntimeError("WebUI backup 缺少 publication.sqlite3")
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        try:
            integrity = db.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise RuntimeError("WebUI backup SQLite integrity_check 失败")
            descriptor_path = destination / "backup.json"
            if not descriptor_path.is_file():
                raise RuntimeError("WebUI backup 缺少 backup.json")
            descriptor = _strict_json_loads(descriptor_path.read_bytes())
            if not isinstance(descriptor, dict):
                raise RuntimeError("WebUI backup descriptor 字段不符合合同")
            if descriptor.get("server_id") != server_id or descriptor.get("sha256") != _tree_digest(destination):
                raise RuntimeError("WebUI backup descriptor 校验失败")
            meta = db.execute("SELECT value FROM webui_meta WHERE key = 'server_id'").fetchone()
            if meta is None or str(meta["value"]) != server_id:
                raise RuntimeError("WebUI backup server_id 不匹配")
            release_row = db.execute("SELECT value FROM webui_meta WHERE key = 'release_epoch'").fetchone()
            if release_row is None:
                raise RuntimeError("WebUI backup 缺少 release_epoch")
            release_epoch = str(release_row["value"])
            _require_canonical_uuid4(release_epoch, "WebUI backup release_epoch")
            rows = db.execute("SELECT digest, size_bytes FROM webui_blobs").fetchall()
            for row in rows:
                digest = str(row["digest"])
                path = destination / "blobs" / "sha256" / digest[:2] / digest
                if not path.is_file() or len(path.read_bytes()) != int(row["size_bytes"]) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"WebUI backup CAS 损坏: {digest}")
            generations = db.execute("SELECT generation_id, target_key, manifest_digest, manifest_json FROM webui_generations").fetchall()
            generation_ids = {str(row["generation_id"]) for row in generations}
            descriptor_ids = descriptor.get("generation_ids")
            if not isinstance(descriptor_ids, list) or set(descriptor_ids) != generation_ids or len(descriptor_ids) != len(generation_ids):
                raise RuntimeError("WebUI backup generation source set 不完整")
            for row in generations:
                try:
                    manifest = manifest_from_json(_strict_json_loads(bytes(row["manifest_json"])))
                except ManifestError as error:
                    raise RuntimeError(f"WebUI backup manifest JSON 损坏: {row['generation_id']}") from error
                digest = str(row["manifest_digest"])
                if manifest_digest(manifest) != digest:
                    raise RuntimeError(f"WebUI backup manifest 损坏: {row['generation_id']}")
                expected_target_key = derive_target_key(server_id, str(row["generation_id"]), digest)
                if str(row["target_key"]) != expected_target_key:
                    raise RuntimeError(f"WebUI backup target_key 损坏: {row['generation_id']}")
                member_rows = db.execute("SELECT path, digest, size_bytes, mime FROM webui_generation_files WHERE generation_id = ?", (row["generation_id"],)).fetchall()
                expected = {item.path: (item.sha256, item.size_bytes, item.mime) for item in manifest.files}
                actual = {str(item["path"]): (str(item["digest"]), int(item["size_bytes"]), str(item["mime"])) for item in member_rows}
                if actual != expected:
                    raise RuntimeError(f"WebUI backup generation files 损坏: {row['generation_id']}")
            state = db.execute("SELECT * FROM webui_release_state WHERE singleton = 1").fetchone()
            if state is not None:
                if str(state["release_epoch"]) != str(release_row["value"]):
                    raise RuntimeError("WebUI backup release epoch 与当前 release_epoch 不一致")
                stable_id = state["stable_generation_id"]
                preview_id = state["preview_generation_id"]
                for pointer in (stable_id, preview_id):
                    if pointer is not None and str(pointer) not in generation_ids:
                        raise RuntimeError("WebUI backup release pointer 不可达")
                stable_key = next((str(row["target_key"]) for row in generations if row["generation_id"] == stable_id), None)
                preview_key = next((str(row["target_key"]) for row in generations if row["generation_id"] == preview_id), None)
                selection_body = json.dumps(
                    {"server_id": server_id, "stable_target_key": stable_key, "preview_target_key": preview_key},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                if str(state["selection_digest"]) != hashlib.sha256(selection_body).hexdigest():
                    raise RuntimeError("WebUI backup selection_digest 损坏")
                journal_max = db.execute("SELECT MAX(sequence) AS value FROM webui_publication_journal").fetchone()["value"]
                if journal_max is None or int(journal_max) != int(state["sequence"]):
                    raise RuntimeError("WebUI backup journal 与 release sequence 不一致")
            journal_rows = db.execute("SELECT sequence, generation_id, operation, release_epoch, stable, preview FROM webui_publication_journal ORDER BY sequence").fetchall()
            if state is None:
                for row in journal_rows:
                    if str(row["operation"]) != "restore" or row["generation_id"] is not None or int(row["stable"]) != 0 or int(row["preview"]) != 0:
                        raise RuntimeError("WebUI backup 无 release state 时仅允许空指针 restore journal")
            for expected_sequence, row in enumerate(journal_rows, start=1):
                if int(row["sequence"]) != expected_sequence or str(row["release_epoch"]) != str(release_row["value"]):
                    raise RuntimeError("WebUI backup journal sequence/epoch 不连续")
        finally:
            db.close()

    def _update_pointers(self, *, stable_from_preview: bool = False, clear_preview: bool = False, actor: str) -> ReleaseView:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            state = self._db.execute("SELECT * FROM webui_release_state WHERE singleton = 1").fetchone()
            if state is None:
                raise UnknownReleaseError("尚未发布任何 WebUI generation")
            stable_id = state["preview_generation_id"] if stable_from_preview else state["stable_generation_id"]
            preview_id = None if clear_preview else state["preview_generation_id"]
            previous_stable_id = state["stable_generation_id"]
            previous_preview_id = state["preview_generation_id"]
            if stable_from_preview:
                if state["preview_generation_id"] is None:
                    raise UnknownReleaseError("没有可提升的 Preview")
                manifest = self._manifest_for_generation(state["preview_generation_id"])
                if not manifest.reproducible or manifest.dirty_provenance is not None:
                    raise ManifestError("Preview 必须 reproducible 且无 dirty provenance 才能 promotion")
            sequence = int(state["sequence"]) + 1
            selection = self._selection_digest(stable_id, preview_id)
            now = _now()
            self._db.execute("UPDATE webui_release_state SET sequence = ?, stable_generation_id = ?, preview_generation_id = ?, selection_digest = ?, updated_at = ? WHERE singleton = 1", (sequence, stable_id, preview_id, selection, now))
            generation_id = stable_id or preview_id
            operation = "promote_preview" if stable_from_preview else "clear_preview"
            self._db.execute("INSERT INTO webui_publication_journal(sequence, generation_id, operation, release_epoch, stable, preview, actor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (sequence, generation_id, operation, state["release_epoch"], int(stable_from_preview), int(not clear_preview), actor, now))
            if stable_id != previous_stable_id:
                self._record_channel_selection_locked("stable", stable_id)
            if preview_id != previous_preview_id:
                self._record_channel_selection_locked("preview", preview_id)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return self.get_release()

    def _target_for_generation(self, generation_id: str | None, *, verify_integrity: bool = True) -> WebUiTarget | None:
        if generation_id is None:
            return None
        row = self._db.execute("SELECT manifest_json, manifest_digest, target_key FROM webui_generations WHERE generation_id = ?", (generation_id,)).fetchone()
        if row is None:
            raise RuntimeError("release pointer 引用了不存在的 generation")
        manifest = manifest_from_json(_strict_json_loads(bytes(row["manifest_json"])))
        target = manifest.target(self.server_id, str(row["manifest_digest"]))
        if str(row["target_key"]) != target.target_key:
            raise RuntimeError("generation target_key 损坏")
        if verify_integrity:
            self._verify_generation_complete(generation_id, manifest, str(row["manifest_digest"]))
        return target

    def _manifest_for_generation(self, generation_id: str) -> WebUiManifest:
        row = self._db.execute("SELECT manifest_json, manifest_digest FROM webui_generations WHERE generation_id = ?", (generation_id,)).fetchone()
        if row is None:
            raise RuntimeError("release pointer 引用了不存在的 generation")
        manifest = manifest_from_json(_strict_json_loads(bytes(row["manifest_json"])))
        digest = str(row["manifest_digest"])
        if manifest_digest(manifest) != digest:
            raise RuntimeError("generation manifest digest 损坏")
        self._verify_generation_complete(generation_id, manifest, digest)
        return manifest

    def _verify_generation_complete(self, generation_id: str, manifest: WebUiManifest, digest: str) -> None:
        manifest_blob = self.read_blob(digest)
        if manifest_blob.path.read_bytes() != canonical_manifest_bytes(manifest):
            raise RuntimeError("generation manifest CAS 不完整")
        rows = self._db.execute("SELECT path, digest, size_bytes, mime FROM webui_generation_files WHERE generation_id = ?", (generation_id,)).fetchall()
        expected = {item.path: (item.sha256, item.size_bytes, item.mime) for item in manifest.files}
        actual = {str(row["path"]): (str(row["digest"]), int(row["size_bytes"]), str(row["mime"])) for row in rows}
        if actual != expected:
            raise RuntimeError("generation file metadata 不完整")
        for file_digest, size_bytes, _mime in actual.values():
            blob = self.read_blob(file_digest)
            if blob.size_bytes != size_bytes:
                raise RuntimeError("generation member size 与 CAS metadata 不一致")

    def _selection_digest(self, stable_id: str | None, preview_id: str | None) -> str:
        stable_key = self._target_key_for_generation(stable_id)
        preview_key = self._target_key_for_generation(preview_id)
        body = json.dumps({"server_id": self.server_id, "stable_target_key": stable_key, "preview_target_key": preview_key}, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def _ensure_blob_row(self, digest: str, size_bytes: int, created_at: str) -> None:
        """Ensure CAS owns only digest/size; MIME remains generation-file metadata."""

        row = self._db.execute("SELECT size_bytes FROM webui_blobs WHERE digest = ?", (digest,)).fetchone()
        if row is not None:
            if int(row["size_bytes"]) != size_bytes:
                raise ReleaseConflictError(f"CAS digest 已存在但 size 不一致: {digest}")
            return
        # Keep the legacy column for on-disk compatibility; never use it for HTTP semantics.
        self._db.execute(
            "INSERT INTO webui_blobs(digest, size_bytes, mime, created_at) VALUES (?, ?, ?, ?)",
            (digest, size_bytes, "application/octet-stream", created_at),
        )

    def _target_key_for_generation(self, generation_id: str | None) -> str | None:
        if generation_id is None:
            return None
        row = self._db.execute("SELECT target_key FROM webui_generations WHERE generation_id = ?", (generation_id,)).fetchone()
        if row is None:
            raise RuntimeError("release pointer 引用的 generation 不存在")
        return str(row["target_key"])

    def _record_channel_selection_locked(self, channel: str, generation_id: str | None) -> None:
        if generation_id is None:
            return
        row = self._db.execute("SELECT target_key FROM webui_generations WHERE generation_id = ?", (generation_id,)).fetchone()
        if row is None:
            raise RuntimeError("selection pointer generation 不存在")
        existing = self._db.execute(
            "SELECT selection_id FROM webui_channel_selections WHERE channel = ? AND target_key = ? LIMIT 1",
            (channel, row["target_key"]),
        ).fetchone()
        if existing is not None:
            self._db.execute("DELETE FROM webui_channel_selections WHERE selection_id = ?", (existing["selection_id"],))
        self._db.execute(
            "INSERT INTO webui_channel_selections(channel, target_key, generation_id, selected_at) VALUES (?, ?, ?, ?)",
            (channel, row["target_key"], generation_id, _now()),
        )
        rows = self._db.execute("SELECT selection_id FROM webui_channel_selections WHERE channel = ? ORDER BY selection_id DESC", (channel,)).fetchall()
        for old in rows[4:]:
            self._db.execute("DELETE FROM webui_channel_selections WHERE selection_id = ?", (old["selection_id"],))

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS webui_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS webui_generations (
                generation_id TEXT PRIMARY KEY, target_key TEXT NOT NULL UNIQUE,
                manifest_digest TEXT NOT NULL UNIQUE, manifest_json BLOB NOT NULL,
                created_at TEXT NOT NULL, source_repository TEXT NOT NULL,
                source_commit TEXT NOT NULL, source_tree TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webui_generation_files (
                generation_id TEXT NOT NULL REFERENCES webui_generations(generation_id) ON DELETE CASCADE,
                path TEXT NOT NULL, digest TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                mime TEXT NOT NULL, PRIMARY KEY(generation_id, path)
            );
            CREATE INDEX IF NOT EXISTS webui_generation_files_digest ON webui_generation_files(digest);
            CREATE TABLE IF NOT EXISTS webui_blobs (
                digest TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mime TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webui_release_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1), release_epoch TEXT NOT NULL,
                sequence INTEGER NOT NULL, stable_generation_id TEXT REFERENCES webui_generations(generation_id),
                preview_generation_id TEXT REFERENCES webui_generations(generation_id), selection_digest TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webui_pins (
                target_key TEXT PRIMARY KEY,
                generation_id TEXT NOT NULL REFERENCES webui_generations(generation_id),
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webui_channel_selections (
                selection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                target_key TEXT NOT NULL,
                generation_id TEXT NOT NULL REFERENCES webui_generations(generation_id),
                selected_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webui_backup_sets (
                backup_id TEXT NOT NULL,
                generation_id TEXT NOT NULL REFERENCES webui_generations(generation_id),
                destination TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(backup_id, generation_id)
            );
            CREATE TABLE IF NOT EXISTS webui_publication_journal (
                sequence INTEGER PRIMARY KEY, generation_id TEXT, operation TEXT NOT NULL, release_epoch TEXT NOT NULL,
                stable INTEGER NOT NULL, preview INTEGER NOT NULL, actor TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )

    def _ensure_server_id(self) -> None:
        row = self._db.execute("SELECT value FROM webui_meta WHERE key = 'server_id'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO webui_meta(key, value) VALUES ('server_id', ?)", (self.server_id,))
        elif str(row["value"]) != self.server_id:
            raise ValueError("mobile WebUI publication store server_id 不匹配")

    def _ensure_release_epoch(self) -> None:
        row = self._db.execute("SELECT value FROM webui_meta WHERE key = 'release_epoch'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO webui_meta(key, value) VALUES ('release_epoch', ?)", (str(uuid4()),))
            return
        _require_canonical_uuid4(str(row["value"]), "WebUI release_epoch")

    def _release_epoch(self) -> str:
        row = self._db.execute("SELECT value FROM webui_meta WHERE key = 'release_epoch'").fetchone()
        if row is None:
            raise RuntimeError("WebUI release_epoch 缺失")
        value = str(row["value"])
        _require_canonical_uuid4(value, "WebUI release_epoch")
        return value

    @staticmethod
    def _restore_marker_path(root: Path) -> Path:
        root = _absolute_path(root)
        return root.with_name(f".{root.name}.restore-pending.json")

    @staticmethod
    def _recover_restore_marker(root: Path, *, server_id: str | None = None) -> None:
        """恢复被中断的 store 切换，不接受未经验证的新 root。"""

        target_root = _absolute_path(root)
        marker = MobileWebUiStore._restore_marker_path(target_root)
        if not marker.exists() and not marker.is_symlink():
            return
        if marker.is_symlink() or not marker.is_file():
            raise RuntimeError("WebUI restore marker 不是普通文件")
        try:
            payload = _strict_json_loads(marker.read_bytes())
        except (OSError, ManifestError) as error:
            raise RuntimeError("WebUI restore marker 损坏") from error
        expected_fields = {
            "expected_tree_digest",
            "had_target",
            "recovery_root",
            "server_id",
            "target_root",
            "temporary_root",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise RuntimeError("WebUI restore marker 字段不符合合同")
        if payload["target_root"] != str(target_root):
            raise RuntimeError("WebUI restore marker target 路径不符合合同")
        marker_server_id = payload["server_id"]
        expected_tree_digest = payload["expected_tree_digest"]
        had_target = payload["had_target"]
        if (
            not isinstance(marker_server_id, str)
            or (server_id is not None and marker_server_id != server_id)
            or not isinstance(expected_tree_digest, str)
            or not _valid_digest(expected_tree_digest)
            or not isinstance(had_target, bool)
        ):
            raise RuntimeError("WebUI restore marker identity 不符合合同")
        recovery_root = _absolute_path(_require_marker_path(payload["recovery_root"], "recovery_root"))
        temporary_root = _absolute_path(_require_marker_path(payload["temporary_root"], "temporary_root"))
        if (
            recovery_root == target_root
            or recovery_root.parent != target_root.parent
            or not recovery_root.name.startswith(f"{target_root.name}.recovery-")
        ):
            raise RuntimeError("WebUI restore marker recovery 路径不符合合同")
        if (
            temporary_root == target_root
            or temporary_root == recovery_root
            or temporary_root.parent != target_root.parent
            or not temporary_root.name.startswith(f".{target_root.name}.")
        ):
            raise RuntimeError("WebUI restore marker temporary 路径不符合合同")
        if target_root.is_symlink():
            raise RuntimeError("WebUI restore marker target 不能是符号链接")
        if recovery_root.is_symlink():
            raise RuntimeError("WebUI restore marker recovery 不能是符号链接")
        if temporary_root.is_symlink():
            raise RuntimeError("WebUI restore marker temporary 不能是符号链接")
        target_exists = target_root.exists()
        recovery_exists = recovery_root.exists()
        if target_exists and not target_root.is_dir():
            raise RuntimeError("WebUI restore marker target 必须是普通目录")
        if recovery_exists and not recovery_root.is_dir():
            raise RuntimeError("WebUI restore marker recovery 必须是普通目录")

        # 1. 两个 root 同时存在时，先完整验证新候选再清理 marker。
        if target_exists and recovery_exists:
            if not had_target:
                raise RuntimeError("WebUI restore marker 无旧 target 却同时存在 recovery")
            try:
                MobileWebUiStore._verify_restored_root(
                    target_root,
                    server_id=marker_server_id,
                    expected_tree_digest=expected_tree_digest,
                )
            except (ManifestError, OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as error:
                failed_root = target_root.with_name(f"{target_root.name}.restore-failed-{uuid4().hex}")
                if failed_root.exists() or failed_root.is_symlink():
                    raise FileExistsError(failed_root) from error
                os.replace(target_root, failed_root)
                MobileWebUiStore._fsync_directory(target_root.parent)
                os.replace(recovery_root, target_root)
                MobileWebUiStore._fsync_directory(target_root.parent)
                MobileWebUiStore._remove_restore_temporary(temporary_root)
                MobileWebUiStore._clear_restore_marker(marker)
                return
            MobileWebUiStore._remove_restore_recovery(recovery_root)
            MobileWebUiStore._remove_restore_temporary(temporary_root)
            MobileWebUiStore._clear_restore_marker(marker)
            return

        # 2. 只有 target 时，按 marker 区分旧 root 与无旧 root 的新候选。
        if target_exists:
            if not had_target:
                MobileWebUiStore._verify_restored_root(
                    target_root,
                    server_id=marker_server_id,
                    expected_tree_digest=expected_tree_digest,
                )
            MobileWebUiStore._remove_restore_temporary(temporary_root)
            MobileWebUiStore._clear_restore_marker(marker)
            return

        # 3. 无旧 root 时，先验证 marker 绑定的完整 temporary，再完成安装。
        if not had_target and not recovery_exists:
            MobileWebUiStore._verify_restored_root(
                temporary_root,
                server_id=marker_server_id,
                expected_tree_digest=expected_tree_digest,
            )
            target_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_root, target_root)
            MobileWebUiStore._fsync_directory(target_root.parent)
            MobileWebUiStore._clear_restore_marker(marker)
            return

        # 4. rename gap 没有 target，先恢复 marker 所有的旧 root。
        if recovery_exists:
            if not had_target:
                raise RuntimeError("WebUI restore marker 无旧 target 却存在 recovery")
            target_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(recovery_root, target_root)
            MobileWebUiStore._fsync_directory(target_root.parent)
            MobileWebUiStore._remove_restore_temporary(temporary_root)
            MobileWebUiStore._clear_restore_marker(marker)
            return
        raise RuntimeError("WebUI restore marker 同时缺少 target 和 recovery")

    @staticmethod
    def _verify_restored_root(root: Path, *, server_id: str, expected_tree_digest: str) -> None:
        """校验新安装的备份树及恢复标记绑定的身份。"""

        MobileWebUiStore.verify_backup(root, server_id=server_id)
        if _tree_digest(root) != expected_tree_digest:
            raise RuntimeError("WebUI restore target 与 marker 绑定摘要不一致")

    @staticmethod
    def _remove_restore_temporary(root: Path) -> None:
        """只删除持久恢复标记指定的 temporary 目录。"""

        if root.is_symlink():
            raise RuntimeError("WebUI restore temporary 不能是符号链接")
        if root.is_dir():
            shutil.rmtree(root)
            MobileWebUiStore._fsync_directory(root.parent)
        elif root.exists():
            raise RuntimeError("WebUI restore temporary 必须是目录")

    @staticmethod
    def _remove_restore_recovery(root: Path) -> None:
        """新 root 验证通过后，持久删除旧 root。"""

        if root.is_symlink():
            raise RuntimeError("WebUI restore recovery 不能是符号链接")
        if root.is_dir():
            shutil.rmtree(root)
            MobileWebUiStore._fsync_directory(root.parent)
        elif root.exists():
            raise RuntimeError("WebUI restore recovery 必须是目录")

    @staticmethod
    def _write_restore_marker(
        target_root: Path,
        recovery_root: Path,
        *,
        temporary_root: Path,
        server_id: str,
        expected_tree_digest: str,
        had_target: bool,
    ) -> Path:
        target_root = _absolute_path(target_root)
        recovery_root = _absolute_path(recovery_root)
        temporary_root = _absolute_path(temporary_root)
        if not _valid_digest(expected_tree_digest):
            raise ValueError("WebUI restore marker expected_tree_digest 无效")
        marker = MobileWebUiStore._restore_marker_path(target_root)
        temporary = marker.with_name(f".{marker.name}.{uuid4().hex}.tmp")
        payload = json.dumps(
            {
                "expected_tree_digest": expected_tree_digest,
                "had_target": had_target,
                "recovery_root": str(recovery_root),
                "server_id": server_id,
                "target_root": str(target_root),
                "temporary_root": str(temporary_root),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        MobileWebUiStore._fsync_directory(marker.parent)
        return marker

    @staticmethod
    def _clear_restore_marker(marker: Path) -> None:
        if marker.exists() or marker.is_symlink():
            marker.unlink()
            MobileWebUiStore._fsync_directory(marker.parent)

    @staticmethod
    def _append_restore_journal(root: Path, *, server_id: str) -> None:
        """Append a restore audit event while preserving the restored pointer state."""

        db_path = root / "publication.sqlite3"
        db = sqlite3.connect(db_path, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = FULL")
            meta = db.execute("SELECT value FROM webui_meta WHERE key = 'server_id'").fetchone()
            if meta is None or str(meta["value"]) != server_id:
                raise RuntimeError("restore backup server_id 不匹配")
            release_row = db.execute("SELECT value FROM webui_meta WHERE key = 'release_epoch'").fetchone()
            if release_row is None:
                raise RuntimeError("restore backup 缺少 release_epoch")
            release_epoch = str(release_row["value"])
            _require_canonical_uuid4(release_epoch, "restore backup release_epoch")
            db.execute("BEGIN IMMEDIATE")
            try:
                state = db.execute("SELECT stable_generation_id, preview_generation_id, sequence FROM webui_release_state WHERE singleton = 1").fetchone()
                if state is None:
                    latest = db.execute("SELECT MAX(sequence) AS value FROM webui_publication_journal").fetchone()["value"]
                    sequence = int(latest or 0) + 1
                    generation_id = None
                    stable = 0
                    preview = 0
                else:
                    sequence = int(state["sequence"]) + 1
                    generation_id = state["stable_generation_id"] or state["preview_generation_id"]
                    stable = int(state["stable_generation_id"] is not None)
                    preview = int(state["preview_generation_id"] is not None)
                    db.execute("UPDATE webui_release_state SET sequence = ?, updated_at = ? WHERE singleton = 1", (sequence, _now()))
                db.execute(
                    "INSERT INTO webui_publication_journal(sequence, generation_id, operation, release_epoch, stable, preview, actor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (sequence, generation_id, "restore", release_epoch, stable, preview, "restore-mobile-webui", _now()),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise RuntimeError("restore backup WAL checkpoint 失败")
        finally:
            db.close()

    @staticmethod
    def _refresh_backup_descriptor(root: Path) -> None:
        descriptor_path = root / "backup.json"
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("restore backup descriptor 损坏") from error
        if not isinstance(descriptor, dict) or set(descriptor) != {"backup_id", "server_id", "generation_ids", "sha256", "created_at"}:
            raise RuntimeError("restore backup descriptor 字段不符合合同")
        descriptor["sha256"] = _tree_digest(root)
        descriptor_path.write_text(json.dumps(descriptor, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        with descriptor_path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        """Durably flush a completed backup tree before its atomic rename."""

        # 1. Flush file contents before making the directory entry durable.
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        # 2. Flush each directory from the leaves back to the backup root.
        directories = sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True)
        for path in (*directories, root):
            MobileWebUiStore._fsync_directory(path)

    def _write_backup_pending_marker(self, backup_id: str, destination: Path, temporary: Path) -> Path:
        """Persist the owner record that makes a pre-rename backup recoverable."""

        marker = self.staging_root / f".backup-{backup_id}.pending.json"
        marker_temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
        payload = {
            "backup_id": backup_id,
            "destination": str(destination),
            "server_id": self.server_id,
            "temporary": str(temporary),
        }
        with marker_temporary.open("xb") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(marker_temporary, marker)
        self._fsync_directory(marker.parent)
        return marker

    @staticmethod
    def _clear_backup_pending_marker(marker: Path) -> None:
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise RuntimeError("WebUI backup pending marker 不是普通文件")
        if marker.exists():
            marker.unlink()
            MobileWebUiStore._fsync_directory(marker.parent)

    @staticmethod
    def _remove_pending_temporary(temporary: Path) -> None:
        if temporary.is_symlink():
            raise RuntimeError("WebUI backup pending temporary 不能是符号链接")
        if temporary.is_dir():
            shutil.rmtree(temporary)
        elif temporary.exists():
            temporary.unlink()

    def _recover_backup_pending(self) -> None:
        """Recover only store-owned backup registrations with no valid destination."""

        with self._exclusive():
            for marker_temporary in self.staging_root.iterdir():
                if re.fullmatch(r"\.backup-[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.pending\.json\.\d+\.tmp", marker_temporary.name):
                    if marker_temporary.is_symlink():
                        raise RuntimeError("WebUI backup pending marker temporary 不能是符号链接")
                    if marker_temporary.is_file():
                        marker_temporary.unlink()
            self._fsync_directory(self.staging_root)
            for marker in sorted(self.staging_root.glob(".backup-*.pending.json")):
                if marker.is_symlink() or not marker.is_file():
                    raise RuntimeError("WebUI backup pending marker 不是普通文件")
                backup_id, destination, temporary = self._read_backup_pending_marker(marker)
                rows = self._db.execute(
                    "SELECT DISTINCT destination FROM webui_backup_sets WHERE backup_id = ?",
                    (backup_id,),
                ).fetchall()
                destinations = {str(row["destination"]) for row in rows}
                if destinations and destinations != {str(destination)}:
                    raise RuntimeError("WebUI backup pending registration destination 不一致")
                published = self._is_valid_published_backup(destination, backup_id)
                if rows and not published:
                    self._db.execute("BEGIN IMMEDIATE")
                    try:
                        self._db.execute("DELETE FROM webui_backup_sets WHERE backup_id = ?", (backup_id,))
                        self._db.execute("COMMIT")
                    except BaseException:
                        self._db.execute("ROLLBACK")
                        raise
                self._remove_pending_temporary(temporary)
                self._clear_backup_pending_marker(marker)

    def _read_backup_pending_marker(self, marker: Path) -> tuple[str, Path, Path]:
        """Validate a pending marker and derive its only permitted temporary path."""

        try:
            payload = _strict_json_loads(marker.read_bytes())
        except (OSError, ManifestError) as error:
            raise RuntimeError("WebUI backup pending marker 损坏") from error
        if not isinstance(payload, dict) or set(payload) != {"backup_id", "destination", "server_id", "temporary"}:
            raise RuntimeError("WebUI backup pending marker 字段不符合合同")
        backup_id = payload["backup_id"]
        destination_raw = payload["destination"]
        temporary_raw = payload["temporary"]
        if not isinstance(backup_id, str) or not isinstance(destination_raw, str) or not isinstance(temporary_raw, str):
            raise RuntimeError("WebUI backup pending marker 类型不符合合同")
        _require_canonical_uuid4(backup_id, "WebUI backup pending backup_id")
        if marker.name != f".backup-{backup_id}.pending.json":
            raise RuntimeError("WebUI backup pending marker 文件名不符合合同")
        if payload["server_id"] != self.server_id:
            raise RuntimeError("WebUI backup pending marker server_id 不匹配")
        destination = _absolute_path(Path(destination_raw))
        if destination_raw != str(destination):
            raise RuntimeError("WebUI backup pending destination 必须是规范绝对路径")
        temporary = destination.parent / f".{destination.name}.{backup_id}.tmp"
        if temporary_raw != str(temporary):
            raise RuntimeError("WebUI backup pending temporary 路径不符合合同")
        return backup_id, destination, temporary

    def _is_valid_published_backup(self, destination: Path, backup_id: str) -> bool:
        """Return whether destination is a complete backup for this pending owner."""

        if destination.is_symlink() or not destination.is_dir():
            return False
        descriptor_path = destination / "backup.json"
        if descriptor_path.is_symlink() or not descriptor_path.is_file():
            return False
        try:
            descriptor = _strict_json_loads(descriptor_path.read_bytes())
            if not isinstance(descriptor, dict) or descriptor.get("backup_id") != backup_id:
                return False
            MobileWebUiStore.verify_backup(destination, server_id=self.server_id)
        except (OSError, ManifestError, RuntimeError, sqlite3.Error, UnicodeError, ValueError, TypeError):
            return False
        return True

    def _recover_blob_temps(self) -> None:
        """Remove only temp names emitted by this store's digest-addressed CAS writer."""

        with self._exclusive():
            for path in self.blob_root.glob("[0-9a-f][0-9a-f]/*"):
                match = re.fullmatch(r"\.([0-9a-f]{64})\.\d+\.tmp", path.name)
                if match is None or path.parent.name != match.group(1)[:2]:
                    continue
                if path.is_file() or path.is_symlink():
                    path.unlink()

    def _recover_trash(self) -> None:
        """启动时恢复未完成 GC 的 CAS 临时 rename，并清除无引用 trash。"""

        with self._exclusive():
            for path in tuple(item for item in self.trash_root.rglob("*") if item.is_file()):
                digest = path.name
                row = self._db.execute("SELECT 1 FROM webui_blobs WHERE digest = ?", (digest,)).fetchone()
                target = self.blob_path(digest)
                if row is not None and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path, target)
                elif path.exists():
                    path.unlink()

    def _write_blob(self, digest: str, data: bytes) -> None:
        if hashlib.sha256(data).hexdigest() != digest:
            raise ManifestError("CAS blob digest 不匹配")
        path = self.blob_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file() or path.read_bytes() != data:
                raise ReleaseConflictError(f"CAS digest 已存在但内容不同: {digest}")
            return
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def blob_path(self, digest: str) -> Path:
        return self.blob_root / digest[:2] / digest


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _absolute_path(path: Path) -> Path:
    """Make a lexical absolute path without following the destination symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_marker_path(value: object, label: str) -> Path:
    """在访问文件系统前校验恢复标记路径。"""

    if not isinstance(value, str):
        raise RuntimeError(f"WebUI restore marker {label} 路径类型无效")
    path = _absolute_path(Path(value))
    if value != str(path):
        raise RuntimeError(f"WebUI restore marker {label} 必须是规范绝对路径")
    return path


def _strict_json_loads(payload: bytes) -> object:
    """Parse persisted manifest JSON without duplicate fields or non-standard numbers."""

    def reject_constant(value: str) -> object:
        raise ManifestError(f"manifest JSON 不允许常量: {value}")

    def reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"manifest JSON 存在重复字段: {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_constant,
        )
    except ManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError("manifest JSON 无效") from error


def _require_canonical_uuid4(value: str, label: str) -> None:
    """要求持久化的 release_epoch 值符合 UUID4 格式。"""

    try:
        parsed = UUID(value)
    except (AttributeError, ValueError, TypeError) as error:
        raise RuntimeError(f"{label} 必须是规范 UUID4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise RuntimeError(f"{label} 必须是规范小写 UUID4")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    ignored = {"backup.json", "publication.sqlite3-wal", "publication.sqlite3-shm"}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in ignored):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


__all__ = [
    "GarbageCollectionReport",
    "MobileWebUiStore",
    "ReleaseConflictError",
    "ReleaseSelectionChangedError",
    "ReleaseView",
    "RollbackUnavailableError",
    "StoredBlob",
    "TargetResourceNotFoundError",
    "UnknownReleaseError",
]
