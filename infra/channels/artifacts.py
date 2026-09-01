from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from agent.media import detect_supported_image_mime
from agent.plugin_composition.channels import AttachmentKind, AttachmentRef
from session.store import AttachmentArtifactRecord, SessionStore


T = TypeVar("T")


@dataclass(frozen=True)
class _SourceFingerprint:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    sha256: str
    signature_head: bytes


@dataclass(frozen=True)
class AttachmentFilesystemIntegrityReport:
    """汇总 artifact filesystem 与 SessionDB authority 的只读证据。"""

    ready_count: int
    verified_bytes: int
    orphan_artifact_ids: tuple[str, ...]
    incomplete_import_ids: tuple[str, ...]


async def _complete_critical(awaitable: Awaitable[T]) -> tuple[T, bool]:
    """完成不可中断的文件发布操作，并报告调用方是否取消。"""

    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


def _verified_attachment_type(
    declared_kind: AttachmentKind,
    declared_media_type: str | None,
    signature_head: bytes,
) -> tuple[AttachmentKind, str | None]:
    """用受支持的文件签名拥有图片 kind，不信任文件名或 MIME。"""

    detected = detect_supported_image_mime(signature_head)
    if detected is not None:
        return AttachmentKind.IMAGE, detected
    if declared_kind is AttachmentKind.IMAGE:
        return AttachmentKind.FILE, declared_media_type
    return declared_kind, declared_media_type


class _ArtifactReadLease:
    """持有一个已核验 artifact 的只读文件描述符。"""

    def __init__(self, ref: AttachmentRef, fd: int) -> None:
        self._ref = ref
        self._fd = fd
        self._lock = asyncio.Lock()

    @property
    def ref(self) -> AttachmentRef:
        return self._ref

    @property
    def model_path(self) -> str:
        """Expose a process-local path while this exact read lease remains open."""

        if self._fd < 0:
            raise RuntimeError("AttachmentReadLease 已关闭")
        return f"/proc/self/fd/{self._fd}"

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes 必须是 int")
        if max_bytes < 0:
            raise ValueError("max_bytes 不能是负数")
        if self._ref.size_bytes > max_bytes:
            raise ValueError(
                f"attachment 超过读取上限: {self._ref.size_bytes} > {max_bytes}"
            )
        async with self._lock:
            if self._fd < 0:
                raise RuntimeError("AttachmentReadLease 已关闭")
            return await asyncio.to_thread(os.pread, self._fd, self._ref.size_bytes, 0)

    async def aclose(self) -> None:
        async with self._lock:
            if self._fd < 0:
                return
            fd = self._fd
            self._fd = -1
            os.close(fd)


class ChannelAttachmentArtifactStore:
    """原子发布并核验 Core-owned channel attachment artifacts。"""

    def __init__(
        self,
        *,
        workspace: Path,
        session_store: SessionStore,
        max_import_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        if isinstance(max_import_bytes, bool) or not isinstance(max_import_bytes, int):
            raise TypeError("max_import_bytes 必须是 int")
        if max_import_bytes < 1:
            raise ValueError("max_import_bytes 必须是正整数")
        self._workspace = workspace
        self._session_store = session_store
        self._max_import_bytes = max_import_bytes
        self._publish_locks = tuple(threading.Lock() for _ in range(64))

    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        """原子发布 bytes，并在 SQLite ready 后返回 opaque ref。"""

        if not isinstance(data, bytes):
            raise TypeError("attachment data 必须是 bytes")
        if len(data) > self._max_import_bytes:
            raise ValueError(
                f"attachment 超过导入上限: {len(data)} > {self._max_import_bytes}"
            )
        kind, media_type = _verified_attachment_type(
            kind,
            media_type,
            data[:12],
        )
        candidate = AttachmentRef(
            artifact_id=uuid4().hex,
            kind=kind,
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

        result, cancelled = await _complete_critical(
            asyncio.to_thread(self._publish_bytes, data, candidate)
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def adopt_file(
        self,
        source: Path,
        *,
        allowed_root: Path,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        """从显式 owner root 复制 regular file，来源变化时拒绝发布。"""

        result, cancelled = await _complete_critical(
            asyncio.to_thread(
                self._adopt_file,
                source,
                allowed_root,
                kind,
                filename,
                media_type,
                None,
                None,
            )
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def adopt_file_with_artifact_id(
        self,
        source: Path,
        *,
        allowed_root: Path,
        expected_ref: AttachmentRef,
    ) -> AttachmentRef:
        """仅在 finalized file 仍等于 durable expected ref 时发布。"""

        if not isinstance(expected_ref, AttachmentRef):
            raise TypeError("expected_ref 必须是 AttachmentRef")

        result, cancelled = await _complete_critical(
            asyncio.to_thread(
                self._adopt_file,
                source,
                allowed_root,
                expected_ref.kind,
                expected_ref.filename,
                expected_ref.media_type,
                expected_ref.artifact_id,
                expected_ref,
            )
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def inspect_file_with_artifact_id(
        self,
        source: Path,
        *,
        allowed_root: Path,
        artifact_id: str,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        """在 durable handoff 前冻结将要发布的 exact ref，不写入状态。"""

        result, cancelled = await _complete_critical(
            asyncio.to_thread(
                self._inspect_file_ref,
                source,
                allowed_root,
                artifact_id,
                kind,
                filename,
                media_type,
            )
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def acquire(self, ref: AttachmentRef) -> _ArtifactReadLease:
        """打开并完整核验一个 ready artifact 的 exact read lease。"""

        if not isinstance(ref, AttachmentRef):
            raise TypeError("ref 必须是 AttachmentRef")
        record = self._session_store.get_attachment(ref.artifact_id)
        if record is None or record.state != "ready":
            raise ValueError(f"attachment 尚未发布: {ref.artifact_id}")
        canonical = self._ref_from_record(record)
        if canonical != ref:
            raise ValueError(f"attachment ref 与权威 metadata 不一致: {ref.artifact_id}")
        fd, cancelled = await _complete_critical(
            asyncio.to_thread(self._open_verified, record)
        )
        if cancelled:
            try:
                os.close(fd)
            except OSError as close_error:
                raise BaseExceptionGroup(
                    "attachment acquire 取消与 fd 清理同时发生",
                    [asyncio.CancelledError(), close_error],
                ) from close_error
            raise asyncio.CancelledError
        return _ArtifactReadLease(canonical, fd)

    def resolve_refs(self, artifact_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]:
        """Resolve ordered opaque identities to immutable ready metadata."""

        refs: list[AttachmentRef] = []
        for artifact_id in artifact_ids:
            record = self._session_store.get_attachment(artifact_id)
            if record is None or record.state != "ready":
                raise ValueError(f"attachment 尚未发布: {artifact_id}")
            refs.append(self._ref_from_record(record))
        return tuple(refs)

    def audit_orphan_artifact_ids(self) -> tuple[str, ...]:
        """报告物理已发布但没有 ready row 的 artifact，不执行删除或恢复。"""

        root = self._resolve_artifact_root(create=False)
        if root is None:
            return ()
        orphans: list[str] = []
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.name.startswith(".") or path.suffix != ".bin":
                continue
            artifact_id = path.stem
            if self._session_store.get_attachment(artifact_id) is None:
                orphans.append(artifact_id)
        return tuple(orphans)

    async def validate_filesystem_integrity(
        self,
    ) -> AttachmentFilesystemIntegrityReport:
        """逐个 nofollow/read/hash 所有 ready artifact，并报告非终态 owner。"""

        database_report = self._session_store.validate_attachment_metadata_integrity()
        records = self._session_store.list_attachments()
        verified_bytes = 0
        for record in records:
            fd, cancelled = await _complete_critical(
                asyncio.to_thread(self._open_verified, record)
            )
            try:
                verified_bytes += record.size_bytes
            finally:
                os.close(fd)
            if cancelled:
                raise asyncio.CancelledError
        root = self._resolve_artifact_root(create=False)
        if root is not None:
            staging = tuple(
                path.name
                for path in root.iterdir()
                if path.name.startswith(".") and path.suffix == ".part"
            )
            if staging:
                raise ValueError(
                    "attachment staging 未收束: " + ", ".join(sorted(staging))
                )
        return AttachmentFilesystemIntegrityReport(
            ready_count=len(records),
            verified_bytes=verified_bytes,
            orphan_artifact_ids=self.audit_orphan_artifact_ids(),
            incomplete_import_ids=database_report.incomplete_import_ids,
        )

    def _publish_bytes(self, data: bytes, ref: AttachmentRef) -> AttachmentRef:
        """在一个同步临界段发布 immutable bytes。"""

        return self._publish_content(
            ref,
            lambda fd: self._write_all(fd, data),
        )

    def _adopt_file(
        self,
        source: Path,
        allowed_root: Path,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
        artifact_id: str | None,
        expected_ref: AttachmentRef | None,
    ) -> AttachmentRef:
        """两次核对 source identity，并把内容复制进 Core artifact root。"""

        source_path, fingerprint = self._inspect_source(source, allowed_root)
        ref = self._ref_from_fingerprint(
            fingerprint,
            artifact_id=uuid4().hex if artifact_id is None else artifact_id,
            kind=kind,
            filename=filename,
            media_type=media_type,
        )
        if expected_ref is not None and ref != expected_ref:
            raise ValueError(
                f"attachment source 与 durable ref 不一致: {expected_ref.artifact_id}"
            )

        def copy_source(target_fd: int) -> None:
            source_fd = os.open(
                source_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                before = os.fstat(source_fd)
                self._require_same_source(before, fingerprint)
                digest = hashlib.sha256()
                copied = 0
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk:
                        break
                    self._write_all(target_fd, chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                after = os.fstat(source_fd)
                self._require_same_source(after, fingerprint)
                if copied != fingerprint.size_bytes or digest.hexdigest() != fingerprint.sha256:
                    raise ValueError(f"attachment source 在复制期间变化: {source}")
                current_path = os.stat(source_path, follow_symlinks=False)
                self._require_same_source(current_path, fingerprint)
            finally:
                os.close(source_fd)

        return self._publish_content(ref, copy_source)

    def _inspect_file_ref(
        self,
        source: Path,
        allowed_root: Path,
        artifact_id: str,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        """核对 Mobile-owned 文件并构造发布时应保持不变的 ref。"""

        _source_path, fingerprint = self._inspect_source(source, allowed_root)
        return self._ref_from_fingerprint(
            fingerprint,
            artifact_id=artifact_id,
            kind=kind,
            filename=filename,
            media_type=media_type,
        )

    @staticmethod
    def _ref_from_fingerprint(
        fingerprint: _SourceFingerprint,
        *,
        artifact_id: str,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        kind, media_type = _verified_attachment_type(
            kind,
            media_type,
            fingerprint.signature_head,
        )
        return AttachmentRef(
            artifact_id=artifact_id,
            kind=kind,
            filename=filename,
            media_type=media_type,
            size_bytes=fingerprint.size_bytes,
            sha256=fingerprint.sha256,
        )

    def _publish_content(
        self,
        ref: AttachmentRef,
        write_content: Callable[[int], None],
    ) -> AttachmentRef:
        """在一个同步临界段完成 file publish 与 ready-row publication。"""

        lock = self._publish_locks[hash(ref.artifact_id) % len(self._publish_locks)]
        with lock:
            return self._publish_content_locked(ref, write_content)

    def _publish_content_locked(
        self,
        ref: AttachmentRef,
        write_content: Callable[[int], None],
    ) -> AttachmentRef:
        """串行恢复或发布同一 artifact identity。"""

        # 1. staging 只在 Core artifact root 内创建。
        root = self._resolve_artifact_root(create=True)
        assert root is not None
        staging = root / f".{ref.artifact_id}.part"
        final = root / f"{ref.artifact_id}.bin"
        storage_key = f"uploads/artifacts/{final.name}"
        created_at = datetime.now(UTC).isoformat()
        intent_started = False
        published = False
        try:
            intent = self._session_store.begin_attachment_import(
                artifact_id=ref.artifact_id,
                storage_key=storage_key,
                expected_size_bytes=ref.size_bytes,
                expected_sha256=ref.sha256,
                created_at=created_at,
            )
            intent_started = True
            if intent.phase == "artifact_committed":
                return self._require_existing_ready(ref)
            if intent.phase == "file_published":
                self._verify_unregistered_file(final, ref)
                return self._register_ready(ref, storage_key)
            if intent.phase != "prepared":
                raise RuntimeError(
                    f"attachment import phase 非法: {ref.artifact_id}:{intent.phase}"
                )
            if final.exists() or final.is_symlink():
                self._verify_unregistered_file(final, ref)
                self._session_store.mark_attachment_import_file_published(
                    ref.artifact_id,
                    updated_at=datetime.now(UTC).isoformat(),
                )
                return self._register_ready(ref, storage_key)
            if staging.exists() or staging.is_symlink():
                if staging.is_symlink() or not staging.is_file():
                    raise ValueError(
                        f"attachment staging 不是安全 regular file: {ref.artifact_id}"
                    )
                staging.unlink()
            fd = os.open(
                staging,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            try:
                write_content(fd)
                os.fsync(fd)
            finally:
                os.close(fd)

            # 2. no-replace publication 与目录 fsync 先让 bytes durable。
            os.link(staging, final, follow_symlinks=False)
            published = True
            staging.unlink()
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._session_store.mark_attachment_import_file_published(
                ref.artifact_id,
                updated_at=datetime.now(UTC).isoformat(),
            )

            # 3. ready row 是唯一可见性 owner；失败时保留可审计 orphan。
            return self._register_ready(ref, storage_key)
        except BaseException as exc:
            failures: list[BaseException] = [exc]
            if intent_started:
                try:
                    self._session_store.record_attachment_import_error(
                        ref.artifact_id,
                        error=f"{type(exc).__name__}: {exc}",
                        updated_at=datetime.now(UTC).isoformat(),
                    )
                except BaseException as record_error:
                    failures.append(record_error)
            if published:
                if len(failures) == 1:
                    raise
                raise BaseExceptionGroup(
                    "attachment publication 与错误记录同时失败",
                    failures,
                ) from exc
            try:
                staging.unlink(missing_ok=True)
            except OSError as cleanup_error:
                failures.append(cleanup_error)
            if len(failures) == 1:
                raise
            raise BaseExceptionGroup(
                "attachment import 与清理同时失败",
                failures,
            ) from exc

    def _register_ready(self, ref: AttachmentRef, storage_key: str) -> AttachmentRef:
        """把已验证的 published bytes 提交为唯一 ready artifact。"""

        record = self._session_store.register_ready_attachment(
            artifact_id=ref.artifact_id,
            storage_key=storage_key,
            kind=ref.kind.value,
            filename=ref.filename,
            media_type=ref.media_type,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
            created_at=datetime.now(UTC).isoformat(),
        )
        return self._ref_from_record(record)

    def _require_existing_ready(self, ref: AttachmentRef) -> AttachmentRef:
        """验证幂等重试命中的 ready row、metadata 与物理 bytes。"""

        record = self._session_store.get_attachment(ref.artifact_id)
        if record is None or record.state != "ready":
            raise RuntimeError(
                f"attachment committed intent 缺少 ready row: {ref.artifact_id}"
            )
        canonical = self._ref_from_record(record)
        if canonical != ref:
            raise RuntimeError(f"attachment ready identity 已漂移: {ref.artifact_id}")
        fd = self._open_verified(record)
        os.close(fd)
        return canonical

    @staticmethod
    def _verify_unregistered_file(path: Path, ref: AttachmentRef) -> None:
        """验证 durable intent 已发布但尚未登记 ready 的 exact bytes。"""

        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != ref.size_bytes:
                raise ValueError(
                    f"attachment published file metadata 已漂移: {ref.artifact_id}"
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != ref.sha256:
                raise ValueError(
                    f"attachment published file hash 已漂移: {ref.artifact_id}"
                )
        finally:
            os.close(fd)

    def _inspect_source(
        self,
        source: Path,
        allowed_root: Path,
    ) -> tuple[Path, _SourceFingerprint]:
        """在复制前核对 source root、regular-file identity、size 与 hash。"""

        if allowed_root.is_symlink() or not allowed_root.is_dir():
            raise ValueError(f"attachment allowed_root 必须是普通目录: {allowed_root}")
        lexical_root = Path(os.path.abspath(allowed_root))
        lexical_source = Path(os.path.abspath(source))
        try:
            relative_source = lexical_source.relative_to(lexical_root)
        except ValueError as exc:
            raise ValueError(f"attachment source 越过 allowed_root: {source}") from exc
        current = lexical_root
        for part in relative_source.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"attachment source 路径不能包含符号链接: {current}")
        root = lexical_root.resolve(strict=True)
        source_parent = lexical_source.parent.resolve(strict=True)
        try:
            source_parent.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"attachment source 越过 allowed_root: {source}") from exc
        source_path = source_parent / lexical_source.name
        fd = os.open(source_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"attachment source 不是 regular file: {source}")
            if file_stat.st_size > self._max_import_bytes:
                raise ValueError(
                    "attachment source 超过导入上限: "
                    f"{file_stat.st_size} > {self._max_import_bytes}"
                )
            digest = hashlib.sha256()
            signature_head = b""
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                if not signature_head:
                    signature_head = chunk[:12]
                digest.update(chunk)
            after = os.fstat(fd)
            if (
                after.st_dev != file_stat.st_dev
                or after.st_ino != file_stat.st_ino
                or after.st_size != file_stat.st_size
                or after.st_mtime_ns != file_stat.st_mtime_ns
            ):
                raise ValueError(f"attachment source 在校验期间变化: {source}")
            return source_path, _SourceFingerprint(
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
                size_bytes=file_stat.st_size,
                mtime_ns=file_stat.st_mtime_ns,
                sha256=digest.hexdigest(),
                signature_head=signature_head,
            )
        finally:
            os.close(fd)

    @staticmethod
    def _require_same_source(
        file_stat: os.stat_result,
        fingerprint: _SourceFingerprint,
    ) -> None:
        if (
            file_stat.st_dev != fingerprint.device
            or file_stat.st_ino != fingerprint.inode
            or file_stat.st_size != fingerprint.size_bytes
            or file_stat.st_mtime_ns != fingerprint.mtime_ns
        ):
            raise ValueError("attachment source identity 已漂移")

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("attachment write 未取得进展")
            view = view[written:]

    def _open_verified(self, record: AttachmentArtifactRecord) -> int:
        """以 nofollow fd 读取并核对不可变 artifact。"""

        root = self._resolve_artifact_root(create=False)
        if root is None:
            raise FileNotFoundError(record.storage_key)
        path = self._workspace.resolve() / Path(record.storage_key)
        resolved_parent = path.parent.resolve(strict=True)
        if resolved_parent != root:
            raise ValueError(f"attachment storage path 越过 artifact root: {record.artifact_id}")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"attachment 不是 regular file: {record.artifact_id}")
            if file_stat.st_size != record.size_bytes:
                raise ValueError(f"attachment size 已漂移: {record.artifact_id}")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != record.sha256:
                raise ValueError(f"attachment hash 已漂移: {record.artifact_id}")
            os.lseek(fd, 0, os.SEEK_SET)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _resolve_artifact_root(self, *, create: bool) -> Path | None:
        """解析 artifact root，并拒绝 workspace 内任一级 symlink。"""

        if self._workspace.is_symlink():
            raise ValueError(f"workspace 不能是符号链接: {self._workspace}")
        if not self._workspace.is_dir():
            raise ValueError(f"workspace 不存在或不是目录: {self._workspace}")
        workspace = self._workspace.resolve(strict=True)
        current = self._workspace
        for name in ("uploads", "artifacts"):
            current = current / name
            if current.is_symlink():
                raise ValueError(f"attachment 目录不能是符号链接: {current}")
            if not current.exists():
                if not create:
                    return None
                current.mkdir(mode=0o700, exist_ok=True)
            if not current.is_dir() or current.is_symlink():
                raise ValueError(f"attachment 目录必须是普通目录: {current}")
        root = current.resolve(strict=True)
        if root != workspace / "uploads" / "artifacts":
            raise ValueError("attachment root 越过 workspace")
        return root

    @staticmethod
    def _ref_from_record(record: AttachmentArtifactRecord) -> AttachmentRef:
        return AttachmentRef(
            artifact_id=record.artifact_id,
            kind=AttachmentKind(record.kind),
            filename=record.filename,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
        )


__all__ = [
    "AttachmentFilesystemIntegrityReport",
    "ChannelAttachmentArtifactStore",
]
