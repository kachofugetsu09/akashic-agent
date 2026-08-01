from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import secrets
import stat
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field

from infra.channels.base import AttachmentStore
from infra.mobile_realtime.protocol import FrameId
from infra.mobile_realtime.storage import (
    AttachmentRecord,
    AttachmentStateError,
    MobileRealtimeStorage,
)


_HEADER_LENGTH = struct.Struct(">I")
_MAX_HEADER_BYTES = 1024
MAX_ATTACHMENT_CHUNK_BYTES = 128 * 1024
ATTACHMENT_PROGRESS_INTERVAL_BYTES = 1024 * 1024
_CONTENT_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$"
)
_ATTACHMENT_LOCK_STRIPES = 64


class AttachmentRequestError(ValueError):
    pass


class AttachmentChunkHeader(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    attachment_id: FrameId
    offset: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class AttachmentChunk:
    attachment_id: str
    offset: int
    data: bytes


@dataclass(frozen=True, slots=True)
class OutboundAttachmentChunk:
    descriptor: dict[str, object]
    offset: int
    data: bytes
    eof: bool


class AttachmentTransferService:
    """持久化上传分片，并在完成时校验文件摘要。"""

    def __init__(
        self,
        storage: MobileRealtimeStorage,
        attachment_store: AttachmentStore,
        *,
        max_attachment_bytes: int,
    ) -> None:
        if max_attachment_bytes <= 0:
            raise ValueError("max_attachment_bytes 必须大于 0")
        self._storage = storage
        self._attachment_store = attachment_store
        self._max_attachment_bytes = max_attachment_bytes
        self._io_locks = tuple(
            threading.Lock() for _ in range(_ATTACHMENT_LOCK_STRIPES)
        )

    def begin_upload(
        self,
        *,
        device_id: str,
        attachment_id: str,
        session_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
    ) -> AttachmentRecord:
        """创建上传记录，或返回相同附件的服务端确认 offset。"""

        # 1. 在附件边界校验声明元数据
        if not 1 <= size_bytes <= self._max_attachment_bytes:
            raise AttachmentRequestError(
                f"附件大小必须在 1..{self._max_attachment_bytes} 字节"
            )
        safe_filename = _validate_filename(filename)
        _ = _validate_content_type(content_type)
        digest = sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise AttachmentRequestError("sha256 必须是 64 位十六进制摘要")

        # 2. 同一 attachment_id 的恢复、建档和文件校准必须串行
        with self._lock_for(attachment_id):
            return self._begin_upload_locked(
                device_id=device_id,
                attachment_id=attachment_id,
                session_id=session_id,
                filename=safe_filename,
                content_type=content_type,
                size_bytes=size_bytes,
                sha256=digest,
            )

    def append_chunk(
        self,
        *,
        device_id: str,
        chunk: AttachmentChunk,
    ) -> tuple[AttachmentRecord, bool]:
        """按严格 offset 追加一个分片，并返回是否需要确认进度。"""

        if not chunk.data or len(chunk.data) > MAX_ATTACHMENT_CHUNK_BYTES:
            raise AttachmentRequestError(
                f"附件分片必须在 1..{MAX_ATTACHMENT_CHUNK_BYTES} 字节"
            )
        with self._lock_for(chunk.attachment_id):
            record = self._storage.require_upload_attachment(
                device_id=device_id,
                attachment_id=chunk.attachment_id,
            )
            if chunk.offset != record.transferred_bytes:
                raise AttachmentStateError(
                    "附件 offset 不连续: "
                    f"expected={record.transferred_bytes} actual={chunk.offset}"
                )
            next_offset = chunk.offset + len(chunk.data)
            if next_offset > record.size_bytes:
                raise AttachmentStateError("附件分片超过声明大小")

            # 1. 文件可能先写成功但数据库未推进；以已提交 offset 截断恢复
            path = Path(record.local_path)
            actual_size = path.stat().st_size
            if actual_size < record.transferred_bytes:
                raise RuntimeError(
                    "附件文件短于已提交 offset: "
                    f"{actual_size} < {record.transferred_bytes}"
                )
            with path.open("r+b") as stream:
                _ = stream.truncate(record.transferred_bytes)
                _ = stream.seek(record.transferred_bytes)
                written = stream.write(chunk.data)
                if written != len(chunk.data):
                    raise OSError(f"附件分片写入不完整: {written}/{len(chunk.data)}")
                stream.flush()
                os.fsync(stream.fileno())

            # 2. 文件持久化后再以 compare-and-set 提交 offset
            updated = self._storage.advance_attachment(
                device_id=device_id,
                attachment_id=record.attachment_id,
                expected_offset=record.transferred_bytes,
                next_offset=next_offset,
                updated_at=_utc_now(),
            )
        crossed_interval = (
            record.transferred_bytes // ATTACHMENT_PROGRESS_INTERVAL_BYTES
            < updated.transferred_bytes // ATTACHMENT_PROGRESS_INTERVAL_BYTES
        )
        return updated, crossed_interval or updated.transferred_bytes == updated.size_bytes

    def finish_upload(
        self,
        *,
        device_id: str,
        session_id: str,
        attachment_id: str,
    ) -> AttachmentRecord:
        """验证完整文件大小和摘要，并把上传推进为 ready。"""

        with self._lock_for(attachment_id):
            record = self._storage.require_owned_upload(
                device_id=device_id,
                session_id=session_id,
                attachment_id=attachment_id,
            )
            if record.state == "ready":
                return record
            if record.state != "transferring":
                raise AttachmentStateError(f"附件不处于传输状态: {attachment_id}")
            if record.transferred_bytes != record.size_bytes:
                raise AttachmentStateError(
                    f"附件尚未上传完成: {record.transferred_bytes}/{record.size_bytes}"
                )

            # 1. 校验文件实际大小和完整摘要
            path = Path(record.local_path)
            if path.stat().st_size != record.size_bytes:
                raise AttachmentStateError("附件落盘大小与声明不一致")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != record.sha256:
                _ = self._storage.fail_attachment_upload(
                    device_id=device_id,
                    attachment_id=attachment_id,
                    updated_at=_utc_now(),
                )
                raise AttachmentStateError("附件 SHA-256 校验失败")

            # 2. 摘要成立后原子推进 ready
            return self._storage.mark_attachment_ready(
                device_id=device_id,
                attachment_id=attachment_id,
                updated_at=_utc_now(),
            )

    def resolve_uploads(
        self,
        *,
        device_id: str,
        session_id: str,
        attachment_ids: list[str],
    ) -> list[str]:
        """把已就绪 media_refs 解析为仅供 Agent 使用的本地路径。"""

        records = [
            self._storage.require_ready_upload(
                device_id=device_id,
                session_id=session_id,
                attachment_id=attachment_id,
            )
            for attachment_id in attachment_ids
        ]
        total_bytes = sum(record.size_bytes for record in records)
        if total_bytes > self._max_attachment_bytes:
            raise AttachmentRequestError(
                f"单条消息附件总量不能超过 {self._max_attachment_bytes} 字节"
            )
        return [record.local_path for record in records]

    def register_outbound(
        self,
        *,
        session_id: str,
        local_media_path: str | Path,
    ) -> AttachmentRecord:
        """复制本地媒体为不可变服务端附件，并稳定注册内容身份。"""

        return self.register_outbound_batch(
            session_id=session_id,
            local_media_paths=(local_media_path,),
        )[0]

    def register_outbound_batch(
        self,
        *,
        session_id: str,
        local_media_paths: tuple[str | Path, ...] | list[str | Path],
        metadata_overrides: tuple[tuple[str, str] | None, ...] | None = None,
        message_id: str | None = None,
    ) -> tuple[AttachmentRecord, ...]:
        """先完整快照一批媒体，再以单个事务注册全部附件。"""

        candidates = self.snapshot_outbound_batch(
            session_id=session_id,
            local_media_paths=local_media_paths,
            metadata_overrides=metadata_overrides,
        )
        try:
            return self._storage.create_or_read_outbound_attachments(
                candidates,
                message_id=message_id,
            )
        except BaseException:
            self.cleanup_outbound_candidates(candidates)
            raise

    def snapshot_outbound_batch(
        self,
        *,
        session_id: str,
        local_media_paths: tuple[str | Path, ...] | list[str | Path],
        metadata_overrides: tuple[tuple[str, str] | None, ...] | None = None,
    ) -> tuple[AttachmentRecord, ...]:
        """完整快照并校验一批尚未提交的 outbound 候选。"""

        sources = tuple(Path(value) for value in local_media_paths)
        if not 1 <= len(sources) <= 10:
            raise AttachmentRequestError("单条消息附件数量必须在 1..10")
        overrides = metadata_overrides or (None,) * len(sources)
        if len(overrides) != len(sources):
            raise ValueError("出站附件元数据数量必须与路径数量一致")

        # 1. 全批复制和校验完成前不写数据库
        candidates: list[AttachmentRecord] = []
        total_bytes = 0
        try:
            for source, metadata in zip(sources, overrides, strict=True):
                candidate = self._snapshot_outbound_candidate(
                    session_id,
                    source,
                    metadata,
                )
                candidates.append(candidate)
                total_bytes += candidate.size_bytes
                if total_bytes > self._max_attachment_bytes:
                    raise AttachmentRequestError(
                        "单条消息附件总量不能超过 "
                        f"{self._max_attachment_bytes} 字节"
                    )

            return tuple(candidates)
        except BaseException:
            _remove_paths([Path(record.local_path) for record in candidates])
            raise

    def cleanup_outbound_candidates(
        self,
        candidates: tuple[AttachmentRecord, ...],
    ) -> None:
        """删除未提交候选；调用方必须保证事务尚未成功。"""

        _remove_paths([Path(record.local_path) for record in candidates])

    def read_message_outbound(
        self,
        *,
        session_id: str,
        message_id: str,
    ) -> tuple[AttachmentRecord, ...]:
        """读取历史消息已绑定的稳定附件，不再访问原始媒体。"""

        return self._storage.read_message_outbound_attachments(
            message_id=message_id,
            session_id=session_id,
        )

    def _snapshot_outbound_candidate(
        self,
        session_id: str,
        source: Path,
        metadata_override: tuple[str, str] | None = None,
    ) -> AttachmentRecord:
        """创建一个已校验的 outbound canonical 候选记录。"""

        if metadata_override is None:
            filename, content_type = self._outbound_metadata(session_id, source)
        else:
            filename = _validate_filename(metadata_override[0])
            content_type = _validate_content_type(metadata_override[1])
        suffix = Path(filename).suffix
        if not suffix or len(suffix) > 16:
            suffix = ".bin"
        canonical_path = self._attachment_store.create_persistent_path(
            "mobile_outbound_",
            suffix,
        )
        try:
            size_bytes, digest = self._copy_outbound_snapshot(
                source=source,
                destination=canonical_path,
            )
        except BaseException:
            canonical_path.unlink(missing_ok=True)
            raise
        now = _utc_now()
        return AttachmentRecord(
            attachment_id=_new_outbound_attachment_id(),
            device_id=None,
            session_id=session_id,
            direction="outbound",
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=digest,
            local_path=str(canonical_path),
            transferred_bytes=size_bytes,
            state="ready",
            created_at=now,
            updated_at=now,
        )

    def _outbound_metadata(
        self,
        session_id: str,
        source: Path,
    ) -> tuple[str, str]:
        """优先沿用同会话 ready upload 的用户可见元数据。"""

        if source.is_symlink():
            raise AttachmentRequestError("outbound 媒体不能是符号链接")
        upload = self._storage.read_ready_upload_by_local_path(
            session_id=session_id,
            local_path=str(source),
        )
        if upload is not None:
            return (
                _validate_filename(upload.filename),
                _validate_content_type(upload.content_type),
            )
        filename = _validate_filename(source.name)
        guessed_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, _validate_content_type(guessed_type)

    def read_outbound_chunk(
        self,
        *,
        session_id: str,
        attachment_id: str,
        offset: int,
    ) -> OutboundAttachmentChunk:
        """读取指定会话 outbound 附件的单个固定上限分片。"""

        if offset < 0:
            raise AttachmentRequestError("附件下载 offset 不能为负数")
        record = self._storage.require_ready_outbound(
            session_id=session_id,
            attachment_id=attachment_id,
        )
        if offset >= record.size_bytes:
            raise AttachmentRequestError("附件下载 offset 必须小于文件大小")

        # 1. 用拒绝符号链接的同一文件描述符校验并读取 canonical 副本
        data = _read_canonical_chunk(record, offset)
        next_offset = offset + len(data)
        return OutboundAttachmentChunk(
            descriptor=attachment_descriptor(record),
            offset=offset,
            data=data,
            eof=next_offset == record.size_bytes,
        )

    def _copy_outbound_snapshot(
        self,
        *,
        source: Path,
        destination: Path,
    ) -> tuple[int, str]:
        """复制一个普通文件，并返回已落盘快照的大小和摘要。"""

        source_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        destination_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        source_descriptor = os.open(source, source_flags)
        try:
            source_stat = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise AttachmentRequestError("outbound 媒体必须是普通文件")
            if not 1 <= source_stat.st_size <= self._max_attachment_bytes:
                raise AttachmentRequestError(
                    f"附件大小必须在 1..{self._max_attachment_bytes} 字节"
                )
            destination_descriptor = os.open(
                destination,
                destination_flags,
                0o600,
            )
            try:
                destination_stat = os.fstat(destination_descriptor)
                if not stat.S_ISREG(destination_stat.st_mode):
                    raise AttachmentStateError("outbound canonical 必须是普通文件")
                with os.fdopen(source_descriptor, "rb", closefd=False) as reader, os.fdopen(
                    destination_descriptor,
                    "wb",
                    closefd=False,
                ) as writer:
                    size_bytes, digest = _copy_limited(
                        reader,
                        writer,
                        self._max_attachment_bytes,
                    )
                    writer.flush()
                    os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
        finally:
            os.close(source_descriptor)
        if size_bytes != source_stat.st_size:
            raise AttachmentStateError("outbound 源文件在复制期间发生变化")
        destination.chmod(0o444)
        return size_bytes, digest

    def _lock_for(self, attachment_id: str) -> threading.Lock:
        index = int.from_bytes(
            hashlib.blake2s(attachment_id.encode("utf-8"), digest_size=2).digest(),
            "big",
        ) % len(self._io_locks)
        return self._io_locks[index]

    def _begin_upload_locked(
        self,
        *,
        device_id: str,
        attachment_id: str,
        session_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
    ) -> AttachmentRecord:
        """在附件条带锁内恢复已有上传，或创建新的上传记录。"""

        # 1. 已有记录只接受完全相同的上传声明
        existing = self._storage.read_attachment(attachment_id)
        if existing is not None:
            expected = (device_id, session_id, filename, content_type, size_bytes, sha256)
            actual = (
                existing.device_id,
                existing.session_id,
                existing.filename,
                existing.content_type,
                existing.size_bytes,
                existing.sha256,
            )
            if existing.direction != "upload" or actual != expected:
                raise AttachmentStateError("attachment_id 已绑定其他附件")
            return self._reconcile_existing_upload(existing)

        # 2. 新上传先创建内部文件，再原子写入元数据
        suffix = Path(filename).suffix
        if not suffix or len(suffix) > 16:
            suffix = ".bin"
        path = self._attachment_store.create_path("mobile_", suffix)
        path.touch(exist_ok=False)
        try:
            return self._storage.create_attachment(
                AttachmentRecord(
                    attachment_id=attachment_id,
                    device_id=device_id,
                    session_id=session_id,
                    direction="upload",
                    filename=filename,
                    content_type=content_type,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    local_path=str(path),
                    transferred_bytes=0,
                    state="transferring",
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                )
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _reconcile_existing_upload(self, record: AttachmentRecord) -> AttachmentRecord:
        """按数据库 offset 校准文件，并为可恢复失败返回 offset 0。"""

        if record.state == "failed":
            return self._reset_failed_upload(record)
        if record.state == "ready":
            return record
        path = Path(record.local_path)
        actual_size = path.stat().st_size
        if actual_size < record.transferred_bytes:
            if record.device_id is None:
                raise RuntimeError("upload 附件缺少 device_id")
            _ = self._storage.fail_attachment_upload(
                device_id=record.device_id,
                attachment_id=record.attachment_id,
                updated_at=_utc_now(),
            )
            return self._reset_failed_upload(record)
        if actual_size > record.transferred_bytes:
            with path.open("r+b") as stream:
                _ = stream.truncate(record.transferred_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        return record

    def _reset_failed_upload(self, record: AttachmentRecord) -> AttachmentRecord:
        """先清空失败文件，再把持久化 offset 重置到零。"""

        path = Path(record.local_path)
        with path.open("r+b") as stream:
            _ = stream.truncate(0)
            stream.flush()
            os.fsync(stream.fileno())
        if record.device_id is None:
            raise RuntimeError("upload 附件缺少 device_id")
        return self._storage.reset_failed_upload(
            device_id=record.device_id,
            attachment_id=record.attachment_id,
            updated_at=_utc_now(),
        )


def encode_attachment_chunk(chunk: AttachmentChunk) -> bytes:
    header = json.dumps(
        {"attachment_id": chunk.attachment_id, "offset": chunk.offset},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("附件分片 header 过大")
    return _HEADER_LENGTH.pack(len(header)) + header + chunk.data


def decode_attachment_chunk(raw: bytes) -> AttachmentChunk:
    """严格解析带 JSON header 的 WebSocket 二进制附件分片。"""

    if len(raw) < _HEADER_LENGTH.size:
        raise ValueError("附件二进制帧缺少 header 长度")
    header_size = _HEADER_LENGTH.unpack(raw[: _HEADER_LENGTH.size])[0]
    if not 1 <= header_size <= _MAX_HEADER_BYTES:
        raise ValueError("附件二进制帧 header 长度无效")
    payload_offset = _HEADER_LENGTH.size + header_size
    if payload_offset >= len(raw):
        raise ValueError("附件二进制帧缺少分片数据")
    header = AttachmentChunkHeader.model_validate_json(
        raw[_HEADER_LENGTH.size : payload_offset],
        strict=True,
    )
    data = raw[payload_offset:]
    if len(data) > MAX_ATTACHMENT_CHUNK_BYTES:
        raise ValueError("附件二进制分片超过 128 KiB")
    return AttachmentChunk(
        attachment_id=header.attachment_id,
        offset=header.offset,
        data=data,
    )


def attachment_descriptor(record: AttachmentRecord) -> dict[str, object]:
    return {
        "attachment_id": record.attachment_id,
        "filename": record.filename,
        "content_type": record.content_type,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
    }


def _validate_filename(filename: str) -> str:
    safe_filename = filename.strip()
    if (
        not safe_filename
        or safe_filename != filename
        or len(safe_filename) > 255
        or "/" in safe_filename
        or "\\" in safe_filename
        or "\x00" in safe_filename
        or any(ord(char) < 32 or ord(char) == 127 for char in safe_filename)
    ):
        raise AttachmentRequestError("filename 必须是 1..255 字符的纯文件名")
    return safe_filename


def _validate_content_type(content_type: str) -> str:
    if len(content_type) > 255 or _CONTENT_TYPE_PATTERN.fullmatch(content_type) is None:
        raise AttachmentRequestError("content_type 必须是合法 MIME type")
    return content_type


def _new_outbound_attachment_id() -> str:
    """生成带 80 位随机量的 opaque ULID。"""

    value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = alphabet[value & 31]
        value >>= 5
    return "".join(chars)


def _read_canonical_chunk(record: AttachmentRecord, offset: int) -> bytes:
    """以同一文件描述符校验并读取一个 canonical 分片。"""

    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(record.local_path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != record.size_bytes:
            raise RuntimeError(
                f"outbound canonical 文件不符合元数据: {record.attachment_id}"
            )
        _ = os.lseek(descriptor, offset, os.SEEK_SET)
        return os.read(descriptor, MAX_ATTACHMENT_CHUNK_BYTES)
    finally:
        os.close(descriptor)


def _copy_limited(
    reader: BinaryIO,
    writer: BinaryIO,
    max_bytes: int,
) -> tuple[int, str]:
    """复制有限长度的字节流，并返回大小和 SHA-256。"""

    digest = hashlib.sha256()
    size_bytes = 0
    for block in iter(lambda: reader.read(1024 * 1024), b""):
        size_bytes += len(block)
        if size_bytes > max_bytes:
            raise AttachmentRequestError(f"附件大小不能超过 {max_bytes} 字节")
        digest.update(block)
        written = writer.write(block)
        if written != len(block):
            raise OSError(f"outbound 附件写入不完整: {written}/{len(block)}")
    return size_bytes, digest.hexdigest()


def _remove_paths(paths: list[Path]) -> None:
    """尽量删除整批临时文件，并在清理不完整时明确失败。"""

    first_error: OSError | None = None
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
