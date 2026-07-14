from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
        if len(content_type) > 255 or _CONTENT_TYPE_PATTERN.fullmatch(content_type) is None:
            raise AttachmentRequestError("content_type 必须是合法 MIME type")
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
