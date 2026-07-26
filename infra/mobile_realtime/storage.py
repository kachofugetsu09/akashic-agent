from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast


PairingStatus = Literal["pending", "confirmed", "consumed", "expired"]
AttachmentDirection = Literal["upload", "outbound"]
AttachmentState = Literal["transferring", "ready", "failed"]
_MAX_REBASE_ACK = 1 << 62


class MobileStorageError(RuntimeError):
    """表示移动端持久化契约被违反。"""


class ServerIdentityConflictError(MobileStorageError):
    """表示数据库中的服务器身份与当前身份冲突。"""


class UnknownPairingError(MobileStorageError):
    """表示配对会话不存在。"""


class PairingStateError(MobileStorageError):
    """表示配对会话状态不允许当前操作。"""


class PairingExpiredError(MobileStorageError):
    """表示配对会话已经过期。"""


class UnknownDeviceError(MobileStorageError):
    """表示设备不存在。"""


class AckRollbackError(MobileStorageError):
    """表示累计 ACK 试图倒退。"""


class AckOverflowError(MobileStorageError):
    """表示累计 ACK 超过设备已发送上限。"""


class SentCursorError(MobileStorageError):
    """表示已发送游标倒退或超过已分配上限。"""


class CommandConflictError(MobileStorageError):
    """表示同一命令 ID 被用于不同请求。"""


class AttachmentStateError(MobileStorageError):
    """表示附件不存在、归属错误或传输状态不允许当前操作。"""


@dataclass(frozen=True, slots=True)
class ServerIdentityReference:
    server_id: str
    keyset_manifest_path: str
    public_key_fingerprint: str


@dataclass(frozen=True, slots=True)
class PairingSessionRecord:
    pairing_id: str
    secret_hash: str | None
    expires_at: datetime
    status: PairingStatus


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    device_id: str
    public_key: str
    display_name: str
    created_at: datetime
    revoked_at: datetime | None
    capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeviceCursor:
    device_id: str
    next_event_seq: int
    sent_event_seq: int
    acknowledged_event_seq: int


@dataclass(frozen=True, slots=True)
class DurableInboxEvent:
    device_id: str
    event_seq: int
    event_id: str
    priority: Literal["P0"]
    envelope_json: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AckAdvance:
    previous_event_seq: int
    acknowledged_event_seq: int
    deleted_events: int


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    device_id: str
    command_id: str
    command_type: str
    request_hash: str
    status: Literal["processing", "completed"]
    reply_type: str | None
    reply_payload_json: str | None
    session_id: str | None
    turn_id: str | None


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    attachment_id: str
    device_id: str | None
    session_id: str
    direction: AttachmentDirection
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    local_path: str
    transferred_bytes: int
    state: AttachmentState
    created_at: datetime
    updated_at: datetime


class MobileRealtimeStorage:
    """持有移动端数据库 schema，并原子维护设备游标与 durable inbox。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._closed = False

        # 1. 建立 SQLite 运行约束
        with self._lock:
            journal_mode = self._db.execute("PRAGMA journal_mode=WAL").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "wal":
                raise RuntimeError("mobile realtime 数据库未能启用 WAL")
            _ = self._db.execute("PRAGMA synchronous=NORMAL")
            _ = self._db.execute("PRAGMA foreign_keys=ON")

            # 2. 创建由本存储层拥有的 schema
            self._init_schema()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def write_server_identity(self, reference: ServerIdentityReference) -> None:
        """首次记录服务器身份，后续只允许同一身份更新 manifest 引用。"""

        # 1. 校验新引用属于完整身份记录
        _ = _require_text(reference.server_id, "server_id")
        _ = _require_text(reference.keyset_manifest_path, "keyset_manifest_path")
        _ = _require_text(reference.public_key_fingerprint, "public_key_fingerprint")

        # 2. 锁定单例身份并拒绝公钥变化
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                """
                SELECT server_id, public_key_fingerprint
                FROM mobile_server_identity
                WHERE singleton = 1
                """
            ).fetchone()
            if row is not None:
                server_id = _row_text(row, "server_id")
                fingerprint = _row_text(row, "public_key_fingerprint")
                if (
                    server_id != reference.server_id
                    or fingerprint != reference.public_key_fingerprint
                ):
                    raise ServerIdentityConflictError(
                        "mobile realtime 服务器身份与数据库记录不一致"
                    )

            # 3. 原子写入当前 keyset manifest 引用
            _ = self._db.execute(
                """
                INSERT INTO mobile_server_identity(
                    singleton, server_id, keyset_manifest_path,
                    public_key_fingerprint
                ) VALUES(1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    keyset_manifest_path = excluded.keyset_manifest_path
                """,
                (
                    reference.server_id,
                    reference.keyset_manifest_path,
                    reference.public_key_fingerprint,
                ),
            )

    def read_server_identity(self) -> ServerIdentityReference | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT server_id, keyset_manifest_path, public_key_fingerprint
                FROM mobile_server_identity
                WHERE singleton = 1
                """
            ).fetchone()
        if row is None:
            return None
        return ServerIdentityReference(
            server_id=_row_text(row, "server_id"),
            keyset_manifest_path=_row_text(row, "keyset_manifest_path"),
            public_key_fingerprint=_row_text(row, "public_key_fingerprint"),
        )

    def create_pairing_session(self, session: PairingSessionRecord) -> None:
        """创建尚未确认的一次性配对会话。"""

        if session.status != "pending":
            raise PairingStateError("新配对会话必须处于 pending 状态")
        secret_hash = _require_text(session.secret_hash, "secret_hash")
        with self._lock, self._db:
            _ = self._db.execute(
                """
                INSERT INTO mobile_pairing_sessions(
                    pairing_id, secret_hash, expires_at, status
                ) VALUES(?, ?, ?, 'pending')
                """,
                (
                    _require_text(session.pairing_id, "pairing_id"),
                    secret_hash,
                    _serialize_datetime(session.expires_at, "expires_at"),
                ),
            )

    def read_pairing_session(self, pairing_id: str) -> PairingSessionRecord | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT pairing_id, secret_hash, expires_at, status
                FROM mobile_pairing_sessions
                WHERE pairing_id = ?
                """,
                (_require_text(pairing_id, "pairing_id"),),
            ).fetchone()
        if row is None:
            return None
        return _pairing_from_row(row)

    def confirm_pairing(self, pairing_id: str, *, now: datetime) -> PairingSessionRecord:
        """确认未过期的 pending 配对会话。"""

        # 1. 在写事务内读取当前状态
        pairing_key = _require_text(pairing_id, "pairing_id")
        current_time = _require_aware_datetime(now, "now")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            row = self._read_pairing_row(pairing_key)
            session = _pairing_from_row(row)
            if session.status != "pending":
                raise PairingStateError(
                    f"配对会话不能确认: {pairing_key} status={session.status}"
                )
            if session.expires_at <= current_time:
                raise PairingExpiredError(f"配对会话已过期: {pairing_key}")

            # 2. 原子推进到电脑已确认状态
            _ = self._db.execute(
                """
                UPDATE mobile_pairing_sessions
                SET status = 'confirmed'
                WHERE pairing_id = ?
                """,
                (pairing_key,),
            )
        return PairingSessionRecord(
            pairing_id=session.pairing_id,
            secret_hash=session.secret_hash,
            expires_at=session.expires_at,
            status="confirmed",
        )

    def consume_pairing(
        self,
        pairing_id: str,
        device: DeviceRecord,
        *,
        now: datetime,
    ) -> DeviceRecord:
        """原子恢复或注册设备，并销毁已确认配对会话的一次性 secret。"""

        # 1. 校验会话仍然处于可消费状态
        pairing_key = _require_text(pairing_id, "pairing_id")
        current_time = _require_aware_datetime(now, "now")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            session = _pairing_from_row(self._read_pairing_row(pairing_key))
            if session.status != "confirmed":
                raise PairingStateError(
                    f"配对会话不能消费: {pairing_key} status={session.status}"
                )
            if session.expires_at <= current_time:
                raise PairingExpiredError(f"配对会话已过期: {pairing_key}")

            # 2. 同一设备公钥沿用原 device_id、cursor 和会话所有权
            row = self._db.execute(
                """
                SELECT device_id, public_key, display_name, created_at,
                       revoked_at, capabilities
                FROM mobile_devices
                WHERE public_key = ?
                """,
                (_require_text(device.public_key, "public_key"),),
            ).fetchone()
            if row is None:
                effective_device = device
                self._insert_device(effective_device)
            else:
                existing = _device_from_row(cast(sqlite3.Row, row))
                if existing.revoked_at is not None:
                    raise PairingStateError("已撤销的设备密钥不能重新配对")
                effective_device = DeviceRecord(
                    device_id=existing.device_id,
                    public_key=existing.public_key,
                    display_name=device.display_name,
                    created_at=existing.created_at,
                    revoked_at=None,
                    capabilities=device.capabilities,
                )
                _ = self._db.execute(
                    """
                    UPDATE mobile_devices
                    SET display_name = ?, capabilities = ?
                    WHERE device_id = ?
                    """,
                    (
                        effective_device.display_name,
                        _serialize_capabilities(effective_device.capabilities),
                        effective_device.device_id,
                    ),
                )

            # 3. 作废一次性 secret 并提交完成状态
            updated = self._db.execute(
                """
                UPDATE mobile_pairing_sessions
                SET secret_hash = NULL, status = 'consumed'
                WHERE pairing_id = ? AND status = 'confirmed'
                """,
                (pairing_key,),
            )
            if updated.rowcount != 1:
                raise PairingStateError(f"配对会话状态并发变化: {pairing_key}")
        return effective_device

    def register_device(self, device: DeviceRecord) -> None:
        """注册设备，并在同一事务建立初始事件游标。"""

        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            self._insert_device(device)

    def read_device(self, device_id: str) -> DeviceRecord | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT device_id, public_key, display_name, created_at,
                       revoked_at, capabilities
                FROM mobile_devices
                WHERE device_id = ?
                """,
                (_require_text(device_id, "device_id"),),
            ).fetchone()
        if row is None:
            return None
        return _device_from_row(row)

    def revoke_device(self, device_id: str, *, revoked_at: datetime) -> DeviceRecord:
        """原子标记设备已撤销，并返回数据库中的最终状态。"""

        # 1. 锁定设备当前撤销状态
        device_key = _require_text(device_id, "device_id")
        timestamp = _serialize_datetime(revoked_at, "revoked_at")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            row = self._read_device_row(device_key)
            existing = _device_from_row(row)

            # 2. 首次撤销时保留准确时间，重复调用保持幂等
            if existing.revoked_at is None:
                _ = self._db.execute(
                    """
                    UPDATE mobile_devices
                    SET revoked_at = ?
                    WHERE device_id = ?
                    """,
                    (timestamp, device_key),
                )
                return DeviceRecord(
                    device_id=existing.device_id,
                    public_key=existing.public_key,
                    display_name=existing.display_name,
                    created_at=existing.created_at,
                    revoked_at=_parse_datetime(timestamp, "revoked_at"),
                    capabilities=existing.capabilities,
                )
        return existing

    def read_cursor(self, device_id: str) -> DeviceCursor:
        with self._lock:
            row = self._db.execute(
                """
                SELECT device_id, next_event_seq, sent_event_seq,
                       acknowledged_event_seq
                FROM mobile_device_cursors
                WHERE device_id = ?
                """,
                (_require_text(device_id, "device_id"),),
            ).fetchone()
        if row is None:
            raise UnknownDeviceError(f"设备不存在或缺少 cursor: {device_id}")
        return _cursor_from_row(row)

    def rebase_cursor_with_durable_event(
        self,
        device_id: str,
        *,
        through_event_seq: int,
        event_id: str,
        envelope_json: str,
        created_at: datetime,
    ) -> DurableInboxEvent:
        """原子重定位回退游标，并写入要求客户端重建的 durable 事件。"""

        # 1. 固化事件字段，并只允许向前跨过服务端当前已分配范围
        device_key = _require_text(device_id, "device_id")
        event_key = _require_text(event_id, "event_id")
        envelope = _require_text(envelope_json, "envelope_json")
        timestamp = _serialize_datetime(created_at, "created_at")
        if not 0 <= through_event_seq <= _MAX_REBASE_ACK:
            raise ValueError("through_event_seq 超出可恢复的 SQLite 序号范围")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                """
                SELECT device_id, next_event_seq, sent_event_seq,
                       acknowledged_event_seq
                FROM mobile_device_cursors
                WHERE device_id = ?
                """,
                (device_key,),
            ).fetchone()
            if row is None:
                raise UnknownDeviceError(f"设备不存在或缺少 cursor: {device_key}")
            cursor = _cursor_from_row(row)
            if through_event_seq < cursor.next_event_seq:
                raise ValueError("客户端 ACK 没有领先服务端 cursor")

            # 2. 丢弃回退窗口，并在同一事务写入紧接客户端 ACK 的 reset
            event_seq = through_event_seq + 1
            _ = self._db.execute(
                "DELETE FROM mobile_device_inbox WHERE device_id = ?",
                (device_key,),
            )
            _ = self._db.execute(
                """
                INSERT INTO mobile_device_inbox(
                    device_id, event_seq, event_id, priority,
                    envelope_json, created_at
                ) VALUES(?, ?, ?, 'P0', ?, ?)
                """,
                (device_key, event_seq, event_key, envelope, timestamp),
            )
            _ = self._db.execute(
                """
                UPDATE mobile_device_cursors
                SET next_event_seq = ?, sent_event_seq = ?, acknowledged_event_seq = ?
                WHERE device_id = ?
                """,
                (event_seq + 1, through_event_seq, through_event_seq, device_key),
            )

        return DurableInboxEvent(
            device_id=device_key,
            event_seq=event_seq,
            event_id=event_key,
            priority="P0",
            envelope_json=envelope,
            created_at=_parse_datetime(timestamp, "created_at"),
        )

    def append_durable_event(
        self,
        *,
        device_id: str,
        event_id: str,
        envelope_json: str,
        created_at: datetime,
    ) -> DurableInboxEvent:
        """在同一写事务分配 event_seq 并插入 P0 durable event。"""

        return self.append_durable_events(
            device_ids=(device_id,),
            event_id=event_id,
            envelope_json=envelope_json,
            created_at=created_at,
        )[0]

    def append_durable_events(
        self,
        *,
        device_ids: tuple[str, ...],
        event_id: str,
        envelope_json: str,
        created_at: datetime,
    ) -> tuple[DurableInboxEvent, ...]:
        """在一个写事务中为多个设备分配序号并插入同一 P0 事件。"""

        # 1. 固化内部已验证的广播目标和事件字段
        device_keys = tuple(
            _require_text(device_id, "device_id") for device_id in device_ids
        )
        if len(set(device_keys)) != len(device_keys):
            raise ValueError("durable event 广播设备不能重复")
        event_key = _require_text(event_id, "event_id")
        envelope = _require_text(envelope_json, "envelope_json")
        timestamp = _serialize_datetime(created_at, "created_at")
        if not device_keys:
            return ()

        # 2. 锁定全部 cursor，并为每个设备分配自己的严格递增序号
        allocated: list[tuple[str, int]] = []
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            for device_key in device_keys:
                row = self._db.execute(
                    """
                    SELECT next_event_seq
                    FROM mobile_device_cursors
                    WHERE device_id = ?
                    """,
                    (device_key,),
                ).fetchone()
                if row is None:
                    raise UnknownDeviceError(f"设备不存在或缺少 cursor: {device_key}")
                allocated.append((device_key, _row_positive_int(row, "next_event_seq")))

            # 3. 同事务写入全部 durable event 后推进各设备 cursor
            for device_key, event_seq in allocated:
                _ = self._db.execute(
                    """
                    INSERT INTO mobile_device_inbox(
                        device_id, event_seq, event_id, priority,
                        envelope_json, created_at
                    ) VALUES(?, ?, ?, 'P0', ?, ?)
                    """,
                    (device_key, event_seq, event_key, envelope, timestamp),
                )
                updated = self._db.execute(
                    """
                    UPDATE mobile_device_cursors
                    SET next_event_seq = ?
                    WHERE device_id = ? AND next_event_seq = ?
                    """,
                    (event_seq + 1, device_key, event_seq),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(f"设备 event_seq 分配发生并发冲突: {device_key}")

        created = _parse_datetime(timestamp, "created_at")
        return tuple(
            DurableInboxEvent(
                device_id=device_key,
                event_seq=event_seq,
                event_id=event_key,
                priority="P0",
                envelope_json=envelope,
                created_at=created,
            )
            for device_key, event_seq in allocated
        )

    def read_durable_events(
        self,
        device_id: str,
        *,
        after_event_seq: int,
        limit: int,
    ) -> tuple[DurableInboxEvent, ...]:
        if after_event_seq < 0:
            raise ValueError("after_event_seq 不能为负数")
        if limit <= 0:
            raise ValueError("limit 必须大于零")
        with self._lock:
            self._require_device_cursor(device_id)
            rows = self._db.execute(
                """
                SELECT device_id, event_seq, event_id, priority,
                       envelope_json, created_at
                FROM mobile_device_inbox
                WHERE device_id = ? AND event_seq > ?
                ORDER BY event_seq ASC
                LIMIT ?
                """,
                (device_id, after_event_seq, limit),
            ).fetchall()
        return tuple(_inbox_event_from_row(row) for row in rows)

    def durable_event_range_is_contiguous(
        self,
        device_id: str,
        *,
        after_event_seq: int,
        through_event_seq: int,
    ) -> bool:
        """确认指定待重放窗口包含每一个已分配序号。"""

        if after_event_seq < 0:
            raise ValueError("after_event_seq 不能为负数")
        if through_event_seq < after_event_seq:
            raise ValueError("through_event_seq 不能小于 after_event_seq")
        expected_count = through_event_seq - after_event_seq
        with self._lock:
            self._require_device_cursor(device_id)
            row = self._db.execute(
                """
                SELECT COUNT(*) AS event_count
                FROM mobile_device_inbox
                WHERE device_id = ?
                  AND event_seq > ?
                  AND event_seq <= ?
                """,
                (device_id, after_event_seq, through_event_seq),
            ).fetchone()
        if row is None:
            raise RuntimeError("mobile_device_inbox COUNT 查询未返回结果行")
        return _row_nonnegative_int(row, "event_count") == expected_count

    def mark_events_sent(self, device_id: str, *, through_event_seq: int) -> DeviceCursor:
        """推进持久化的已发送上限，供累计 ACK 做越界判断。"""

        # 1. 锁定当前 cursor 并验证范围
        device_key = _require_text(device_id, "device_id")
        if through_event_seq < 0:
            raise ValueError("through_event_seq 不能为负数")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            row = self._read_cursor_row(device_key)
            cursor = _cursor_from_row(row)
            if through_event_seq < cursor.sent_event_seq:
                raise SentCursorError(
                    f"已发送 cursor 不能倒退: {through_event_seq} < {cursor.sent_event_seq}"
                )
            allocated_event_seq = cursor.next_event_seq - 1
            if through_event_seq > allocated_event_seq:
                raise SentCursorError(
                    f"已发送 cursor 超过已分配上限: {through_event_seq} > {allocated_event_seq}"
                )

            # 2. 同值视为重试，前进值写入数据库
            if through_event_seq > cursor.sent_event_seq:
                _ = self._db.execute(
                    """
                    UPDATE mobile_device_cursors
                    SET sent_event_seq = ?
                    WHERE device_id = ?
                    """,
                    (through_event_seq, device_key),
                )
        return DeviceCursor(
            device_id=cursor.device_id,
            next_event_seq=cursor.next_event_seq,
            sent_event_seq=through_event_seq,
            acknowledged_event_seq=cursor.acknowledged_event_seq,
        )

    def acknowledge_durable_events(
        self,
        device_id: str,
        *,
        through_event_seq: int,
    ) -> AckAdvance:
        """原子推进累计 ACK，并删除该设备已确认的 P0 事件。"""

        # 1. 锁定 cursor 并拒绝倒退或越界 ACK
        device_key = _require_text(device_id, "device_id")
        if through_event_seq < 0:
            raise ValueError("through_event_seq 不能为负数")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            cursor = _cursor_from_row(self._read_cursor_row(device_key))
            if through_event_seq < cursor.acknowledged_event_seq:
                raise AckRollbackError(
                    "累计 ACK 不能倒退: "
                    f"{through_event_seq} < {cursor.acknowledged_event_seq}"
                )
            if through_event_seq > cursor.sent_event_seq:
                raise AckOverflowError(
                    "累计 ACK 超过已发送上限: "
                    f"{through_event_seq} > {cursor.sent_event_seq}"
                )
            if through_event_seq == cursor.acknowledged_event_seq:
                return AckAdvance(
                    previous_event_seq=cursor.acknowledged_event_seq,
                    acknowledged_event_seq=through_event_seq,
                    deleted_events=0,
                )

            # 2. 在同一事务推进 cursor 并批量删除 durable event
            _ = self._db.execute(
                """
                UPDATE mobile_device_cursors
                SET acknowledged_event_seq = ?
                WHERE device_id = ?
                """,
                (through_event_seq, device_key),
            )
            deleted = self._db.execute(
                """
                DELETE FROM mobile_device_inbox
                WHERE device_id = ? AND event_seq <= ?
                """,
                (device_key, through_event_seq),
            )
        return AckAdvance(
            previous_event_seq=cursor.acknowledged_event_seq,
            acknowledged_event_seq=through_event_seq,
            deleted_events=deleted.rowcount,
        )

    def has_unacked_event_before(self, device_id: str, *, cutoff: datetime) -> bool:
        """检查设备是否存在超过保留期的未确认 P0 事件。"""

        device_key = _require_text(device_id, "device_id")
        cutoff_text = _serialize_datetime(cutoff, "cutoff")
        with self._lock:
            self._require_device_cursor(device_key)
            row = self._db.execute(
                """
                SELECT 1
                FROM mobile_device_inbox
                WHERE device_id = ? AND created_at < ?
                LIMIT 1
                """,
                (device_key, cutoff_text),
            ).fetchone()
        return row is not None

    def count_durable_events(self, device_id: str) -> int:
        device_key = _require_text(device_id, "device_id")
        with self._lock:
            self._require_device_cursor(device_key)
            row = self._db.execute(
                """
                SELECT COUNT(*) AS event_count
                FROM mobile_device_inbox
                WHERE device_id = ?
                """,
                (device_key,),
            ).fetchone()
        if row is None:
            raise RuntimeError("mobile_device_inbox COUNT 查询未返回结果行")
        return _row_nonnegative_int(row, "event_count")

    def list_active_devices(self) -> tuple[DeviceRecord, ...]:
        """返回所有尚未撤销的移动设备。"""

        with self._lock:
            rows = self._db.execute(
                """
                SELECT device_id, public_key, display_name, created_at,
                       revoked_at, capabilities
                FROM mobile_devices
                WHERE revoked_at IS NULL
                ORDER BY created_at ASC, device_id ASC
                """
            ).fetchall()
        return tuple(_device_from_row(row) for row in rows)

    def allocate_connection_epoch(self) -> int:
        """原子分配跨进程重启仍严格递增的连接代际。"""

        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                """
                SELECT last_epoch
                FROM mobile_connection_epoch
                WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                raise RuntimeError("mobile_connection_epoch 单例记录缺失")
            last_epoch = _row_nonnegative_int(row, "last_epoch")
            next_epoch = last_epoch + 1
            _ = self._db.execute(
                """
                UPDATE mobile_connection_epoch
                SET last_epoch = ?
                WHERE singleton = 1
                """,
                (next_epoch,),
            )
        return next_epoch

    def claim_session(
        self,
        *,
        device_id: str,
        session_id: str,
        created_at: datetime,
    ) -> None:
        """记录移动会话的首次创建设备，不限制其他已认证设备使用。"""

        # 1. 设备身份只负责认证，会话归属不再作为访问边界
        device_key = _require_text(device_id, "device_id")
        session_key = _require_text(session_id, "session_id")
        timestamp = _serialize_datetime(created_at, "created_at")
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            _ = self._read_device_row(device_key)
            _ = self._db.execute(
                """
                INSERT INTO mobile_device_sessions(device_id, session_id, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (device_key, session_key, timestamp),
            )

    def has_session_claim(self, session_id: str) -> bool:
        """判断移动会话是否曾在服务端建立过持久化身份。"""

        session_key = _require_text(session_id, "session_id")
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM mobile_device_sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
        return row is not None

    def list_device_sessions(self, device_id: str) -> tuple[str, ...]:
        """按最近绑定顺序列出设备拥有的移动会话。"""

        device_key = _require_text(device_id, "device_id")
        with self._lock:
            _ = self._read_device_row(device_key)
            rows = self._db.execute(
                """
                SELECT session_id
                FROM mobile_device_sessions
                WHERE device_id = ?
                ORDER BY created_at DESC
                """,
                (device_key,),
            ).fetchall()
        return tuple(_row_text(row, "session_id") for row in rows)

    def create_attachment(self, record: AttachmentRecord) -> AttachmentRecord:
        """创建附件传输记录，并保持客户端标识全局唯一。"""

        with self._lock, self._db:
            if record.device_id is not None:
                _ = self._read_device_row(record.device_id)
            _ = self._db.execute(
                """
                INSERT INTO mobile_attachments(
                    attachment_id, device_id, session_id, direction,
                    filename, content_type, size_bytes, sha256, local_path,
                    transferred_bytes, state, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _require_text(record.attachment_id, "attachment_id"),
                    record.device_id,
                    _require_text(record.session_id, "session_id"),
                    record.direction,
                    _require_text(record.filename, "filename"),
                    _require_text(record.content_type, "content_type"),
                    record.size_bytes,
                    _require_text(record.sha256, "sha256"),
                    _require_text(record.local_path, "local_path"),
                    record.transferred_bytes,
                    record.state,
                    _serialize_datetime(record.created_at, "created_at"),
                    _serialize_datetime(record.updated_at, "updated_at"),
                ),
            )
        return record

    def create_or_read_outbound_attachment(
        self,
        record: AttachmentRecord,
    ) -> tuple[AttachmentRecord, bool]:
        """原子创建 outbound 附件，或返回相同内容身份的既有记录。"""

        resolved = self.create_or_read_outbound_attachments((record,))[0]
        return resolved, resolved.local_path == record.local_path

    def create_or_read_outbound_attachments(
        self,
        records: tuple[AttachmentRecord, ...],
        *,
        message_id: str | None = None,
    ) -> tuple[AttachmentRecord, ...]:
        """在单个事务中创建附件，并可绑定稳定的历史消息槽位。"""

        if not records:
            raise ValueError("outbound 附件批次不能为空")
        for record in records:
            self._validate_outbound_candidate(record)

        # 1. 按内容身份串行去重，任何异常都会回滚本批数据库写入
        resolved_by_identity: dict[
            tuple[str, str, str, str, int], AttachmentRecord
        ] = {}
        result: list[AttachmentRecord] = []
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            if message_id is not None:
                bound = self._read_message_outbound_attachments(
                    message_id=message_id,
                    session_id=records[0].session_id,
                )
                if bound:
                    if len(bound) != len(records):
                        raise AttachmentStateError("历史消息附件槽位数量发生变化")
                    for candidate in records:
                        Path(candidate.local_path).unlink()
                    return bound
            for candidate in records:
                identity = (
                    candidate.session_id,
                    candidate.filename,
                    candidate.content_type,
                    candidate.sha256,
                    candidate.size_bytes,
                )
                resolved = resolved_by_identity.get(identity)
                if resolved is None:
                    resolved = self._read_outbound_for_candidate(candidate)
                    if resolved is None:
                        self._insert_outbound_attachment(candidate)
                        resolved = candidate
                    resolved_by_identity[identity] = resolved
                if resolved.local_path != candidate.local_path:
                    Path(candidate.local_path).unlink()
                result.append(resolved)
            if message_id is not None:
                for ordinal, record in enumerate(result):
                    _ = self._db.execute(
                        """
                        INSERT INTO mobile_message_attachments(
                            message_id, ordinal, attachment_id
                        ) VALUES(?, ?, ?)
                        """,
                        (_require_text(message_id, "message_id"), ordinal, record.attachment_id),
                    )
        return tuple(result)

    def read_message_outbound_attachments(
        self,
        *,
        message_id: str,
        session_id: str,
    ) -> tuple[AttachmentRecord, ...]:
        """按消息槽位顺序读取已物化的 outbound 附件。"""

        with self._lock:
            return self._read_message_outbound_attachments(
                message_id=message_id,
                session_id=session_id,
            )

    def _read_message_outbound_attachments(
        self,
        *,
        message_id: str,
        session_id: str,
    ) -> tuple[AttachmentRecord, ...]:
        rows = self._db.execute(
            """
            SELECT attachment.*
            FROM mobile_message_attachments AS binding
            INNER JOIN mobile_attachments AS attachment
                ON attachment.attachment_id = binding.attachment_id
            WHERE binding.message_id = ? AND attachment.session_id = ?
            ORDER BY binding.ordinal ASC
            """,
            (
                _require_text(message_id, "message_id"),
                _require_text(session_id, "session_id"),
            ),
        ).fetchall()
        records = tuple(_attachment_from_row(row) for row in rows)
        for record in records:
            if record.direction != "outbound" or record.state != "ready":
                raise AttachmentStateError("历史消息绑定了不可下载的附件")
            self._require_outbound_canonical(record)
        return records

    def _validate_outbound_candidate(self, record: AttachmentRecord) -> None:
        if (
            record.direction != "outbound"
            or record.device_id is not None
            or record.state != "ready"
            or record.transferred_bytes != record.size_bytes
        ):
            raise ValueError("outbound 附件必须以无设备归属的 ready 完整记录创建")
        self._require_outbound_canonical(record)

    def _read_outbound_for_candidate(
        self,
        candidate: AttachmentRecord,
    ) -> AttachmentRecord | None:
        """在当前事务中读取并校验候选内容身份。"""

        row = self._db.execute(
            """
            SELECT * FROM mobile_attachments
            WHERE session_id = ? AND filename = ? AND content_type = ? AND sha256 = ?
              AND size_bytes = ? AND direction = 'outbound' AND state = 'ready'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (
                _require_text(candidate.session_id, "session_id"),
                _require_text(candidate.filename, "filename"),
                _require_text(candidate.content_type, "content_type"),
                _require_text(candidate.sha256, "sha256"),
                candidate.size_bytes,
            ),
        ).fetchone()
        if row is None:
            return None
        existing = _attachment_from_row(row)
        if (
            existing.direction != "outbound"
            or existing.device_id is not None
            or existing.state != "ready"
            or existing.transferred_bytes != existing.size_bytes
        ):
            raise AttachmentStateError("outbound 内容身份命中了非法记录")
        self._require_outbound_canonical(existing)
        return existing

    def _require_outbound_canonical(self, record: AttachmentRecord) -> None:
        """以拒绝符号链接的同一文件描述符验证 canonical 文件。"""

        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(record.local_path, flags)
        try:
            file_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size != record.size_bytes
            ):
                raise AttachmentStateError(
                    "既有 outbound canonical 文件不符合元数据"
                )
        finally:
            os.close(descriptor)

    def _insert_outbound_attachment(self, record: AttachmentRecord) -> None:
        """在当前事务中插入完整 outbound 元数据。"""

        _ = self._db.execute(
            """
            INSERT INTO mobile_attachments(
                attachment_id, device_id, session_id, direction,
                filename, content_type, size_bytes, sha256, local_path,
                transferred_bytes, state, created_at, updated_at
            ) VALUES(?, NULL, ?, 'outbound', ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
            """,
            (
                record.attachment_id,
                _require_text(record.session_id, "session_id"),
                _require_text(record.filename, "filename"),
                _require_text(record.content_type, "content_type"),
                record.size_bytes,
                _require_text(record.sha256, "sha256"),
                _require_text(record.local_path, "local_path"),
                record.transferred_bytes,
                _serialize_datetime(record.created_at, "created_at"),
                _serialize_datetime(record.updated_at, "updated_at"),
            ),
        )

    def read_attachment(self, attachment_id: str) -> AttachmentRecord | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM mobile_attachments WHERE attachment_id = ?",
                (_require_text(attachment_id, "attachment_id"),),
            ).fetchone()
        return _attachment_from_row(row) if row is not None else None

    def read_ready_upload_by_local_path(
        self,
        *,
        session_id: str,
        local_path: str,
    ) -> AttachmentRecord | None:
        """按会话和内部路径查找已就绪的原始上传元数据。"""

        with self._lock:
            row = self._db.execute(
                """
                SELECT * FROM mobile_attachments
                WHERE session_id = ? AND local_path = ?
                  AND direction = 'upload' AND state = 'ready'
                """,
                (
                    _require_text(session_id, "session_id"),
                    _require_text(local_path, "local_path"),
                ),
            ).fetchone()
        return _attachment_from_row(row) if row is not None else None

    def require_ready_outbound(
        self,
        *,
        session_id: str,
        attachment_id: str,
    ) -> AttachmentRecord:
        """返回属于指定会话且已就绪的 outbound 附件。"""

        record = self.read_attachment(attachment_id)
        if record is None:
            raise AttachmentStateError(f"附件不存在: {attachment_id}")
        if (
            record.direction != "outbound"
            or record.session_id != session_id
            or record.state != "ready"
        ):
            raise AttachmentStateError(
                f"附件未就绪或不属于当前下载会话: {attachment_id}"
            )
        return record

    def require_upload_attachment(
        self,
        *,
        device_id: str,
        attachment_id: str,
    ) -> AttachmentRecord:
        record = self.read_attachment(attachment_id)
        if record is None:
            raise AttachmentStateError(f"附件不存在: {attachment_id}")
        if record.direction != "upload" or record.device_id != device_id:
            raise AttachmentStateError(f"附件不属于当前上传设备: {attachment_id}")
        if record.state != "transferring":
            raise AttachmentStateError(f"附件不处于传输状态: {attachment_id}")
        return record

    def require_owned_upload(
        self,
        *,
        device_id: str,
        session_id: str,
        attachment_id: str,
    ) -> AttachmentRecord:
        """返回属于指定设备和会话的上传，不限制其传输状态。"""

        record = self.read_attachment(attachment_id)
        if record is None:
            raise AttachmentStateError(f"附件不存在: {attachment_id}")
        if (
            record.direction != "upload"
            or record.device_id != device_id
            or record.session_id != session_id
        ):
            raise AttachmentStateError(f"附件不属于当前上传会话: {attachment_id}")
        return record

    def fail_attachment_upload(
        self,
        *,
        device_id: str,
        attachment_id: str,
        updated_at: datetime,
    ) -> AttachmentRecord:
        """把传输中的上传标记为失败，允许后续 begin 显式重置。"""

        with self._lock, self._db:
            updated = self._db.execute(
                """
                UPDATE mobile_attachments
                SET state = 'failed', updated_at = ?
                WHERE attachment_id = ? AND device_id = ?
                  AND direction = 'upload' AND state = 'transferring'
                """,
                (
                    _serialize_datetime(updated_at, "updated_at"),
                    _require_text(attachment_id, "attachment_id"),
                    _require_text(device_id, "device_id"),
                ),
            )
            if updated.rowcount != 1:
                raise AttachmentStateError(f"附件不能标记为失败: {attachment_id}")
            row = self._db.execute(
                "SELECT * FROM mobile_attachments WHERE attachment_id = ?",
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("已标记失败的附件记录在同一事务中消失")
        return _attachment_from_row(row)

    def reset_failed_upload(
        self,
        *,
        device_id: str,
        attachment_id: str,
        updated_at: datetime,
    ) -> AttachmentRecord:
        """把已失败上传重置到 offset 0，供同一附件重新发送。"""

        with self._lock, self._db:
            updated = self._db.execute(
                """
                UPDATE mobile_attachments
                SET state = 'transferring', transferred_bytes = 0, updated_at = ?
                WHERE attachment_id = ? AND device_id = ?
                  AND direction = 'upload' AND state = 'failed'
                """,
                (
                    _serialize_datetime(updated_at, "updated_at"),
                    _require_text(attachment_id, "attachment_id"),
                    _require_text(device_id, "device_id"),
                ),
            )
            if updated.rowcount != 1:
                raise AttachmentStateError(f"附件不能重置: {attachment_id}")
            row = self._db.execute(
                "SELECT * FROM mobile_attachments WHERE attachment_id = ?",
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("已重置附件记录在同一事务中消失")
        return _attachment_from_row(row)

    def advance_attachment(
        self,
        *,
        device_id: str,
        attachment_id: str,
        expected_offset: int,
        next_offset: int,
        updated_at: datetime,
    ) -> AttachmentRecord:
        """以 compare-and-set 推进已落盘的上传 offset。"""

        if next_offset <= expected_offset:
            raise ValueError("next_offset 必须大于 expected_offset")
        with self._lock, self._db:
            updated = self._db.execute(
                """
                UPDATE mobile_attachments
                SET transferred_bytes = ?, updated_at = ?
                WHERE attachment_id = ? AND device_id = ?
                  AND direction = 'upload' AND state = 'transferring'
                  AND transferred_bytes = ? AND size_bytes >= ?
                """,
                (
                    next_offset,
                    _serialize_datetime(updated_at, "updated_at"),
                    _require_text(attachment_id, "attachment_id"),
                    _require_text(device_id, "device_id"),
                    expected_offset,
                    next_offset,
                ),
            )
            if updated.rowcount != 1:
                raise AttachmentStateError(
                    f"附件 offset 推进冲突: {attachment_id}/{expected_offset}"
                )
            row = self._db.execute(
                "SELECT * FROM mobile_attachments WHERE attachment_id = ?",
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("已推进附件记录在同一事务中消失")
        return _attachment_from_row(row)

    def mark_attachment_ready(
        self,
        *,
        device_id: str,
        attachment_id: str,
        updated_at: datetime,
    ) -> AttachmentRecord:
        """只允许完整上传从 transferring 推进到 ready。"""

        with self._lock, self._db:
            updated = self._db.execute(
                """
                UPDATE mobile_attachments
                SET state = 'ready', updated_at = ?
                WHERE attachment_id = ? AND device_id = ?
                  AND direction = 'upload' AND state = 'transferring'
                  AND transferred_bytes = size_bytes
                """,
                (
                    _serialize_datetime(updated_at, "updated_at"),
                    _require_text(attachment_id, "attachment_id"),
                    _require_text(device_id, "device_id"),
                ),
            )
            if updated.rowcount != 1:
                raise AttachmentStateError(f"附件尚不能完成: {attachment_id}")
            row = self._db.execute(
                "SELECT * FROM mobile_attachments WHERE attachment_id = ?",
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("已完成附件记录在同一事务中消失")
        return _attachment_from_row(row)

    def require_ready_upload(
        self,
        *,
        device_id: str,
        session_id: str,
        attachment_id: str,
    ) -> AttachmentRecord:
        record = self.read_attachment(attachment_id)
        if record is None:
            raise AttachmentStateError(f"附件不存在: {attachment_id}")
        if (
            record.direction != "upload"
            or record.device_id != device_id
            or record.session_id != session_id
            or record.state != "ready"
        ):
            raise AttachmentStateError(f"附件未就绪或不属于当前消息: {attachment_id}")
        return record

    def reserve_command(
        self,
        *,
        device_id: str,
        command_id: str,
        command_type: str,
        request_hash: str,
        created_at: datetime,
    ) -> tuple[CommandReceipt, bool]:
        """原子占用命令 ID，并返回收据及是否首次创建。"""

        # 1. 固化来自协议边界的命令身份
        device_key = _require_text(device_id, "device_id")
        command_key = _require_text(command_id, "command_id")
        command_name = _require_text(command_type, "command_type")
        digest = _require_text(request_hash, "request_hash")
        timestamp = _serialize_datetime(created_at, "created_at")

        # 2. 在写事务内复用相同请求，拒绝命令 ID 冲突
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            _ = self._read_device_row(device_key)
            row = self._db.execute(
                """
                SELECT device_id, command_id, command_type, request_hash,
                       status, reply_type, reply_payload_json, session_id, turn_id
                FROM mobile_command_receipts
                WHERE device_id = ? AND command_id = ?
                """,
                (device_key, command_key),
            ).fetchone()
            if row is not None:
                receipt = _command_receipt_from_row(row)
                if (
                    receipt.command_type != command_name
                    or receipt.request_hash != digest
                ):
                    raise CommandConflictError(
                        f"命令 ID 已绑定其他请求: {device_key}/{command_key}"
                    )
                return receipt, False
            _ = self._db.execute(
                """
                INSERT INTO mobile_command_receipts(
                    device_id, command_id, command_type, request_hash,
                    status, created_at
                ) VALUES(?, ?, ?, ?, 'processing', ?)
                """,
                (device_key, command_key, command_name, digest, timestamp),
            )
        return CommandReceipt(
            device_id=device_key,
            command_id=command_key,
            command_type=command_name,
            request_hash=digest,
            status="processing",
            reply_type=None,
            reply_payload_json=None,
            session_id=None,
            turn_id=None,
        ), True

    def complete_command(
        self,
        *,
        device_id: str,
        command_id: str,
        reply_type: str,
        reply_payload_json: str,
        session_id: str | None,
        turn_id: str | None,
        completed_at: datetime,
    ) -> CommandReceipt:
        """原子保存命令结果，使断线重试得到同一回复。"""

        # 1. 校验要写入 SQLite 的稳定回复字段
        device_key = _require_text(device_id, "device_id")
        command_key = _require_text(command_id, "command_id")
        reply_name = _require_text(reply_type, "reply_type")
        payload = _require_text(reply_payload_json, "reply_payload_json")
        timestamp = _serialize_datetime(completed_at, "completed_at")

        # 2. 只允许 processing 收据推进为 completed
        with self._lock, self._db:
            _ = self._db.execute("BEGIN IMMEDIATE")
            updated = self._db.execute(
                """
                UPDATE mobile_command_receipts
                SET status = 'completed', reply_type = ?, reply_payload_json = ?,
                    session_id = ?, turn_id = ?, completed_at = ?
                WHERE device_id = ? AND command_id = ? AND status = 'processing'
                """,
                (
                    reply_name,
                    payload,
                    session_id,
                    turn_id,
                    timestamp,
                    device_key,
                    command_key,
                ),
            )
            if updated.rowcount != 1:
                raise MobileStorageError(
                    f"命令收据不处于 processing: {device_key}/{command_key}"
                )
            row = self._db.execute(
                """
                SELECT device_id, command_id, command_type, request_hash,
                       status, reply_type, reply_payload_json, session_id, turn_id
                FROM mobile_command_receipts
                WHERE device_id = ? AND command_id = ?
                """,
                (device_key, command_key),
            ).fetchone()
        if row is None:
            raise RuntimeError("已完成命令收据在同一事务中消失")
        return _command_receipt_from_row(row)

    def _init_schema(self) -> None:
        """创建移动端身份、配对、设备、cursor 和 inbox 表。"""

        # 1. 身份与配对状态
        _ = self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS mobile_server_identity (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                server_id TEXT NOT NULL,
                keyset_manifest_path TEXT NOT NULL,
                public_key_fingerprint TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mobile_connection_epoch (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                last_epoch INTEGER NOT NULL CHECK(last_epoch >= 0)
            );

            INSERT INTO mobile_connection_epoch(singleton, last_epoch)
            VALUES(1, 0)
            ON CONFLICT(singleton) DO NOTHING;

            CREATE TABLE IF NOT EXISTS mobile_pairing_sessions (
                pairing_id TEXT PRIMARY KEY,
                secret_hash TEXT,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(
                    status IN ('pending', 'confirmed', 'consumed', 'expired')
                ),
                CHECK(
                    (status = 'consumed' AND secret_hash IS NULL)
                    OR (status != 'consumed' AND secret_hash IS NOT NULL)
                )
            );

            CREATE INDEX IF NOT EXISTS idx_mobile_pairing_expiry
            ON mobile_pairing_sessions(status, expires_at);

            -- 2. 已配对设备与严格单调 cursor
            CREATE TABLE IF NOT EXISTS mobile_devices (
                device_id TEXT PRIMARY KEY,
                public_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                capabilities TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_mobile_devices_public_key
            ON mobile_devices(public_key);

            CREATE TABLE IF NOT EXISTS mobile_device_cursors (
                device_id TEXT PRIMARY KEY,
                next_event_seq INTEGER NOT NULL CHECK(next_event_seq >= 1),
                sent_event_seq INTEGER NOT NULL CHECK(sent_event_seq >= 0),
                acknowledged_event_seq INTEGER NOT NULL CHECK(
                    acknowledged_event_seq >= 0
                ),
                CHECK(acknowledged_event_seq <= sent_event_seq),
                CHECK(sent_event_seq < next_event_seq),
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
                    ON DELETE CASCADE
            );

            -- 3. 每设备 P0 durable inbox
            CREATE TABLE IF NOT EXISTS mobile_device_inbox (
                device_id TEXT NOT NULL,
                event_seq INTEGER NOT NULL CHECK(event_seq >= 1),
                event_id TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority = 'P0'),
                envelope_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(device_id, event_seq),
                UNIQUE(device_id, event_id),
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_mobile_inbox_created
            ON mobile_device_inbox(device_id, created_at);

            -- 4. 命令收据跨重连提供幂等回复
            CREATE TABLE IF NOT EXISTS mobile_command_receipts (
                device_id TEXT NOT NULL,
                command_id TEXT NOT NULL,
                command_type TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('processing', 'completed')),
                reply_type TEXT,
                reply_payload_json TEXT,
                session_id TEXT,
                turn_id TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                PRIMARY KEY(device_id, command_id),
                CHECK(
                    (status = 'processing' AND reply_type IS NULL
                     AND reply_payload_json IS NULL AND completed_at IS NULL)
                    OR
                    (status = 'completed' AND reply_type IS NOT NULL
                     AND reply_payload_json IS NOT NULL AND completed_at IS NOT NULL)
                ),
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
                    ON DELETE CASCADE
            );

            -- 5. 记录首次创建关系；认证设备共享 mobile 会话读取权限
            CREATE TABLE IF NOT EXISTS mobile_device_sessions (
                device_id TEXT NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(device_id, session_id),
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
                    ON DELETE CASCADE
            );

            -- 6. 附件元数据只暴露不透明 ID，本地路径留在服务端边界内
            CREATE TABLE IF NOT EXISTS mobile_attachments (
                attachment_id TEXT PRIMARY KEY,
                device_id TEXT,
                session_id TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('upload', 'outbound')),
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                sha256 TEXT NOT NULL,
                local_path TEXT NOT NULL,
                transferred_bytes INTEGER NOT NULL CHECK(transferred_bytes >= 0),
                state TEXT NOT NULL CHECK(state IN ('transferring', 'ready', 'failed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(transferred_bytes <= size_bytes),
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
                    ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_mobile_attachments_session
            ON mobile_attachments(session_id, created_at);

            DROP INDEX IF EXISTS idx_mobile_attachments_outbound_identity;
            CREATE INDEX idx_mobile_attachments_outbound_identity
            ON mobile_attachments(
                session_id, direction, state, filename, content_type, sha256, size_bytes
            );

            CREATE TABLE IF NOT EXISTS mobile_message_attachments (
                message_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                attachment_id TEXT NOT NULL,
                PRIMARY KEY(message_id, ordinal),
                FOREIGN KEY(attachment_id) REFERENCES mobile_attachments(attachment_id)
                    ON DELETE CASCADE
            );
            """
        )
        self._db.commit()

    def _insert_device(self, device: DeviceRecord) -> None:
        if device.revoked_at is not None:
            raise ValueError("新设备不能在注册时已撤销")
        capabilities = _serialize_capabilities(device.capabilities)
        _ = self._db.execute(
            """
            INSERT INTO mobile_devices(
                device_id, public_key, display_name, created_at,
                revoked_at, capabilities
            ) VALUES(?, ?, ?, ?, NULL, ?)
            """,
            (
                _require_text(device.device_id, "device_id"),
                _require_text(device.public_key, "public_key"),
                _require_text(device.display_name, "display_name"),
                _serialize_datetime(device.created_at, "created_at"),
                capabilities,
            ),
        )
        _ = self._db.execute(
            """
            INSERT INTO mobile_device_cursors(
                device_id, next_event_seq, sent_event_seq,
                acknowledged_event_seq
            ) VALUES(?, 1, 0, 0)
            """,
            (device.device_id,),
        )

    def _read_pairing_row(self, pairing_id: str) -> sqlite3.Row:
        row = self._db.execute(
            """
            SELECT pairing_id, secret_hash, expires_at, status
            FROM mobile_pairing_sessions
            WHERE pairing_id = ?
            """,
            (pairing_id,),
        ).fetchone()
        if row is None:
            raise UnknownPairingError(f"配对会话不存在: {pairing_id}")
        return cast(sqlite3.Row, row)

    def _read_device_row(self, device_id: str) -> sqlite3.Row:
        row = self._db.execute(
            """
            SELECT device_id, public_key, display_name, created_at,
                   revoked_at, capabilities
            FROM mobile_devices
            WHERE device_id = ?
            """,
            (device_id,),
        ).fetchone()
        if row is None:
            raise UnknownDeviceError(f"设备不存在: {device_id}")
        return cast(sqlite3.Row, row)

    def _read_cursor_row(self, device_id: str) -> sqlite3.Row:
        row = self._db.execute(
            """
            SELECT device_id, next_event_seq, sent_event_seq,
                   acknowledged_event_seq
            FROM mobile_device_cursors
            WHERE device_id = ?
            """,
            (device_id,),
        ).fetchone()
        if row is None:
            raise UnknownDeviceError(f"设备不存在或缺少 cursor: {device_id}")
        return cast(sqlite3.Row, row)

    def _require_device_cursor(self, device_id: str) -> None:
        _ = _require_text(device_id, "device_id")
        row = self._db.execute(
            "SELECT 1 FROM mobile_device_cursors WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        if row is None:
            raise UnknownDeviceError(f"设备不存在或缺少 cursor: {device_id}")


def _pairing_from_row(row: sqlite3.Row) -> PairingSessionRecord:
    status = _row_text(row, "status")
    if status not in {"pending", "confirmed", "consumed", "expired"}:
        raise ValueError(f"mobile_pairing_sessions.status 非法: {status}")
    secret_raw = row["secret_hash"]
    if secret_raw is not None and not isinstance(secret_raw, str):
        raise TypeError("mobile_pairing_sessions.secret_hash 必须为文本或 NULL")
    if status == "consumed" and secret_raw is not None:
        raise ValueError("consumed 配对会话仍然包含 secret_hash")
    if status != "consumed" and (not isinstance(secret_raw, str) or not secret_raw):
        raise ValueError(f"{status} 配对会话缺少 secret_hash")
    return PairingSessionRecord(
        pairing_id=_row_text(row, "pairing_id"),
        secret_hash=secret_raw,
        expires_at=_parse_datetime(_row_text(row, "expires_at"), "expires_at"),
        status=cast(PairingStatus, status),
    )


def _device_from_row(row: sqlite3.Row) -> DeviceRecord:
    revoked_raw = row["revoked_at"]
    if revoked_raw is not None and not isinstance(revoked_raw, str):
        raise TypeError("mobile_devices.revoked_at 必须为文本或 NULL")
    return DeviceRecord(
        device_id=_row_text(row, "device_id"),
        public_key=_row_text(row, "public_key"),
        display_name=_row_text(row, "display_name"),
        created_at=_parse_datetime(_row_text(row, "created_at"), "created_at"),
        revoked_at=(
            _parse_datetime(revoked_raw, "revoked_at")
            if revoked_raw is not None
            else None
        ),
        capabilities=_parse_capabilities(_row_text(row, "capabilities")),
    )


def _command_receipt_from_row(row: sqlite3.Row) -> CommandReceipt:
    status = _row_text(row, "status")
    if status not in {"processing", "completed"}:
        raise ValueError(f"mobile_command_receipts.status 非法: {status}")
    optional: dict[str, str | None] = {}
    for field in ("reply_type", "reply_payload_json", "session_id", "turn_id"):
        value = row[field]
        if value is not None and not isinstance(value, str):
            raise TypeError(f"mobile_command_receipts.{field} 必须为文本或 NULL")
        optional[field] = value
    if status == "processing" and (
        optional["reply_type"] is not None
        or optional["reply_payload_json"] is not None
    ):
        raise ValueError("processing 命令收据包含回复")
    if status == "completed" and (
        not optional["reply_type"] or not optional["reply_payload_json"]
    ):
        raise ValueError("completed 命令收据缺少回复")
    return CommandReceipt(
        device_id=_row_text(row, "device_id"),
        command_id=_row_text(row, "command_id"),
        command_type=_row_text(row, "command_type"),
        request_hash=_row_text(row, "request_hash"),
        status=cast(Literal["processing", "completed"], status),
        reply_type=optional["reply_type"],
        reply_payload_json=optional["reply_payload_json"],
        session_id=optional["session_id"],
        turn_id=optional["turn_id"],
    )


def _attachment_from_row(row: sqlite3.Row) -> AttachmentRecord:
    direction = _row_text(row, "direction")
    if direction not in {"upload", "outbound"}:
        raise ValueError(f"mobile_attachments.direction 非法: {direction}")
    state = _row_text(row, "state")
    if state not in {"transferring", "ready", "failed"}:
        raise ValueError(f"mobile_attachments.state 非法: {state}")
    device_id = row["device_id"]
    if device_id is not None and not isinstance(device_id, str):
        raise TypeError("mobile_attachments.device_id 必须为文本或 NULL")
    record = AttachmentRecord(
        attachment_id=_row_text(row, "attachment_id"),
        device_id=device_id,
        session_id=_row_text(row, "session_id"),
        direction=cast(AttachmentDirection, direction),
        filename=_row_text(row, "filename"),
        content_type=_row_text(row, "content_type"),
        size_bytes=_row_positive_int(row, "size_bytes"),
        sha256=_row_text(row, "sha256"),
        local_path=_row_text(row, "local_path"),
        transferred_bytes=_row_nonnegative_int(row, "transferred_bytes"),
        state=cast(AttachmentState, state),
        created_at=_parse_datetime(_row_text(row, "created_at"), "created_at"),
        updated_at=_parse_datetime(_row_text(row, "updated_at"), "updated_at"),
    )
    if record.transferred_bytes > record.size_bytes:
        raise ValueError("mobile_attachments.transferred_bytes 超过 size_bytes")
    return record


def _cursor_from_row(row: sqlite3.Row) -> DeviceCursor:
    cursor = DeviceCursor(
        device_id=_row_text(row, "device_id"),
        next_event_seq=_row_positive_int(row, "next_event_seq"),
        sent_event_seq=_row_nonnegative_int(row, "sent_event_seq"),
        acknowledged_event_seq=_row_nonnegative_int(
            row, "acknowledged_event_seq"
        ),
    )
    if cursor.sent_event_seq >= cursor.next_event_seq:
        raise ValueError("mobile_device_cursors.sent_event_seq 超过已分配上限")
    if cursor.acknowledged_event_seq > cursor.sent_event_seq:
        raise ValueError("mobile_device_cursors.acknowledged_event_seq 超过已发送上限")
    return cursor


def _inbox_event_from_row(row: sqlite3.Row) -> DurableInboxEvent:
    priority = _row_text(row, "priority")
    if priority != "P0":
        raise ValueError(f"mobile_device_inbox.priority 非法: {priority}")
    envelope_json = _row_text(row, "envelope_json")
    parsed = json.loads(envelope_json)
    if not isinstance(parsed, dict):
        raise TypeError("mobile_device_inbox.envelope_json 必须编码 JSON object")
    return DurableInboxEvent(
        device_id=_row_text(row, "device_id"),
        event_seq=_row_positive_int(row, "event_seq"),
        event_id=_row_text(row, "event_id"),
        priority="P0",
        envelope_json=envelope_json,
        created_at=_parse_datetime(_row_text(row, "created_at"), "created_at"),
    )


def _serialize_capabilities(capabilities: tuple[str, ...]) -> str:
    if len(set(capabilities)) != len(capabilities):
        raise ValueError("capabilities 不能包含重复项")
    values = [_require_text(value, "capability") for value in capabilities]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _parse_capabilities(raw: str) -> tuple[str, ...]:
    parsed: object = json.loads(raw)
    if not isinstance(parsed, list):
        raise TypeError("mobile_devices.capabilities 必须编码 JSON array")
    raw_values = cast(list[object], parsed)
    values: list[str] = []
    for value in raw_values:
        if not isinstance(value, str) or not value:
            raise TypeError("mobile_devices.capabilities 必须只包含非空文本")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("mobile_devices.capabilities 包含重复项")
    return tuple(values)


def _serialize_datetime(value: datetime, field: str) -> str:
    return _require_aware_datetime(value, field).astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str, field: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _require_aware_datetime(parsed, field)


def _require_aware_datetime(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} 必须包含时区")
    return value


def _require_text(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须为非空文本")
    return value


def _row_text(row: sqlite3.Row, field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value:
        raise TypeError(f"SQLite 字段 {field} 必须为非空文本")
    return value


def _row_positive_int(row: sqlite3.Row, field: str) -> int:
    value = row[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"SQLite 字段 {field} 必须为整数")
    if value < 1:
        raise ValueError(f"SQLite 字段 {field} 必须大于零")
    return value


def _row_nonnegative_int(row: sqlite3.Row, field: str) -> int:
    value = row[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"SQLite 字段 {field} 必须为整数")
    if value < 0:
        raise ValueError(f"SQLite 字段 {field} 不能为负数")
    return value
