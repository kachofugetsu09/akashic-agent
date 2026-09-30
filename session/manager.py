import asyncio
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from session.identities import ChannelIdentities
from session.admissions import SessionAdmissions
from session.inbound_store import InboundHandoffStore
from session.store import (
    SessionDeleteAudit,
    SessionStore,
    validate_message_delivery_id,
)

_STORED_TOOL_RESULT_CHAR_BUDGET = 20000
_MSG_KEYS = {"id", "session_key", "seq", "role", "content", "timestamp", "tool_chain"}


def _truncate_tool_chain_for_storage(tool_chain: object) -> object:
    """Copy a tool chain and bound every persisted tool result."""

    if tool_chain is None:
        return None

    # 1. Preserve the caller-owned runtime trace while preparing durable data.
    stored = deepcopy(cast(list[dict[str, object]], tool_chain))

    # 2. Truncate each result independently so one tool cannot dominate a turn.
    for group in stored:
        calls = cast(list[dict[str, object]], group["calls"])
        for call in calls:
            result = call.get("result")
            if (
                not isinstance(result, str)
                or len(result) <= _STORED_TOOL_RESULT_CHAR_BUDGET
            ):
                continue
            omitted = len(result) - _STORED_TOOL_RESULT_CHAR_BUDGET
            while True:
                marker = f"…{omitted} chars truncated before persistence…"
                keep = max(0, _STORED_TOOL_RESULT_CHAR_BUDGET - len(marker))
                actual_omitted = len(result) - keep
                if actual_omitted == omitted:
                    break
                omitted = actual_omitted
            head = keep // 2
            tail = keep - head
            call["result"] = result[:head] + marker + (result[-tail:] if tail else "")
    return stored


@dataclass
class Session:
    """单次对话中的 session。"""

    key: str
    messages: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    last_consolidated: int = 0
    def add_message(
        self, role: str, content: str, media: list[str] | None = None, **kwargs: object
    ) -> dict[str, object]:
        """向 session 追加一条消息并更新时间。"""
        msg: dict[str, object] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
            **kwargs,
        }
        if media:
            msg["media"] = list(media)
        self.messages.append(msg)
        self.updated_at = datetime.now(UTC)
        return msg

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now(UTC)


class SessionManager:
    _METADATA_REFRESH_EVERY: int = 10

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.session_dir = workspace / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = workspace / "sessions.db"
        with ExitStack() as stores:
            self._store = stores.enter_context(closing(SessionStore(self.db_path)))
            self.inbound_store = stores.enter_context(closing(InboundHandoffStore(self.db_path)))
            self.admissions = stores.enter_context(closing(SessionAdmissions(self.db_path)))
            self.identities = stores.enter_context(closing(ChannelIdentities(self.db_path)))
            self._stores = stores.pop_all()
        self._cache: dict[str, Session] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._write_locks:
            self._write_locks[key] = asyncio.Lock()
        return self._write_locks[key]

    def clear_stale_admissions(self) -> None:
        """由持有 workspace 独占锁的 runtime 清理上次进程遗留租约。"""
        self.admissions.clear_stale()

    def get_or_create(self, key: str) -> Session:
        cached = self._cache.get(key)
        meta = self._store.get_session_meta(key)
        if (
            cached is not None
            and meta is not None
            and self._cache_matches_meta(cached, meta)
        ):
            return cached

        session = self._load(key)
        if session is None:
            self.invalidate(key)
            session = Session(key)
            self._ensure_session_meta(session)
        self._cache[key] = session
        return session

    def get_existing(self, key: str) -> Session:
        """读取仍存在的会话，禁止把已删除身份重新创建。"""

        # 1. 先读取 Store-owned revision，缓存不能覆盖删除或外部更新事实
        meta = self._store.get_session_meta(key)
        if meta is None:
            self.invalidate(key)
            raise KeyError(f"session 不存在: {key}")

        # 2. 只有 revision 一致时复用缓存，否则从 canonical rows 重载
        cached = self._cache.get(key)
        if cached is not None and self._cache_matches_meta(cached, meta):
            return cached
        session = self._load(key)
        if session is None:
            raise KeyError(f"session 不存在: {key}")
        self._cache[key] = session
        return session

    @staticmethod
    def _cache_matches_meta(session: Session, meta: dict[str, Any]) -> bool:
        """比较缓存会话与 Store 持有的元数据修订字段。"""

        return session.updated_at.isoformat() == str(
            meta["updated_at"]
        ) and session.last_consolidated == int(meta["last_consolidated"])

    def admit_existing(self, key: str) -> tuple[Session, str]:
        """为仍存在的会话建立跨连接处理租约并返回会话。"""

        # 1. 持久化 owner 原子核对身份并建立租约
        try:
            admission_id = self.admissions.acquire(key)
        except KeyError:
            self.invalidate(key)
            raise

        # 2. 租约覆盖装载窗口；失败时立即回收
        try:
            return self.get_existing(key), admission_id
        except BaseException:
            self.admissions.release_admission(admission_id)
            raise

    def release_admission(self, admission_id: str) -> None:
        self.admissions.release_admission(admission_id)

    def peek_next_message_id(self, session_key: str) -> str:
        next_seq = self._store.next_seq(session_key)
        return f"{session_key}:{next_seq}"

    def _load(self, key: str) -> Session | None:
        meta = self._store.get_session_meta(key)
        messages = self._store.fetch_session_messages(key)
        if meta is None:
            if messages:
                raise ValueError(f"session metadata 缺失但存在 messages: {key}")
            return None

        created_at = datetime.fromisoformat(meta["created_at"])
        updated_at = datetime.fromisoformat(meta["updated_at"])
        metadata = meta["metadata"]
        last_consolidated = int(meta["last_consolidated"])
        return Session(
            key=key,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            last_consolidated=last_consolidated,
        )

    def _ensure_session_meta(self, session: Session) -> None:
        self._store.upsert_session(
            session.key,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            metadata=session.metadata,
        )

    def _persist_session(
        self,
        session: Session,
        messages: list[dict[str, object]],
        *,
        updated_at: datetime,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """准备待写消息并原子追加 session 元数据和消息。"""

        pending_messages: list[dict[str, object]] = []
        pending_payloads: list[dict[str, object]] = []

        if not self._store.session_exists(session.key) and session.last_consolidated:
            raise ValueError("新 session 的 last_consolidated 必须由 ledger 建立")

        # 1. 准备尚未持久化的消息，不提前修改内存中的稳定 id。
        for msg in messages:
            if msg.get("id"):
                continue
            ts = str(msg.get("timestamp") or datetime.now(UTC).isoformat())
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            pending_messages.append(msg)
            pending_payloads.append(
                {
                    "role": str(msg.get("role") or "assistant"),
                    "content": content,
                    "timestamp": ts,
                    "tool_chain": _truncate_tool_chain_for_storage(
                        msg.get("tool_chain")
                    ),
                    "extra": {k: v for k, v in msg.items() if k not in _MSG_KEYS},
                }
            )

        # 2. session 元数据和消息在同一事务中提交。
        rows = self._store.persist_session(
            session.key,
            created_at=session.created_at.isoformat(),
            updated_at=updated_at.isoformat(),
            metadata=dict(session.metadata if metadata is None else metadata),
            messages=pending_payloads,
        )
        for msg, row in zip(pending_messages, rows):
            msg.update(row)

        # 3. 保持会话消息缓存里的时间字段完整。
        for msg in messages:
            if "timestamp" not in msg:
                msg["timestamp"] = datetime.now(UTC).isoformat()

        session.updated_at = updated_at
        return len(rows)

    def save(self, session: Session) -> None:
        _ = self._persist_session(
            session,
            session.messages,
            updated_at=datetime.now(UTC),
        )
        self._cache[session.key] = session

    def close(self) -> None:
        self._stores.close()

    async def save_async(self, session: Session) -> None:
        async with self._lock(session.key):
            self.save(session)

    async def append_messages(
        self,
        session: Session,
        messages: list[dict[str, object]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        updated_at = datetime.now(UTC)
        msgs_copy = list(messages)
        async with self._lock(session.key):
            # 1. 原子追加消息并回填稳定 ID。
            _ = self._persist_session(
                session,
                msgs_copy,
                updated_at=updated_at,
                metadata=metadata,
            )

            # 2. 同一无 await 临界段把 pending rows 挂回当前 Session cache。
            attached = {id(message) for message in session.messages}
            session.messages.extend(
                message for message in msgs_copy if id(message) not in attached
            )
            if metadata is not None:
                session.metadata = dict(metadata)
            self._cache[session.key] = session

    async def append_durable_delivery(
        self,
        *,
        session_key: str,
        content: str,
        delivery_id: str,
        control_turn_id: str,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Append one proactive assistant projection exactly once per delivery id."""

        delivery_id = validate_message_delivery_id(delivery_id)
        message_metadata = dict(metadata or {})
        reserved = {
            "id",
            "role",
            "content",
            "timestamp",
            "delivery_id",
            "control_turn_id",
        }
        if conflict := reserved.intersection(message_metadata):
            raise ValueError(
                "durable delivery metadata 不得覆盖 Session 字段: "
                + ", ".join(sorted(conflict))
            )
        async with self._lock(session_key):
            # 1. A committed Session message is the crash-recovery receipt.
            existing = self._store.get_message_by_delivery_id(session_key, delivery_id)
            if existing is not None:
                if (
                    existing["content"] != content
                    or existing.get("control_turn_id") != control_turn_id
                    or any(
                        existing.get(key) != value
                        for key, value in message_metadata.items()
                    )
                ):
                    raise RuntimeError(
                        f"durable delivery Session projection conflict: {delivery_id}"
                    )
                return str(existing["id"])

            # 2. Persist and publish the new append-only projection under one lock.
            session = self.get_or_create(session_key)
            message: dict[str, object] = {
                **message_metadata,
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
                "proactive": True,
                "delivery_id": delivery_id,
                "control_turn_id": control_turn_id,
            }
            updated_at = datetime.now(UTC)
            _ = self._persist_session(
                session,
                [message],
                updated_at=updated_at,
            )
            session.messages.append(message)
            self._cache[session.key] = session
            return str(message["id"])

    def invalidate(self, key: str) -> None:
        _ = self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = self._store.list_sessions()
        for item in sessions:
            item["path"] = str(self.db_path)
        return sessions

    @property
    def control_store(self) -> SessionStore:
        """向会话控制服务暴露同一 SQLite owner，避免建立第二条连接。"""
        return self._store

    def session_exists(self, key: str) -> bool:
        return self._store.session_exists(key)

    def delete_session_with_audit(self, key: str) -> SessionDeleteAudit:
        """删除 thread 的会话、消息和 turn 记录。"""

        deletion = self._store.delete_session_with_audit(
            key,
            cascade=True,
            action_source="control.thread_delete",
        )
        if deletion.result == "committed":
            self.invalidate(key)
        return deletion

    def delete_session(self, key: str) -> bool:
        """删除 thread，并保留原有 bool 结果供 control service 使用。"""

        return self.delete_session_with_audit(key).result == "committed"
