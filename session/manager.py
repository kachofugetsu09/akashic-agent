import asyncio
import base64
import json
import mimetypes
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from agent.plugin_composition.request_projection import SessionHistoryUnit
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from session.store import (
    ChannelIdentityWriteReceipt,
    SessionDeleteAudit,
    SessionStore,
    validate_message_delivery_id,
)

_TOOL_RESULT_CHAR_BUDGET = 10000
_STORED_TOOL_RESULT_CHAR_BUDGET = 20000
_PROACTIVE_META_HISTORY_CHAR_BUDGET = 1200
_MSG_KEYS = {"id", "session_key", "seq", "role", "content", "timestamp", "tool_chain"}


def _truncate_tool_result(content: object) -> str:
    text = content if isinstance(content, str) else str(content)
    if len(text) <= _TOOL_RESULT_CHAR_BUDGET:
        return text
    omitted = len(text) - _TOOL_RESULT_CHAR_BUDGET
    while True:
        marker = f"…{omitted} chars truncated…"
        keep = max(0, _TOOL_RESULT_CHAR_BUDGET - len(marker))
        actual_omitted = len(text) - keep
        if actual_omitted == omitted:
            break
        omitted = actual_omitted
    head = keep // 2
    tail = keep - head
    truncated = text[:head] + marker + (text[-tail:] if tail else "")
    return f"Total output lines: {len(text.splitlines())}\n\n{truncated}"


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


def _append_proactive_meta(content: str, msg: dict[str, object]) -> str:
    """向模型补充来源和状态标签，但不改变用户可见正文。"""
    if not msg.get("proactive"):
        return content
    meta_lines: list[str] = []
    raw_state_tag = msg.get("state_summary_tag")
    state_tag = cast(str, raw_state_tag).strip() if raw_state_tag is not None else ""
    if state_tag and state_tag != "none":
        meta_lines.append(f"state_summary_tag={state_tag}")
    raw_source_refs = msg.get("source_refs")
    if raw_source_refs is not None:
        source_refs = cast(list[dict[str, object]], raw_source_refs)
        if source_refs:
            meta_lines.append("sources:")
            raw = source_refs[0]
            parts: list[str] = []
            for field in ("source_name", "title", "url"):
                value = raw.get(field)
                if value is not None and str(value).strip():
                    parts.append(str(value).strip())
            if parts:
                meta_lines.append("- " + " | ".join(parts))
    if not meta_lines:
        return content
    return f"{content}\n\n[proactive_meta]\n" + "\n".join(meta_lines)


def _build_proactive_history_messages(
    content: str,
    msg: dict[str, object],
) -> list[dict[str, object]]:
    """将已送达的主动消息完整投影为历史正文和来源 frame。"""

    # 1. 保留完整正文，让后续指代可以命中主动消息中的任意内容。
    messages: list[dict[str, object]] = [
        {
            "role": "assistant",
            "content": f"[主动推送] {content}" if content else "[主动推送]",
        }
    ]

    # 2. 追加独立 metadata frame，保持来源不伪装成用户陈述。
    meta = _append_proactive_meta("", msg).strip()
    if not meta:
        return messages
    frame = cast(
        dict[str, object],
        build_context_frame_message(
            build_context_frame_content(
                [
                    PromptSectionRender(
                        name="recent_proactive_message_meta",
                        content=(
                            "上一条 assistant 消息是系统主动推送。"
                            "以下 metadata 仅用于理解用户后续指代，不是用户陈述。\n"
                            + _truncate_text(meta, _PROACTIVE_META_HISTORY_CHAR_BUDGET)
                        ),
                        is_static=False,
                    )
                ]
            )
        ),
    )
    messages.append(frame)
    return messages


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"…（截断 {len(text) - limit} 字）"


def _rebuild_user_content(
    text: str, media_paths: list[str]
) -> "str | list[dict[str, object]]":
    """重建带附件的用户消息。图片内联 base64；非图片文件保留路径引用供 agent 调用 read_file。"""
    images: list[dict[str, object]] = []
    file_refs: list[str] = []
    for path in media_paths:
        p = Path(path)
        mime, _ = mimetypes.guess_type(p)
        if mime and mime.startswith("image/") and p.is_file():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode()
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            except OSError:
                file_refs.append(f"[图片（读取失败）: {p.name}]")
        else:
            if p.is_file():
                file_refs.append(f"[文件: {path}]")
            else:
                file_refs.append(f"[文件（已失效）: {p.name}]")

    prefix = "\n".join(file_refs) + "\n" if file_refs else ""
    combined_text = (prefix + text).strip()

    if not images:
        return combined_text
    return images + [{"type": "text", "text": combined_text}]


def logical_history_unit_ranges(
    messages: list[dict[str, object]],
) -> list[tuple[int, int]]:
    """把 canonical messages 划分为不可拆分的历史单元。"""

    # 1. 新格式按连续 control turn 聚合；proactive 永远是独立事件。
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        start = index
        message = messages[index]
        raw_turn_id = message.get("control_turn_id")
        if raw_turn_id is not None:
            if not isinstance(raw_turn_id, str) or not raw_turn_id:
                raise ValueError("session message control_turn_id 必须是非空字符串")
            index += 1
            while (
                index < len(messages)
                and messages[index].get("control_turn_id") == raw_turn_id
            ):
                index += 1
            ranges.append((start, index))
            continue
        if message.get("role") == "assistant" and message.get("proactive"):
            ranges.append((start, start + 1))
            index += 1
            continue

        # 2. legacy 数据沿既有 user/proactive 边界分组，不反推新 turn identity。
        index += 1
        while index < len(messages):
            candidate = messages[index]
            if candidate.get("control_turn_id") is not None:
                break
            if candidate.get("role") == "user" or (
                candidate.get("role") == "assistant" and candidate.get("proactive")
            ):
                break
            index += 1
        ranges.append((start, index))
    return ranges


def logical_history_tail_start(
    messages: list[dict[str, object]],
    max_units: int,
) -> int:
    """返回最后 max_units 个完整历史单元的消息起点。"""

    if max_units <= 0:
        return len(messages)
    ranges = logical_history_unit_ranges(messages)
    if len(ranges) <= max_units:
        return 0
    return ranges[-max_units][0]


def _render_session_messages(
    messages: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Render canonical rows and retain the originating row beside each provider message."""

    rendered: list[tuple[dict[str, object], dict[str, object]]] = []
    for row in messages:
        role = row["role"]
        if role == "user":
            user_content = row.get("llm_user_content")
            if user_content is None:
                text = cast(str, row["content"])
                raw_media_paths = row.get("media")
                user_content = (
                    text
                    if raw_media_paths is None
                    else _rebuild_user_content(text, cast(list[str], raw_media_paths))
                )
            rendered.append(({"role": "user", "content": user_content}, row))
            continue
        if role != "assistant":
            raise ValueError(f"session message role 无效: {role!r}")
        content = cast(str, row["content"])
        if row.get("proactive"):
            rendered.extend(
                (message, row)
                for message in _build_proactive_history_messages(str(content), row)
            )
            continue
        raw_tool_chain = row.get("tool_chain")
        tool_chain = (
            cast(list[dict[str, object]], raw_tool_chain)
            if raw_tool_chain is not None
            else []
        )
        for group in tool_chain:
            calls = cast(list[dict[str, object]], group["calls"])
            if not calls:
                continue
            assistant_msg: dict[str, object] = {
                "role": "assistant",
                "content": group.get("text"),
                "tool_calls": [
                    {
                        "id": c["call_id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(
                                c["arguments"] if "arguments" in c else {},
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for c in calls
                ],
            }
            reasoning_content = group.get("reasoning_content")
            if reasoning_content is not None:
                assistant_msg["reasoning_content"] = reasoning_content
            model_state = group.get("model_state")
            if model_state is not None:
                assistant_msg["model_state"] = model_state
            rendered.append((assistant_msg, row))
            for call in calls:
                rendered.append(
                    (
                        {
                            "role": "tool",
                            "tool_call_id": call["call_id"],
                            "content": _truncate_tool_result(
                                call["result"] if "result" in call else ""
                            ),
                        },
                        row,
                    )
                )
        if content:
            content = _append_proactive_meta(content, row)
        assistant_msg = {"role": "assistant", "content": content}
        reasoning_content = row.get("reasoning_content")
        if reasoning_content is not None:
            assistant_msg["reasoning_content"] = reasoning_content
        model_state = row.get("model_state")
        if model_state is not None:
            assistant_msg["model_state"] = model_state
        rendered.append((assistant_msg, row))
    return rendered


@dataclass
class Session:
    """单次对话中的 session。"""

    key: str
    messages: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    last_consolidated: int = 0
    _projection_grant_issuer: Callable[[str], object] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _projection_grant_revoker: Callable[[object], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def issue_projection_grant(self, turn_id: str) -> object:
        """Issue one opaque current-turn capability through the Session owner."""

        issuer = self._projection_grant_issuer
        if issuer is None:
            raise RuntimeError("Session 缺少 projection grant issuer")
        return issuer(turn_id)

    def bind_projection_grant_issuer(self, issuer: Callable[[str], object]) -> None:
        """Let the Session owner attach its opaque capability issuer."""

        self._projection_grant_issuer = issuer

    def bind_projection_grant_revoker(self, revoker: Callable[[object], None]) -> None:
        """Let the Session owner revoke a previously issued capability."""

        self._projection_grant_revoker = revoker

    def revoke_projection_grant(self, grant: object) -> None:
        """Revoke a request-projection capability at the Turn boundary."""

        revoker = self._projection_grant_revoker
        if revoker is None:
            raise RuntimeError("Session 缺少 projection grant revoker")
        revoker(grant)

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

    def get_history(self, max_messages: int = 500) -> list[dict[str, object]]:
        """按完整历史单元选择尾部窗口并展开为 provider 消息。"""
        if max_messages <= 0:
            messages = []
        else:
            start = logical_history_tail_start(self.messages, max_messages)
            messages = self.messages[start:]
        return [message for message, _ in _render_session_messages(messages)]

    def history_units(self, *, after_seq: int = -1) -> tuple[SessionHistoryUnit, ...]:
        """Render complete canonical history units with immutable DB provenance."""

        if (
            not isinstance(after_seq, int)
            or isinstance(after_seq, bool)
            or after_seq < -1
        ):
            raise ValueError("history unit after_seq 必须是大于等于 -1 的整数")
        units: list[SessionHistoryUnit] = []
        for unit_index, (start, end) in enumerate(
            logical_history_unit_ranges(self.messages)
        ):
            source_rows = self.messages[start:end]
            rendered_with_refs: list[tuple[dict[str, object], tuple[str, int]]] = []
            source_ids: list[str] = []
            source_seqs: list[int] = []
            for row in source_rows:
                raw_id = row.get("id")
                raw_seq = row.get("seq")
                if (
                    not isinstance(raw_id, str)
                    or not raw_id
                    or not isinstance(raw_seq, int)
                ):
                    source_ids = [f"active:unpersisted:{unit_index}"]
                    source_seqs = []
                    break
                source_ids.append(raw_id)
                source_seqs.append(raw_seq)
            for row in source_rows:
                row_rendered = _render_session_messages([row])
                raw_id = row.get("id")
                raw_seq = row.get("seq")
                row_ref = (
                    (str(raw_id), int(raw_seq))
                    if isinstance(raw_id, str) and raw_id and isinstance(raw_seq, int)
                    else (source_ids[0], 0)
                )
                rendered_with_refs.extend(
                    (message, row_ref) for message, _ in row_rendered
                )
            if not rendered_with_refs:
                continue
            if source_seqs and max(source_seqs) <= after_seq:
                continue
            units.append(
                SessionHistoryUnit(
                    source_from_seq=min(source_seqs) if source_seqs else 0,
                    consolidated_through_seq=max(source_seqs) if source_seqs else 0,
                    source_message_ids=tuple(dict.fromkeys(source_ids)),
                    messages=tuple(message for message, _ in rendered_with_refs),
                    message_refs=tuple(ref for _, ref in rendered_with_refs),
                )
            )
        return tuple(units)

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
        self._store = SessionStore(self.db_path)
        self._cache: dict[str, Session] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._projection_grant_key = object()
        self._active_projection_grants: set[object] = set()

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._write_locks:
            self._write_locks[key] = asyncio.Lock()
        return self._write_locks[key]

    def clear_stale_admissions(self) -> None:
        """由持有 workspace 独占锁的 runtime 清理上次进程遗留租约。"""
        self._store.clear_session_admissions()

    def get_or_create(self, key: str) -> Session:
        cached = self._cache.get(key)
        meta = self._store.get_session_meta(key)
        if (
            cached is not None
            and meta is not None
            and self._cache_matches_meta(cached, meta)
        ):
            self._bind_projection_access(cached)
            return cached

        session = self._load(key)
        if session is None:
            self.invalidate(key)
            session = Session(key)
            self._ensure_session_meta(session)
        self._bind_projection_access(session)
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
            self._bind_projection_access(cached)
            return cached
        session = self._load(key)
        if session is None:
            raise KeyError(f"session 不存在: {key}")
        self._bind_projection_access(session)
        self._cache[key] = session
        return session

    def _bind_projection_access(self, session: Session) -> None:
        """Attach an issuer without exposing SessionStore or manager methods."""

        from agent.plugin_composition.session_compaction import SessionProjectionGrant

        def issue(turn_id: str) -> object:
            grant = SessionProjectionGrant.issue(
                self._projection_grant_key,
                session_key=session.key,
                session_created_at=session.created_at.isoformat(),
                turn_id=turn_id,
            )
            self._active_projection_grants.add(grant._nonce)
            return grant

        session.bind_projection_grant_issuer(issue)
        session.bind_projection_grant_revoker(self.revoke_projection_grant)

    def revoke_projection_grant(self, grant: object) -> None:
        """Revoke one owner-issued projection grant idempotently."""

        from agent.plugin_composition.session_compaction import SessionProjectionGrant

        if isinstance(grant, SessionProjectionGrant):
            self._active_projection_grants.discard(grant._nonce)

    def validate_projection_grant(
        self,
        grant: object,
        *,
        session_key: str,
        session_created_at: str,
    ) -> bool:
        """Validate an opaque grant without exposing its issuer to plugins."""

        from agent.plugin_composition.session_compaction import SessionProjectionGrant

        return (
            isinstance(grant, SessionProjectionGrant)
            and grant._nonce in self._active_projection_grants
            and grant.allows(
            self._projection_grant_key,
            session_key=session_key,
            session_created_at=session_created_at,
            )
        )

    @staticmethod
    def _cache_matches_meta(session: Session, meta: dict[str, Any]) -> bool:
        """比较缓存会话与 Store 持有的元数据修订字段。"""

        return session.updated_at.isoformat() == str(
            meta["updated_at"]
        ) and session.last_consolidated == int(meta["last_consolidated"])

    def admit_existing(self, key: str) -> tuple[Session, str]:
        """为仍存在的会话建立跨连接处理租约并返回会话。"""

        # 1. 持久化 owner 原子核对身份并建立租约
        admission_id = uuid4().hex
        if not self._store.acquire_session_admission(key, admission_id):
            self.invalidate(key)
            raise KeyError(f"session 不存在: {key}")

        # 2. 租约覆盖装载窗口；失败时立即回收
        try:
            return self.get_existing(key), admission_id
        except BaseException:
            self._store.release_session_admission(admission_id)
            raise

    def release_admission(self, admission_id: str) -> None:
        self._store.release_session_admission(admission_id)

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
        self._store.close()

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

    def get_channel_metadata(self, channel: str) -> list[dict[str, Any]]:
        return self._store.get_channel_metadata(channel)

    def get_channel_identities(self, channel: str) -> dict[str, str]:
        return self._store.get_channel_identities(channel)

    def channel_identity_migration_completed(self, channel: str) -> bool:
        return self._store.channel_identity_migration_completed(channel)

    def seed_channel_identities(
        self,
        channel: str,
        mapping: Mapping[str, tuple[str, str]],
    ) -> None:
        self._store.seed_channel_identities(channel, mapping)

    async def remember_channel_identity(
        self,
        *,
        channel: str,
        identity: str,
        chat_id: str,
        metadata_key: str,
    ) -> ChannelIdentityWriteReceipt:
        """Atomically move one durable identity to its target Session."""

        session_key = f"{channel}:{chat_id}"
        async with self._lock(session_key):
            # 1. Build a transient Session without creating a durable row.
            stored = self._store.get_session_meta(session_key)
            session = (
                Session(session_key)
                if stored is None
                else self.get_existing(session_key)
            )
            updated_at = datetime.now(UTC)
            metadata = dict(session.metadata)
            metadata[metadata_key] = identity

            # 2. Commit the Session metadata and unique identity owner together.
            receipt = self._store.persist_channel_identity(
                channel=channel,
                identity=identity,
                chat_id=chat_id,
                session_key=session.key,
                created_at=session.created_at.isoformat(),
                updated_at=updated_at.isoformat(),
                metadata=metadata,
            )

            # 3. Adopt the committed state only after SQLite succeeds.
            session.metadata = metadata
            session.updated_at = datetime.fromisoformat(receipt.committed_updated_at)
            self._cache[session.key] = session
            return receipt

    async def rollback_channel_identity(
        self,
        receipt: ChannelIdentityWriteReceipt,
    ) -> bool:
        """撤销仍由失败 acceptance attempt 拥有的 identity 写入。"""

        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("channel identity rollback receipt 类型无效")
        async with self._lock(receipt.session_key):
            rolled_back = self._store.rollback_channel_identity(receipt)
            if rolled_back:
                self.invalidate(receipt.session_key)
            return rolled_back
