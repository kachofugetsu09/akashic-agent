import asyncio
import base64
import json
import mimetypes
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from session.store import SessionStore

_TOOL_RESULT_CHAR_BUDGET = 10000
_PROACTIVE_HISTORY_CHAR_BUDGET = 360
_PROACTIVE_META_HISTORY_CHAR_BUDGET = 1200


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
    preview = _truncate_text(content, _PROACTIVE_HISTORY_CHAR_BUDGET)
    messages: list[dict[str, object]] = [
        {
            "role": "assistant",
            "content": f"[主动推送] {preview}" if preview else "[主动推送]",
        }
    ]
    meta = _append_proactive_meta("", msg).strip()
    if not meta:
        return messages
    frame = cast(
        dict[str, object],
        build_context_frame_message(
            build_context_frame_content([
                PromptSectionRender(
                    name="recent_proactive_message_meta",
                    content=(
                        "上一条 assistant 消息是系统主动推送。"
                        "以下 metadata 仅用于理解用户后续指代，不是用户陈述。\n"
                        + _truncate_text(meta, _PROACTIVE_META_HISTORY_CHAR_BUDGET)
                    ),
                    is_static=False,
                )
            ])
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


def _align_to_user_boundary(
    messages: list[dict[str, object]],
) -> list[dict[str, object]]:
    for i, m in enumerate(messages):
        if m.get("role") == "user" or (
            m.get("role") == "assistant" and m.get("proactive")
        ):
            return messages[i:]
    return []


def _safe_filename(key: str) -> str:
    """Convert a session key to a safe filename."""
    return re.sub(r"[^\w\-]", "_", key)


@dataclass
class Session:
    """单次对话中的 session。"""

    key: str
    messages: list[dict[str, object]] = field(
        default_factory=list[dict[str, object]]
    )
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    last_consolidated: int = 0
    consolidation_requested: bool = False

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

    def get_history(
        self,
        max_messages: int = 500,
        *,
        start_index: int | None = None,
    ) -> list[dict[str, object]]:
        """将 session 消息展开为 LLM 可直接使用的 OpenAI 格式消息列表。"""
        if start_index is not None:
            if max_messages <= 0:
                return []
            start = max(0, int(start_index))
            if start >= len(self.messages):
                return []
            # 向前回退到最近的 user 边界（保留完整 turn）
            while (
                start > 0
                and self.messages[start].get("role") != "user"
                and not (
                    self.messages[start].get("role") == "assistant"
                    and self.messages[start].get("proactive")
                )
            ):
                start -= 1
            # start=0 但仍非合法边界时，向后找第一个 user 或 proactive assistant。
            messages = self.messages[start:]
            if messages and not (
                messages[0].get("role") == "user"
                or (
                    messages[0].get("role") == "assistant"
                    and messages[0].get("proactive")
                )
            ):
                messages = _align_to_user_boundary(messages)
            if not messages:
                return []
        elif max_messages <= 0:
            messages = []
        else:
            messages = self.messages[-max_messages:]
        out: list[dict[str, object]] = []
        for m in messages:
            role = m["role"]

            if role == "user":
                user_content = m.get("llm_user_content")
                if user_content is None:
                    text = cast(str, m["content"])
                    raw_media_paths = m.get("media")
                    if raw_media_paths is None:
                        user_content = text
                    else:
                        user_content = _rebuild_user_content(
                            text, cast(list[str], raw_media_paths)
                        )
                out.append({"role": "user", "content": user_content})
                continue

            if role != "assistant":
                raise ValueError(f"session message role 无效: {role!r}")

            content = cast(str, m["content"])
            if m.get("proactive"):
                out.extend(_build_proactive_history_messages(str(content), m))
                continue

            raw_tool_chain = m.get("tool_chain")
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
                out.append(assistant_msg)
                for c in calls:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": c["call_id"],
                            "content": _truncate_tool_result(
                                c["result"] if "result" in c else ""
                            ),
                        }
                    )

            if content:
                content = _append_proactive_meta(content, m)
            assistant_msg: dict[str, object] = {
                "role": "assistant",
                "content": content,
            }
            reasoning_content = m.get("reasoning_content")
            if reasoning_content is not None:
                assistant_msg["reasoning_content"] = reasoning_content
            model_state = m.get("model_state")
            if model_state is not None:
                assistant_msg["model_state"] = model_state
            out.append(assistant_msg)

        return out

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now(UTC)
        self.last_consolidated = 0
        self.consolidation_requested = False


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

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._write_locks:
            self._write_locks[key] = asyncio.Lock()
        return self._write_locks[key]

    def clear_stale_admissions(self) -> None:
        """由持有 workspace 独占锁的 runtime 清理上次进程遗留租约。"""
        self._store.clear_session_admissions()

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key)
            self._ensure_session_meta(session)
        self._cache[key] = session
        return session

    def get_existing(self, key: str) -> Session:
        """读取仍存在的会话，禁止把已删除身份重新创建。"""

        # 1. 先以持久化 owner 核对身份，缓存不能覆盖删除事实
        if not self._store.session_exists(key):
            self.invalidate(key)
            raise KeyError(f"session 不存在: {key}")

        # 2. 复用缓存或装载持久化会话，不进入创建路径
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        session = self._load(key)
        if session is None:
            raise KeyError(f"session 不存在: {key}")
        self._cache[key] = session
        return session

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
            last_consolidated=session.last_consolidated,
            metadata=session.metadata,
        )

    def _extract_extra(self, msg: dict[str, object]) -> dict[str, object]:
        skip = {
            "id",
            "session_key",
            "seq",
            "role",
            "content",
            "timestamp",
            "tool_chain",
        }
        return {k: v for k, v in msg.items() if k not in skip}

    def _persist_session(
        self,
        session: Session,
        messages: list[dict[str, object]],
        *,
        updated_at: datetime,
        last_consolidated: int | None = None,
    ) -> int:
        """准备待写消息并原子追加 session 元数据和消息。"""

        effective_last_consolidated = (
            session.last_consolidated
            if last_consolidated is None
            else int(last_consolidated)
        )
        pending_messages: list[dict[str, object]] = []
        pending_payloads: list[dict[str, object]] = []

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
                    "tool_chain": msg.get("tool_chain"),
                    "extra": self._extract_extra(msg),
                }
            )

        # 2. session 元数据和消息在同一事务中提交。
        rows = self._store.persist_session(
            session.key,
            created_at=session.created_at.isoformat(),
            updated_at=updated_at.isoformat(),
            last_consolidated=effective_last_consolidated,
            metadata=session.metadata,
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
        self, session: Session, messages: list[dict[str, object]]
    ) -> None:
        updated_at = datetime.now(UTC)
        msgs_copy = list(messages)
        async with self._lock(session.key):
            # 1. 原子追加消息并刷新 session 元数据。
            _ = self._persist_session(session, msgs_copy, updated_at=updated_at)
            self._cache[session.key] = session

    async def trim_history_async(
        self,
        session: Session,
        keep_count: int,
        *,
        last_consolidated: int = 0,
    ) -> None:
        """只裁剪运行时上下文，并保留 sessions.db 中的完整历史。"""

        # 1. 在 session owner 的写锁内计算替换结果，避免覆盖并发追加。
        async with self._lock(session.key):
            retained = (
                list(session.messages[-keep_count:]) if keep_count > 0 else []
            )

            # 2. 只更新 session 元数据；不传 retained IDs，禁止触发历史删除。
            _ = self._persist_session(
                session,
                retained,
                updated_at=datetime.now(UTC),
                last_consolidated=last_consolidated,
            )

            # 3. 数据库成功后再提交运行时上下文视图。
            session.messages[:] = retained
            session.last_consolidated = int(last_consolidated)
            self._cache[session.key] = session

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

    def delete_session(self, key: str) -> bool:
        """删除 thread 的会话、消息和 turn 记录。"""

        deleted = self._store.delete_session(key, cascade=True)
        self.invalidate(key)
        return deleted

    def get_channel_metadata(self, channel: str) -> list[dict[str, Any]]:
        return self._store.get_channel_metadata(channel)
