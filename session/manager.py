import asyncio
import base64
import json
import logging
import mimetypes
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
    is_context_frame,
)
from session.store import SessionStore

logger = logging.getLogger(__name__)

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


def _append_proactive_meta(content: str, msg: dict[str, Any]) -> str:
    """Expose source trace and state tag back to the model without changing user-visible text."""
    if not msg.get("proactive"):
        return content
    meta_lines: list[str] = []
    state_tag = str(msg.get("state_summary_tag", "") or "").strip()
    if state_tag and state_tag != "none":
        meta_lines.append(f"state_summary_tag={state_tag}")
    source_refs = msg.get("source_refs") or []
    if isinstance(source_refs, list) and source_refs:
        meta_lines.append("sources:")
        for raw in source_refs[:1]:
            if not isinstance(raw, dict):
                continue
            parts = [
                str(raw.get("source_name", "") or "").strip(),
                str(raw.get("title", "") or "").strip(),
                str(raw.get("url", "") or "").strip(),
            ]
            meta_lines.append("- " + " | ".join(p for p in parts if p))
    if not meta_lines:
        return content
    return f"{content}\n\n[proactive_meta]\n" + "\n".join(meta_lines)


def _build_proactive_history_messages(
    content: str,
    msg: dict[str, Any],
) -> list[dict[str, str]]:
    preview = _truncate_text(content, _PROACTIVE_HISTORY_CHAR_BUDGET)
    messages = [
        {
            "role": "assistant",
            "content": f"[主动推送] {preview}" if preview else "[主动推送]",
        }
    ]
    meta = _append_proactive_meta("", msg).strip()
    if not meta:
        return messages
    frame = build_context_frame_message(
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
    )
    messages.append(frame)
    return messages


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"…（截断 {len(text) - limit} 字）"


def _rebuild_user_content(text: str, media_paths: list[str]) -> "str | list[dict]":
    """重建带附件的用户消息。图片内联 base64；非图片文件保留路径引用供 agent 调用 read_file。"""
    images = []
    file_refs = []
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
            except Exception:
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


def _align_to_user_boundary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
class UndoLastTurnPreview:
    message_ids: list[str]
    target_user_id: str
    target_assistant_id: str


@dataclass
class UndoLastTurnResult:
    deleted_ids: list[str]
    target_user_id: str
    target_assistant_id: str
    rollback_index: int
    last_consolidated_before: int
    last_consolidated_after: int


@dataclass
class Session:
    """单次对话中的 session。"""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0
    consolidation_requested: bool = False

    def add_message(
        self, role: str, content: str, media: list[str] | None = None, **kwargs: Any
    ) -> None:
        """Add a message to session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().astimezone().isoformat(),
            **kwargs,
        }
        if media:
            msg["media"] = list(media)
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(
        self,
        max_messages: int = 500,
        *,
        start_index: int | None = None,
    ) -> list[dict[str, Any]]:
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
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")

            if role == "user":
                context_frame = m.get("llm_context_frame")
                if isinstance(context_frame, str) and context_frame.strip():
                    out.append({"role": "user", "content": context_frame})
                user_content = m.get("llm_user_content")
                if user_content is None:
                    text = m.get("content", "")
                    media_paths = m.get("media") or []
                    user_content = (
                        _rebuild_user_content(text, media_paths)
                        if media_paths
                        else text
                    )
                out.append({"role": "user", "content": user_content})
                continue

            if role != "assistant":
                continue

            content = m.get("content", "") or ""
            if m.get("proactive"):
                out.extend(_build_proactive_history_messages(str(content), m))
                continue

            tool_chain: list[dict] = m.get("tool_chain") or []
            for group in tool_chain:
                calls: list[dict] = group.get("calls") or []
                if not calls:
                    continue
                assistant_msg = {
                    "role": "assistant",
                    "content": group.get("text"),
                    "tool_calls": [
                        {
                            "id": c["call_id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(
                                    c.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                        for c in calls
                    ],
                }
                reasoning_content = group.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    assistant_msg["reasoning_content"] = reasoning_content
                out.append(assistant_msg)
                for c in calls:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": c["call_id"],
                            "content": _truncate_tool_result(c.get("result", "")),
                        }
                    )

            if content:
                content = _append_proactive_meta(content, m)
            assistant_msg = {"role": "assistant", "content": content}
            reasoning_content = m.get("reasoning_content")
            if isinstance(reasoning_content, str):
                assistant_msg["reasoning_content"] = reasoning_content
            out.append(assistant_msg)

        return out

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now()
        self.last_consolidated = 0
        self.consolidation_requested = False


def _is_context_frame_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    return is_context_frame(str(message.get("content") or ""))


def _is_real_user_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and not _is_context_frame_message(message)


def _is_passive_assistant_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" and not bool(message.get("proactive"))


def _find_last_passive_turn(
    messages: list[dict[str, Any]],
) -> tuple[list[int], int, int] | None:
    for assistant_index in range(len(messages) - 1, -1, -1):
        if not _is_passive_assistant_message(messages[assistant_index]):
            continue
        user_index = assistant_index - 1
        while user_index >= 0 and _is_context_frame_message(messages[user_index]):
            user_index -= 1
        if user_index < 0 or not _is_real_user_message(messages[user_index]):
            continue
        delete_indices = [user_index, assistant_index]
        context_index = user_index - 1
        while context_index >= 0 and _is_context_frame_message(messages[context_index]):
            delete_indices.insert(0, context_index)
            context_index -= 1
        return delete_indices, user_index, assistant_index
    return None


def _compute_rollback_index(
    messages: list[dict[str, Any]],
    *,
    delete_indices: list[int],
    old_last_consolidated: int,
    rollback_source_ids: list[str],
) -> int:
    if not delete_indices:
        return min(old_last_consolidated, len(messages))
    rollback_index = min(delete_indices)
    if rollback_index >= old_last_consolidated:
        return min(old_last_consolidated, len(messages) - len(delete_indices))
    source_ids = {str(item).strip() for item in rollback_source_ids if str(item).strip()}
    for index, message in enumerate(messages):
        msg_id = str(message.get("id") or "").strip()
        if msg_id and msg_id in source_ids:
            rollback_index = min(rollback_index, index)
    return max(0, min(rollback_index, old_last_consolidated))


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

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key)
            self._ensure_session_meta(session)
        self._cache[key] = session
        return session

    def peek_next_message_id(self, session_key: str) -> str:
        next_seq = self._store.next_seq(session_key)
        return f"{session_key}:{next_seq}"

    def _load(self, key: str) -> Session | None:
        meta = self._store.get_session_meta(key)
        messages = self._store.fetch_session_messages(key)
        if meta is None and not messages:
            return None

        created_at = (
            datetime.fromisoformat(meta["created_at"])
            if meta and meta.get("created_at")
            else datetime.now()
        )
        updated_at = (
            datetime.fromisoformat(meta["updated_at"])
            if meta and meta.get("updated_at")
            else datetime.now()
        )
        metadata = meta.get("metadata", {}) if meta else {}
        last_consolidated = int(meta.get("last_consolidated", 0)) if meta else 0
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

    def _extract_extra(self, msg: dict[str, Any]) -> dict[str, Any]:
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

    def _persist_messages(self, session: Session, messages: list[dict[str, Any]]) -> int:
        next_seq = self._store.next_seq(session.key)
        inserted = 0

        # 1. 只写入尚未持久化（没有 id）的消息。
        for msg in messages:
            if msg.get("id"):
                continue
            ts = str(msg.get("timestamp") or datetime.now().astimezone().isoformat())
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            row = self._store.insert_message(
                session.key,
                role=str(msg.get("role") or "assistant"),
                content=content,
                ts=ts,
                seq=next_seq,
                tool_chain=msg.get("tool_chain"),
                extra=self._extract_extra(msg),
            )
            msg.update(row)
            next_seq += 1
            inserted += 1

        # 2. 保持会话消息缓存里的时间字段完整。
        for msg in messages:
            if "timestamp" not in msg:
                msg["timestamp"] = datetime.now().astimezone().isoformat()

        return inserted

    def save(self, session: Session) -> None:
        session.updated_at = datetime.now()
        self._ensure_session_meta(session)
        self._persist_messages(session, session.messages)
        self._store.upsert_session(
            session.key,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            last_consolidated=session.last_consolidated,
            metadata=session.metadata,
        )
        self._cache[session.key] = session

    async def save_async(self, session: Session) -> None:
        session.updated_at = datetime.now()
        async with self._lock(session.key):
            self.save(session)

    async def append_messages(self, session: Session, messages: list[dict]) -> None:
        session.updated_at = datetime.now()
        msgs_copy = list(messages)
        async with self._lock(session.key):
            # 1. 确保 session 元数据存在并刷新 updated_at。
            self._ensure_session_meta(session)
            # 2. 追加写入本次新增消息，并补齐稳定 id。
            self._persist_messages(session, msgs_copy)
            # 3. 回写 session 元数据（含 last_consolidated / metadata）。
            self._store.upsert_session(
                session.key,
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
                last_consolidated=session.last_consolidated,
                metadata=session.metadata,
            )
            self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def preview_last_turn_undo(self, session_key: str) -> UndoLastTurnPreview | None:
        session = self.get_or_create(session_key)
        target = _find_last_passive_turn(session.messages)
        if target is None:
            return None
        delete_indices, user_index, assistant_index = target
        message_ids = [
            str(session.messages[i].get("id") or "")
            for i in delete_indices
            if str(session.messages[i].get("id") or "").strip()
        ]
        if len(message_ids) != len(delete_indices):
            return None
        return UndoLastTurnPreview(
            message_ids=message_ids,
            target_user_id=str(session.messages[user_index].get("id") or ""),
            target_assistant_id=str(session.messages[assistant_index].get("id") or ""),
        )

    async def undo_last_turn(
        self,
        session_key: str,
        *,
        rollback_source_ids: list[str] | None = None,
        expected_message_ids: list[str] | None = None,
        rollback_source_resolver: Callable[[list[str]], list[str]] | None = None,
    ) -> UndoLastTurnResult | None:
        async with self._lock(session_key):
            session = self.get_or_create(session_key)
            target = _find_last_passive_turn(session.messages)
            if target is None:
                return None
            delete_indices, user_index, assistant_index = target
            deleted_ids = [
                str(session.messages[i].get("id") or "")
                for i in delete_indices
                if str(session.messages[i].get("id") or "").strip()
            ]
            if len(deleted_ids) != len(delete_indices):
                return None
            expected = [
                str(message_id).strip()
                for message_id in (expected_message_ids or [])
                if str(message_id).strip()
            ]
            if expected and expected != deleted_ids:
                return None
            if rollback_source_resolver is not None:
                rollback_source_ids = rollback_source_resolver(list(deleted_ids))
            target_user_id = str(session.messages[user_index].get("id") or "")
            target_assistant_id = str(session.messages[assistant_index].get("id") or "")
            old_last = max(0, int(session.last_consolidated))
            rollback_index = _compute_rollback_index(
                session.messages,
                delete_indices=delete_indices,
                old_last_consolidated=old_last,
                rollback_source_ids=rollback_source_ids or [],
            )
            remaining = [
                msg for i, msg in enumerate(session.messages) if i not in set(delete_indices)
            ]
            deleted_before = sum(1 for i in delete_indices if i < rollback_index)
            new_last = max(0, rollback_index - deleted_before)
            new_last = min(new_last, len(remaining))
            deleted_count = self._store.delete_session_messages_and_update_cursor(
                session.key,
                ids=deleted_ids,
                last_consolidated=new_last,
            )
            if deleted_count != len(deleted_ids):
                self.invalidate(session.key)
                return None
            session.messages = remaining
            session.last_consolidated = new_last
            session.updated_at = datetime.now()
            self._cache[session.key] = session
            return UndoLastTurnResult(
                deleted_ids=deleted_ids,
                target_user_id=target_user_id,
                target_assistant_id=target_assistant_id,
                rollback_index=rollback_index,
                last_consolidated_before=old_last,
                last_consolidated_after=new_last,
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = self._store.list_sessions()
        for item in sessions:
            item["path"] = str(self.db_path)
        return sessions

    def get_channel_metadata(self, channel: str) -> list[dict[str, Any]]:
        try:
            return self._store.get_channel_metadata(channel)
        except Exception as e:
            logging.warning("Failed to read channel metadata for %s: %s", channel, e)
            return []
