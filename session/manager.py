import asyncio
import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from session.store import SessionStore

logger = logging.getLogger(__name__)

_TOOL_RESULT_CHAR_BUDGET = 4_000


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
        if m.get("role") == "user":
            return messages[i:]
    return []


def _safe_filename(key: str) -> str:
    """Convert a session key to a safe filename."""
    return re.sub(r"[^\w\-]", "_", key)


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
        cache_tail_mode = False
        if start_index is not None:
            if max_messages <= 0:
                return []
            start = max(0, int(start_index))
            if start >= len(self.messages):
                return []
            # 向前回退到最近的 user 边界（保留完整 turn）
            while start > 0 and self.messages[start].get("role") != "user":
                start -= 1
            # start=0 但仍非 user 时，向后找第一个 user
            messages = self.messages[start:]
            if messages and messages[0].get("role") != "user":
                messages = _align_to_user_boundary(messages)
            if not messages:
                return []
            if len(messages) > max_messages * 2:
                self.consolidation_requested = True
                logger.warning(
                    "get_history: consolidated tail (%d) exceeds 2x window (%d), "
                    "falling back to sliding window",
                    len(messages), max_messages,
                )
                messages = self.messages[-max_messages:]
                messages = _align_to_user_boundary(messages)
            else:
                cache_tail_mode = True
        elif max_messages <= 0:
            messages = []
        else:
            messages = self.messages[-max_messages:]
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")

            if role == "user":
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

            content = m.get("content", "") or ""
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
