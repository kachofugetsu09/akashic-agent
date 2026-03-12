import asyncio
import base64
import json
import logging
import mimetypes
import re
from dataclasses import field, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 保留完整 tool_result 的最近轮次数；更早的轮次仅保留调用结构，结果替换为占位符
_RECENT_TOOL_ROUNDS = 3
_CLEARED = "[已清除]"
_INFERENCE_TAG = "[以下为推演内容，本轮未调用工具，不可作为事实依据]\n"


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


def _safe_filename(key: str) -> str:
    """Convert a session key to a safe filename."""
    return re.sub(r"[^\w\-]", "_", key)


@dataclass
class Session:
    """
    单次对话中的session,用JSONL格式储存。
    消息是append-only的。
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

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

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """将 session 消息展开为 LLM 可直接使用的 OpenAI 格式消息列表。

        assistant 消息中的 tool_chain 会被展开为：
          assistant(tool_calls) → tool(result) → ... → assistant(final_text)

        近期 _RECENT_TOOL_ROUNDS 个 assistant 轮次保留完整 tool_result；
        更早的轮次将 tool_result 内容替换为占位符，节省 token 同时保留因果结构。
        """
        messages = self.messages[-max_messages:]

        # 找到"近期边界"：倒数第 _RECENT_TOOL_ROUNDS 个 assistant 消息的索引
        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]
        if len(assistant_indices) > _RECENT_TOOL_ROUNDS:
            recent_boundary = assistant_indices[-_RECENT_TOOL_ROUNDS]
        else:
            recent_boundary = 0  # 全部视为近期

        out: list[dict[str, Any]] = []
        for i, m in enumerate(messages):
            role = m.get("role")
            is_recent = i >= recent_boundary

            if role == "user":
                text = m.get("content", "")
                media_paths = m.get("media") or []
                user_content = (
                    _rebuild_user_content(text, media_paths) if media_paths else text
                )
                out.append({"role": "user", "content": user_content})

            elif role == "assistant":
                tool_chain: list[dict] = m.get("tool_chain") or []

                # 展开每个迭代组：assistant(tool_calls) + tool(results)
                for group in tool_chain:
                    calls: list[dict] = group.get("calls") or []
                    if not calls:
                        continue
                    out.append(
                        {
                            "role": "assistant",
                            "content": group.get("text"),  # 可能为 None
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
                    )
                    for c in calls:
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": c["call_id"],
                                "content": c["result"] if is_recent else _CLEARED,
                            }
                        )

                # 最终文本回复：若该轮没有工具链，标记为推演内容，避免被后续轮次当成事实引用
                content = m.get("content", "") or ""
                if (
                    not tool_chain
                    and content
                    and not content.startswith(_INFERENCE_TAG)
                ):
                    content = _INFERENCE_TAG + content
                if content:
                    content = _append_proactive_meta(content, m)
                out.append({"role": "assistant", "content": content})

        return out

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now()
        self.last_consolidated = 0


class SessionManager:
    # 每 N 次增量追加后触发一次全量重写，以刷新 metadata 首行（updated_at / last_consolidated）
    _METADATA_REFRESH_EVERY: int = 10

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.session_dir = workspace / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}
        # per-session 写锁，防止全量重写与增量追加在 executor 中交错
        self._write_locks: dict[str, asyncio.Lock] = {}
        # 记录各 session 自上次全量重写后的追加次数
        self._append_counts: dict[str, int] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = _safe_filename(key)
        return self.session_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session:
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = (
                            datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else None
                        )
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as e:
            logging.warning(f"Failed to load {key}: {e}")
            return None

    # ── per-session 写锁 ──────────────────────────────────────────────────────

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._write_locks:
            self._write_locks[key] = asyncio.Lock()
        return self._write_locks[key]

    # ── 同步底层实现（无锁，仅供 executor 或非 async 上下文调用）────────────────

    def _write_full(self, session: Session) -> None:
        """全量重写 JSONL，刷新 metadata 首行（last_consolidated / updated_at）。"""
        path = self._get_session_path(session.key)
        with open(path, "w") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "last_consolidated": session.last_consolidated,
                "metadata": session.metadata,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._cache[session.key] = session

    def _write_append(self, session: Session, messages: list[dict]) -> None:
        """追加消息行（不重写 metadata 首行，速度快）。"""
        path = self._get_session_path(session.key)
        with open(path, "a") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._cache[session.key] = session

    # ── 公共 API ──────────────────────────────────────────────────────────────

    def save(self, session: Session) -> None:
        """同步全量重写（兼容非 async 上下文，如 CLI / 安全重试降级）。

        不持 asyncio 锁，仅在确认无并发写操作时调用（例如启动/关闭路径）。
        """
        session.updated_at = datetime.now()
        self._append_counts[session.key] = 0
        self._write_full(session)

    async def save_async(self, session: Session) -> None:
        """异步全量重写，持有 per-session 写锁，用于 consolidation 和 proactive 写入。"""
        session.updated_at = datetime.now()
        async with self._lock(session.key):
            self._append_counts[session.key] = 0
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_full, session)

    async def append_messages(self, session: Session, messages: list[dict]) -> None:
        """增量追加消息（普通对话路径），持有 per-session 写锁。

        每 _METADATA_REFRESH_EVERY 次触发一次全量重写，保持 metadata 首行时效。
        """
        session.updated_at = datetime.now()
        msgs_copy = list(messages)
        async with self._lock(session.key):
            cnt = self._append_counts.get(session.key, 0) + 1
            loop = asyncio.get_event_loop()
            if cnt >= self._METADATA_REFRESH_EVERY:
                self._append_counts[session.key] = 0
                await loop.run_in_executor(None, self._write_full, session)
            else:
                self._append_counts[session.key] = cnt
                await loop.run_in_executor(None, self._write_append, session, msgs_copy)

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.session_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append(
                                {
                                    "key": path.stem.replace("_", ":"),
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "path": str(path),
                                }
                            )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_channel_metadata(self, channel: str) -> list[dict[str, Any]]:
        """返回指定 channel 的所有 session 的 metadata（只读首行，不加载消息）。

        返回列表元素形如：{"key": "telegram:123456", "chat_id": "123456", "metadata": {...}}
        """
        results = []
        prefix = _safe_filename(channel + ":")
        for path in self.session_dir.glob(f"{prefix}*.jsonl"):
            try:
                with open(path) as f:
                    first_line = f.readline().strip()
                if not first_line:
                    continue
                data = json.loads(first_line)
                if data.get("_type") != "metadata":
                    continue
                key = data.get("key") or path.stem.replace("_", ":", 1)
                chat_id = (
                    key.split(":", 1)[-1] if ":" in key else path.stem[len(prefix) :]
                )
                results.append(
                    {
                        "key": key,
                        "chat_id": chat_id,
                        "metadata": data.get("metadata", {}),
                    }
                )
            except Exception:
                continue
        return results
