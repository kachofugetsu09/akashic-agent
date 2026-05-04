from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from agent.lifecycle.types import BeforeTurnCtx, TurnState
from agent.plugins import Plugin
from agent.prompting import is_context_frame

logger = logging.getLogger("plugin.status_commands")

_SESSION_SLOT = "session:session"
_CTX_SLOT = "session:ctx"
_TS_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})")
_BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class MemoryStatusCommandModule:
    requires = (_SESSION_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, plugin_name: str) -> None:
        self._plugin_name = plugin_name

    async def run(self, frame) -> object:
        if _CTX_SLOT in frame.slots:
            return frame
        state = frame.input
        command = _normalize_command(state.msg.content)
        if command not in {
            "/memorystatus",
            "/memory_status",
            "/compact_status",
        }:
            return frame
        session = state.session
        if session is None:
            return frame
        messages = list(getattr(session, "messages", []))
        last = max(0, int(getattr(session, "last_consolidated", 0)))
        last = min(last, len(messages))
        logger.info(
            "[%s:%s] 命中命令: %s",
            self._plugin_name,
            self.__class__.__name__,
            command,
        )
        frame.slots[_CTX_SLOT] = _abort_ctx(
            state, _format_memory_status_reply(messages, last)
        )
        return frame


class KVCacheCommandModule:
    requires = (_SESSION_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, plugin_name: str, db_path: Path | None) -> None:
        self._plugin_name = plugin_name
        self._db_path = db_path

    async def run(self, frame) -> object:
        if _CTX_SLOT in frame.slots:
            return frame
        state = frame.input
        command = _normalize_command(state.msg.content)
        if command not in {"/kvcache", "/cache_status"}:
            return frame
        logger.info(
            "[%s:%s] 命中命令: %s",
            self._plugin_name,
            self.__class__.__name__,
            command,
        )
        reply = self._build_reply(state)
        frame.slots[_CTX_SLOT] = _abort_ctx(state, reply)
        return frame

    def _build_reply(self, state: TurnState) -> str:
        db_path = self._db_path
        if not db_path or not db_path.exists():
            return "暂无 KVCache 数据（observe 数据库不存在）。"

        args = (state.msg.content or "").strip().split()
        limit = 5
        if len(args) > 1:
            try:
                limit = max(1, min(30, int(args[1])))
            except ValueError:
                pass

        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.execute(
                    """SELECT llm_output, ts, react_cache_prompt_tokens, react_cache_hit_tokens
                       FROM turns WHERE session_key=? AND source='agent'
                       ORDER BY id DESC LIMIT ?""",
                    [state.session_key, limit],
                )
                rows = cursor.fetchall()
            finally:
                conn.close()
        except Exception:
            logger.exception("KVCache 查询失败")
            return "KVCache 查询失败。"

        if not rows:
            return "暂无 KVCache 数据。"

        overall_prompt = sum(r[2] or 0 for r in rows)
        overall_hit = sum(r[3] or 0 for r in rows)
        overall_pct = (overall_hit / overall_prompt * 100) if overall_prompt > 0 else 0.0

        lines = [
            f"⚡ KVCache · 最近 {len(rows)} 轮",
            "",
            f"命中率  {overall_pct:.1f}%  {_pct_bar(overall_pct)}",
            f"Token  {overall_hit:,} / {overall_prompt:,}",
        ]
        for row in rows:
            llm_output, ts, prompt_tokens, hit_tokens = row
            content = _content_to_text(llm_output or "")
            if is_context_frame(content):
                content = ""
            preview = _preview_text(content, limit=72)
            hit = hit_tokens or 0
            prompt = prompt_tokens or 0
            pct = (hit / prompt * 100) if prompt > 0 else 0.0
            lines.extend(["", ""])
            lines.append(
                f"{_format_ts(ts)}   {_pct_emoji(pct)} {pct:.1f}%  {_pct_bar(pct)}"
            )
            lines.append(f"    {hit:,} / {prompt:,} tokens")
            if preview:
                lines.append(f"    {preview}")
        return "\n".join(lines)


class StatusCommands(Plugin):
    name = "status_commands"

    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        return [
            ("memorystatus", "查看记忆整理状态"),
            ("kvcache", "查看 KVCache 状态"),
        ]

    def before_turn_modules_early(self) -> list[object]:
        plugin_name = self.name or "status_commands"
        db_path = None
        if self.context.workspace is not None:
            db_path = self.context.workspace / "observe" / "observe.db"
        return cast(
            "list[object]",
            [
                MemoryStatusCommandModule(plugin_name),
                KVCacheCommandModule(plugin_name, db_path),
            ],
        )


def _normalize_command(content: str) -> str:
    parts = (content or "").strip().split(maxsplit=1)
    if not parts:
        return ""
    head = parts[0].lower()
    if "@" in head:
        head = head.split("@", 1)[0]
    return head


def _abort_ctx(state: TurnState, reply: str) -> BeforeTurnCtx:
    return BeforeTurnCtx(
        session_key=state.session_key,
        channel=state.msg.channel,
        chat_id=state.msg.chat_id,
        content=state.msg.content,
        timestamp=state.msg.timestamp,
        skill_names=[],
        retrieved_memory_block="",
        retrieval_trace_raw=None,
        history_messages=(),
        abort=True,
        abort_reply=reply,
    )


def _format_ts(ts: str) -> str:
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_BEIJING_TZ)
        return f"{parsed.month}-{parsed.day} {parsed.hour:02d}:{parsed.minute:02d}"
    except ValueError:
        pass
    m = _TS_PATTERN.search(ts)
    if m:
        return f"{int(m.group(2))}-{int(m.group(3))} {m.group(4)}:{m.group(5)}"
    return ts


def _format_memory_status_reply(messages: list[dict], last_consolidated: int) -> str:
    consolidated_user = _count_real_user_messages(messages[:last_consolidated])
    total_user = _count_real_user_messages(messages)
    pending_user = max(0, total_user - consolidated_user)
    last_user_message = _latest_real_user_content(messages[:last_consolidated])

    lines = ["🧠 记忆整理状态："]
    if last_consolidated <= 0 or not last_user_message:
        lines.append("当前会话还没有完成过记忆整理。")
    elif pending_user == 0:
        lines.append("当前会话已经整理到最新的用户消息。")
    else:
        lines.append(f"上次整理到 {pending_user} 条用户消息之前。")
    if last_user_message:
        lines.extend(["", "最后已整理的用户消息：", f"“{_preview_text(last_user_message)}”"])
    lines.extend(
        [
            "",
            f"尚未整理的用户消息数：{pending_user}",
            f"当前会话消息数：{len(messages)}",
        ]
    )
    return "\n".join(lines)


def _count_real_user_messages(messages: list[dict]) -> int:
    return sum(1 for item in messages if _is_real_user_message(item))


def _latest_real_user_content(messages: list[dict]) -> str:
    for item in reversed(messages):
        if _is_real_user_message(item):
            return _content_to_text(item.get("content", ""))
    return ""


def _is_real_user_message(item: dict) -> bool:
    if item.get("role") != "user":
        return False
    content = _content_to_text(item.get("content", ""))
    return bool(content) and not is_context_frame(content)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _preview_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _pct_bar(pct: float, width: int = 10) -> str:
    filled = round(pct / 100 * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _pct_emoji(pct: float) -> str:
    if pct >= 80:
        return "🟢"
    if pct >= 40:
        return "🟡"
    return "🔴"
