from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from agent.lifecycle.types import BeforeTurnCtx, TurnState
from agent.plugins import Plugin
from plugins.akasha.config import load_akasha_config, resolve_akasha_db_path
from plugins.akasha.store import AkashaStore

_CTX_SLOT = "session:ctx"
_BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class AkashaLastCommandModule:
    slot = "akasha.last_query"
    requires = ("before_turn.acquire_session", "session:session")
    produces = (_CTX_SLOT,)

    def __init__(self, plugin: "AkashaPlugin") -> None:
        self._plugin = plugin

    async def run(self, frame: Any) -> Any:
        if _CTX_SLOT in frame.slots:
            return frame
        state = cast(Any, frame.input)
        command = _normalize_command(state.msg.content)
        if command not in {"/akashalast", "/akasha_last"}:
            return frame
        if not self._plugin.is_active():
            return frame
        frame.slots[_CTX_SLOT] = _abort_ctx(
            state,
            self._plugin.render_last_query(state.session_key),
        )
        return frame


class AkashaPlugin(Plugin):
    name = "akasha"

    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        if not self.is_active():
            return []
        return [("akashalast", "查看上一轮 Akasha 检索诊断")]

    def before_turn_modules(self) -> list[object]:
        if not self.is_active():
            return []
        return [AkashaLastCommandModule(self)]

    def is_active(self) -> bool:
        return _is_memory_engine(getattr(self.context, "memory_engine", None), "akasha")

    def render_last_query(self, session_key: str) -> str:
        workspace = self.context.workspace
        if workspace is None:
            return "Akasha 诊断不可用：workspace 不存在。"
        store = AkashaStore(
            resolve_akasha_db_path(
                workspace=workspace,
                akasha_config=load_akasha_config(plugin_dir=Path(__file__).resolve().parent),
            )
        )
        try:
            rows, _ = store.list_query_logs(
                session_key=session_key,
                page=1,
                page_size=1,
            )
            if not rows:
                return "暂无 Akasha 检索诊断记录。"
            query_id = str(rows[0]["query_id"])
            raw = store.get_query_log(query_id)
        finally:
            store.close()
        if raw is None:
            return "暂无 Akasha 检索诊断记录。"
        return _render_query_detail(raw)


def _render_query_detail(raw: dict[str, object]) -> str:
    activation_items = _json_items(raw.get("activation_items_json"))
    dense_items = _json_items(raw.get("dense_items_json"))
    ripple_items = _json_items(raw.get("ripple_items_json"))
    threshold = _float(raw.get("activation_threshold"))
    lines = [
        "🧠 Akasha 记忆检索诊断",
        f"📍 会话: `{raw.get('session_key')}` | seq `{raw.get('seq')}`",
        f"❓ 提问: {_clip(str(raw.get('query_text') or ''), 60)}",
        f"🏷️ 意图: `{raw.get('intent')}` | 🕒 `{_format_ts(str(raw.get('ts') or ''))}`",
        "",
        "⚡ 图扩散状态 (Activation):",
        f"• 种子节点 (Seeds): `{raw.get('seed_count')}` 个",
        f"• 扩散范围 (Pool): `{raw.get('pool_count')}` 个",
        (
            f"• 实际激活 (Activated): `{raw.get('activated_count')}` 个"
            f" | 门槛: `{threshold:.3f}`"
        ),
    ]
    lines.extend(_render_activated_nodes(
        activation_items,
        threshold=threshold,
        limit=8,
    ))
    lines.extend(_render_memory_items(
        "🎯 左脑精确回忆 (Dense):",
        "(最终注入大模型的左脑候选)",
        dense_items,
        show_signals=False,
        show_path=False,
        score_label="得",
        limit=8,
    ))
    lines.extend(_render_memory_items(
        "🌊 右脑联想记忆 (Ripple):",
        "(最终注入大模型的右脑候选)",
        ripple_items,
        show_signals=True,
        show_path=True,
        score_label="得",
        limit=8,
    ))
    return "\n".join(lines).strip()


def _render_activated_nodes(
    items: list[dict[str, object]],
    *,
    threshold: float,
    limit: int,
) -> list[str]:
    lines = [
        "",
        "──────",
        "🔥 本轮图激活节点 (Activated Nodes):",
        f"(得分配分超过 `{threshold:.3f}`，执行状态更新并与本轮新节点建边的节点)",
    ]
    if not items:
        lines.append("无")
        return lines
    for index, item in enumerate(items[:limit], start=1):
        lines.extend(_render_item(
            index,
            item,
            inline=False,
            score_label="分",
            show_path=True,
            show_signals=False,
        ))
    if len(items) > limit:
        lines.append(f"(后略，还有 `{len(items) - limit}` 条)")
    return lines


def _render_memory_items(
    title: str,
    subtitle: str,
    items: list[dict[str, object]],
    *,
    show_signals: bool,
    show_path: bool,
    score_label: str,
    limit: int,
) -> list[str]:
    lines = ["", "──────", title, subtitle]
    if not items:
        lines.append("无")
        return lines
    for index, item in enumerate(items[:limit], start=1):
        lines.extend(_render_item(
            index,
            item,
            inline=True,
            score_label=score_label,
            show_path=show_path,
            show_signals=show_signals,
        ))
    if len(items) > limit:
        lines.append(f"(后略，还有 `{len(items) - limit}` 条)")
    return lines


def _render_item(
    index: int,
    item: dict[str, object],
    *,
    inline: bool,
    score_label: str,
    show_path: bool,
    show_signals: bool,
) -> list[str]:
    user_text = _clip(str(item.get("user_message") or item.get("summary") or ""), 32)
    assistant = _clip(str(item.get("assistant_preview") or ""), 24)
    score = _float(item.get("score"))
    source = str(item.get("source") or item.get("lane") or "")
    path = str(item.get("path_type") or "")
    lines: list[str] = []
    meta: list[str] = [f"{score_label}: `{score:.3f}`"]
    if source:
        meta.append(f"源: `{source}`")
    if show_path and path:
        meta.append(f"径: `{path}`")
    if inline:
        text = f"{_rank_label(index)} U: {user_text}"
        if assistant:
            text += f" ➔ A: {assistant}"
        text += " | " + " | ".join(meta)
        lines.append(text)
    else:
        lines.append(f"{_rank_label(index)} U: {user_text}")
        if assistant:
            lines.append(f"   A: {assistant}")
        lines.append(" | ".join(meta))
    if show_signals:
        lines.append(
            "因: "
            f"`dir:{_float(item.get('direct')):.2f} "
            f"st:{_float(item.get('state')):.2f} "
            f"edg:{_float(item.get('edge')):.2f} "
            f"res:{_float(item.get('resource')):.2f} "
            f"fan:{int(_float(item.get('fan')))}"
            "`"
        )
    return lines


def _rank_label(index: int) -> str:
    labels = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
    if 1 <= index <= len(labels):
        return labels[index - 1]
    return f"{index}."


def _json_items(value: object) -> list[dict[str, object]]:
    try:
        loaded = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [
        cast(dict[str, object], item)
        for item in cast(list[object], loaded)
        if isinstance(item, dict)
    ]


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


def _is_memory_engine(engine: object, name: str) -> bool:
    describe = getattr(engine, "describe", None)
    if not callable(describe):
        return False
    description = cast(Any, describe())
    return str(getattr(description, "name", "")) == name


def _format_ts(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_BEIJING_TZ)
        return f"{parsed.month}-{parsed.day} {parsed.hour:02d}:{parsed.minute:02d}"
    except ValueError:
        return value


def _clip(text: str, limit: int) -> str:
    clean = " ".join(text.split()).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _float(value: object) -> float:
    try:
        return float(cast(Any, value) or 0.0)
    except (TypeError, ValueError):
        return 0.0
