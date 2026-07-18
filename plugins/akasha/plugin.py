from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from agent.lifecycle.types import BeforeTurnCtx, TurnState
from agent.plugins import MobileUiContribution, Plugin
from agent.plugins.mobile_ui import MobileUiRpcInvalidRequest
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
    @classmethod
    def dashboard_module(cls) -> str | None:
        return "dashboard.py"

    @classmethod
    def mobile_ui(cls) -> MobileUiContribution:
        return MobileUiContribution(
            module="mobile_ui.js",
            stylesheet="mobile_ui.css",
            slots=("turn.before_reasoning",),
        )

    name = "akasha"

    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        if not self.is_active():
            return []
        return [("akashalast", "查看上一轮 Akasha 检索诊断")]

    def mobile_bot_commands(self) -> list[tuple[str, str]]:
        if not self.is_active():
            return []
        return [("akashalast", "查看上一轮 Akasha 检索诊断")]

    def before_turn_modules(self) -> list[object]:
        return [AkashaLastCommandModule(self)]

    def is_active(self) -> bool:
        return _is_memory_engine(getattr(self.context, "memory_engine", None), "akasha")

    def render_last_query(self, session_key: str) -> str:
        workspace = self.context.workspace
        if workspace is None:
            return "Akasha 诊断不可用：workspace 不存在。"
        data_dir = self.context.data_dir
        if data_dir is None:
            return "Akasha 诊断不可用：插件数据目录不存在。"
        store, _workspace = self._open_mobile_store()
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

    def mobile_ui_query(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """返回 mobile 消息召回或最近检索检查数据。"""

        # 1. 消息内召回严格绑定当前 session 和 turn。
        if method == "recall.current":
            if set(payload) - {"message_id"}:
                raise MobileUiRpcInvalidRequest("Akasha recall.current 参数无效")
            if session_id is None:
                raise MobileUiRpcInvalidRequest("Akasha recall.current 需要 session_id")
            message_id = payload.get("message_id")
            if message_id is not None and not isinstance(message_id, str):
                raise MobileUiRpcInvalidRequest("Akasha recall.current 的 message_id 必须是字符串")
            if message_id is not None and message_id.startswith("assistant:"):
                if turn_id is None or message_id != f"assistant:{turn_id}":
                    return {"left": [], "right": []}
                historical_message_id = None
            else:
                historical_message_id = message_id
            return self._load_mobile_recall(session_id, historical_message_id)

        # 2. 看板只读取既有诊断日志，不改变 Akasha 状态。
        if method == "inspector.recent":
            if payload:
                raise MobileUiRpcInvalidRequest("Akasha inspector.recent 不接受参数")
            return self._load_mobile_inspections()
        if method == "inspector.detail":
            if set(payload) != {"query_id"}:
                raise MobileUiRpcInvalidRequest("Akasha inspector.detail 需要 query_id")
            query_id = payload["query_id"]
            if not isinstance(query_id, str) or not query_id.strip():
                raise MobileUiRpcInvalidRequest("Akasha inspector.detail 的 query_id 必须是非空字符串")
            return self._load_mobile_inspection(query_id.strip())
        raise MobileUiRpcInvalidRequest(f"Akasha mobile UI 方法无效: {method}")

    def _load_mobile_recall(
        self,
        session_id: str,
        message_id: str | None,
    ) -> dict[str, object]:
        """在线程中读取指定 assistant 消息所属轮次的召回。"""

        # 1. message_id 属于 RPC 输入，必须先于数据库边界完成归属校验。
        before_seq = (
            _assistant_message_seq(session_id, message_id)
            if message_id is not None
            else None
        )

        # 2. 校验完成后才打开只读 store 并读取对应 context 记录。
        store, workspace = self._open_mobile_store()
        try:
            raw = store.get_latest_context_query_log(
                session_id,
                before_seq=before_seq,
            )
        finally:
            store.close()
        if raw is None:
            return {"left": [], "right": []}
        dense_items = _json_items(raw.get("dense_items_json"))
        ripple_items = _json_items(raw.get("ripple_items_json"))
        message_times = _load_message_times(
            workspace / "sessions.db",
            [*dense_items, *ripple_items],
        )
        return {
            "left": _mobile_recall_items(dense_items, message_times),
            "right": _mobile_recall_items(ripple_items, message_times),
        }

    def _load_mobile_inspections(self) -> dict[str, object]:
        """读取最近检索的移动端轻量摘要。"""

        # 1. 看板固定读取最近 30 轮，避免把桌面诊断全集搬上手机。
        store, _workspace = self._open_mobile_store()
        try:
            rows, total = store.list_query_logs(page=1, page_size=30)
        finally:
            store.close()
        return {
            "items": [_mobile_inspection_summary(row) for row in rows],
            "total": total,
        }

    def _load_mobile_inspection(self, query_id: str) -> dict[str, object]:
        """读取一轮检索实际注入的左右脑记忆。"""

        # 1. 完整诊断只在用户展开对应轮次时按需读取。
        store, workspace = self._open_mobile_store()
        try:
            raw = store.get_query_log(query_id)
        finally:
            store.close()
        if raw is None:
            raise MobileUiRpcInvalidRequest("Akasha 检索记录不存在")

        # 2. 复用消息内召回的排序和时间补全语义。
        dense_items = _json_items(raw.get("dense_items_json"))
        ripple_items = _json_items(raw.get("ripple_items_json"))
        message_times = _load_message_times(
            workspace / "sessions.db",
            [*dense_items, *ripple_items],
        )
        return {
            **_mobile_inspection_summary(raw),
            "query_text": str(raw.get("query_text") or ""),
            "left": _mobile_recall_items(dense_items, message_times),
            "right": _mobile_recall_items(ripple_items, message_times),
        }

    def _open_mobile_store(self) -> tuple[AkashaStore, Path]:
        """解析当前插件数据目录并打开 Akasha store。"""

        workspace = self.context.workspace
        if workspace is None:
            raise RuntimeError("Akasha workspace 不存在")
        data_dir = self.context.data_dir
        if data_dir is None:
            raise RuntimeError("Akasha 插件数据目录不存在")
        return (
            AkashaStore(
                resolve_akasha_db_path(
                    workspace=workspace,
                    akasha_config=load_akasha_config(plugin_dir=data_dir),
                ),
                read_only=True,
            ),
            workspace,
        )


def _assistant_message_seq(session_id: str, message_id: str) -> int:
    prefix = f"{session_id}:"
    seq_text = message_id.removeprefix(prefix)
    if not message_id.startswith(prefix) or not seq_text.isdigit():
        raise MobileUiRpcInvalidRequest("Akasha message_id 不属于当前 session")
    return int(seq_text)


def _mobile_inspection_summary(raw: dict[str, object]) -> dict[str, object]:
    return {
        "query_id": str(raw.get("query_id") or ""),
        "query_preview": _clip(str(raw.get("query_text") or ""), 180),
        "ts": str(raw.get("ts") or ""),
        "left_count": int(_float(raw.get("dense_count"))),
        "right_count": int(_float(raw.get("ripple_count"))),
        "inject_chars": int(_float(raw.get("inject_chars"))),
    }


def _load_message_times(
    sessions_db_path: Path,
    items: list[dict[str, object]],
) -> dict[str, str]:
    """为旧诊断日志从原消息库补回发生时间。"""

    # 1. 新日志已携带 happened_at，只回源旧日志缺失的消息。
    keys = {
        str(item.get("key") or "")
        for item in items
        if not str(item.get("happened_at") or "").strip()
        and str(item.get("key") or "").strip()
    }
    if not keys:
        return {}

    # 2. sessions.db 是消息时间的拥有层；只读查询不得修改线上库。
    path = str(sessions_db_path)
    placeholders = ",".join("?" for _ in keys)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        rows = db.execute(
            f"SELECT id, ts FROM messages WHERE id IN ({placeholders})",
            tuple(keys),
        ).fetchall()
    return {str(row[0]): str(row[1] or "") for row in rows}


def _mobile_recall_items(
    items: list[dict[str, object]],
    message_times: dict[str, str],
) -> list[dict[str, object]]:
    ranked: list[tuple[float, dict[str, object]]] = [
        (
            _recall_timestamp(
                str(item.get("happened_at") or "")
                or message_times.get(str(item.get("key") or ""), "")
            ),
            {
                "summary": _clip(
                    str(item.get("user_message") or item.get("summary") or ""),
                    120,
                ),
                "preview": _clip(str(item.get("assistant_preview") or ""), 100),
                "score": round(_float(item.get("score")), 3),
            },
        )
        for item in items
    ]
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in ranked]


def _recall_timestamp(value: str) -> float:
    if not value:
        return float("-inf")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("-inf")


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
    if not isinstance(value, str):
        raise ValueError("Akasha 检索诊断缺少 JSON 数组")
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Akasha 检索诊断 JSON 损坏") from exc
    if not isinstance(loaded, list):
        raise ValueError("Akasha 检索诊断字段必须是 JSON 数组")
    items: list[dict[str, object]] = []
    for index, item in enumerate(cast(list[object], loaded)):
        if not isinstance(item, dict):
            raise ValueError(f"Akasha 检索诊断字段[{index}]必须是 JSON 对象")
        items.append(cast(dict[str, object], item))
    return items


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
