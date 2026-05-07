from __future__ import annotations

import logging
from typing import Any, cast

from agent.lifecycle.types import BeforeTurnCtx
from agent.plugins import Plugin

logger = logging.getLogger("plugin.undo")

_SESSION_SLOT = "session:session"
_CTX_SLOT = "session:ctx"


class UndoCommandModule:
    requires = (_SESSION_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, plugin: "PluginUndo") -> None:
        self._plugin = plugin

    async def run(self, frame) -> object:
        if _CTX_SLOT in frame.slots:
            return frame
        state = frame.input
        if _normalize_command(state.msg.content) != "/undo":
            return frame
        reply = await self._plugin.undo(state.session_key)
        frame.slots[_CTX_SLOT] = _abort_ctx(state, reply)
        return frame


class PluginUndo(Plugin):
    name = "plugin_undo"

    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        return [("undo", "撤销上一轮对话")]

    def before_turn_modules_early(self) -> list[object]:
        return [UndoCommandModule(self)]

    async def undo(self, session_key: str) -> str:
        session_manager = getattr(self.context, "session_manager", None)
        if session_manager is None:
            return "撤销失败：session 管理器不可用。"
        memory_result: dict[str, object] = {
            "affected_ids": [],
            "restored_ids": [],
            "rollback_source_ids": [],
        }
        message_ids_for_memory: list[str] = []

        def resolve_sources(message_ids: list[str]) -> list[str]:
            nonlocal memory_result, message_ids_for_memory
            message_ids_for_memory = list(message_ids)
            memory_result = _undo_memory_sources(
                getattr(self.context, "memory_engine", None),
                message_ids,
                dry_run=True,
            )
            return _string_list(memory_result.get("rollback_source_ids"))

        result = await session_manager.undo_last_turn(
            session_key,
            rollback_source_resolver=resolve_sources,
        )
        if result is None:
            return "没有可撤销的上一轮对话。"
        memory_result = _undo_memory_sources(
            getattr(self.context, "memory_engine", None),
            message_ids_for_memory or result.deleted_ids,
            dry_run=False,
        )
        logger.info(
            "undo session=%s deleted=%d memory_superseded=%d memory_restored=%d last=%d->%d",
            session_key,
            len(result.deleted_ids),
            len(_string_list(memory_result.get("affected_ids"))),
            len(_string_list(memory_result.get("restored_ids"))),
            result.last_consolidated_before,
            result.last_consolidated_after,
        )
        return (
            "已撤销上一轮对话。"
            f"\n删除消息：{len(result.deleted_ids)} 条"
            f"\n失效记忆：{len(_string_list(memory_result.get('affected_ids')))} 条"
            f"\n恢复旧记忆：{len(_string_list(memory_result.get('restored_ids')))} 条"
        )


def _undo_memory_sources(
    memory_engine: Any,
    message_ids: list[str],
    *,
    dry_run: bool,
) -> dict[str, object]:
    if memory_engine is None:
        return {"affected_ids": [], "restored_ids": [], "rollback_source_ids": []}
    undo = getattr(memory_engine, "undo_by_message_sources", None)
    if not callable(undo):
        return {"affected_ids": [], "restored_ids": [], "rollback_source_ids": []}
    result = undo(message_ids, dry_run=dry_run)
    return cast(dict[str, object], result if isinstance(result, dict) else {})


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_command(content: str) -> str:
    parts = (content or "").strip().split(maxsplit=1)
    if not parts:
        return ""
    head = parts[0].lower()
    if "@" in head:
        head = head.split("@", 1)[0]
    return head


def _abort_ctx(state, reply: str) -> BeforeTurnCtx:
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
