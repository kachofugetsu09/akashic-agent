from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from agent.lifecycle.types import BeforeTurnCtx
from agent.plugins import Plugin
from agent.prompting import is_context_frame

logger = logging.getLogger("plugin.undo")

_SESSION_SLOT = "session:session"
_CTX_SLOT = "session:ctx"


@dataclass
class _UndoSessionResult:
    deleted_ids: list[str]
    target_user_id: str
    target_assistant_id: str
    rollback_index: int
    last_consolidated_before: int
    last_consolidated_after: int


class UndoCommandModule:
    slot = "plugin_undo.undo"
    requires = ("before_turn.acquire_session", _SESSION_SLOT)
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

    def before_turn_modules(self) -> list[object]:
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

        result = await _undo_last_turn(
            session_manager,
            session_key,
            rollback_source_resolver=resolve_sources,
        )
        if result is None:
            return "没有可撤销的上一轮对话。"
        try:
            memory_result = _undo_memory_sources(
                getattr(self.context, "memory_engine", None),
                message_ids_for_memory or result.deleted_ids,
                dry_run=False,
            )
        except Exception:
            logger.exception(
                "undo memory cleanup failed after session delete: session=%s deleted_ids=%s dry_run=%s",
                session_key,
                result.deleted_ids,
                memory_result,
            )
            return (
                "已撤销上一轮对话，但记忆清理失败。"
                f"\n删除消息：{len(result.deleted_ids)} 条"
                "\n请查看日志后手动清理对应记忆。"
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


async def _undo_last_turn(
    session_manager: Any,
    session_key: str,
    *,
    rollback_source_ids: list[str] | None = None,
    expected_message_ids: list[str] | None = None,
    rollback_source_resolver: Any = None,
) -> _UndoSessionResult | None:
    async with session_manager._lock(session_key):
        session = session_manager.get_or_create(session_key)
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
        delete_set = set(delete_indices)
        remaining = [
            msg for i, msg in enumerate(session.messages) if i not in delete_set
        ]
        deleted_before = sum(1 for i in delete_indices if i < rollback_index)
        new_last = max(0, rollback_index - deleted_before)
        new_last = min(new_last, len(remaining))
        deleted_count = session_manager._store.delete_session_messages_and_update_cursor(
            session.key,
            ids=deleted_ids,
            last_consolidated=new_last,
        )
        if deleted_count != len(deleted_ids):
            session_manager.invalidate(session.key)
            return None
        session.messages = remaining
        session.last_consolidated = new_last
        session.updated_at = datetime.now()
        session_manager._cache[session.key] = session
        return _UndoSessionResult(
            deleted_ids=deleted_ids,
            target_user_id=target_user_id,
            target_assistant_id=target_assistant_id,
            rollback_index=rollback_index,
            last_consolidated_before=old_last,
            last_consolidated_after=new_last,
        )


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
