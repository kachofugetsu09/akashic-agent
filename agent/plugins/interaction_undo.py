from __future__ import annotations

import asyncio
from typing import Protocol, TypeVar, cast

from agent.plugin_composition.interaction_undo import (
    InteractionUndoResult,
    SourceMutationFence,
)
from session.store import InteractionDeletion


class _ControlStore(Protocol):
    def latest_completed_interaction_id(self, session_key: str) -> str | None: ...

    def delete_interaction(
        self,
        control_turn_id: str,
        *,
        action_source: str,
        expected_latest_session_key: str,
    ) -> InteractionDeletion | None: ...


class _SessionManager(Protocol):
    @property
    def control_store(self) -> _ControlStore: ...

    def invalidate(self, key: str) -> None: ...


class InteractionUndoCoordinator:
    """串行删除 Session interaction，并委托可选派生插件封住一致性窗口。"""

    def __init__(self, session_manager: _SessionManager) -> None:
        self._sessions = session_manager
        self._lock = asyncio.Lock()

    async def undo_latest(
        self,
        session_key: str,
        source_fence: SourceMutationFence | None,
    ) -> InteractionUndoResult | None:
        """撤销最后一个 completed interaction，并等待临界区完整收束。"""

        task = asyncio.create_task(
            self._undo_latest_critical(session_key, source_fence),
            name=f"interaction-undo:{session_key}",
        )
        return await _finish_critical(task)

    async def _undo_latest_critical(
        self,
        session_key: str,
        source_fence: SourceMutationFence | None,
    ) -> InteractionUndoResult | None:
        """选择、删除并刷新一个 interaction 的唯一 Session truth。"""

        async with self._lock:
            # 1. 在锁内冻结当前最后一轮，避免并发撤销选择同一 owner。
            sessions = self._sessions
            store = sessions.control_store
            control_turn_id = await asyncio.to_thread(
                store.latest_completed_interaction_id,
                session_key,
            )
            if control_turn_id is None:
                return None

            # 2. 派生插件可在同一个窗口封住读取并重建自己的状态。
            def delete_source() -> InteractionDeletion | None:
                return store.delete_interaction(
                    control_turn_id,
                    action_source="plugin_undo.interaction_delete",
                    expected_latest_session_key=session_key,
                )

            if source_fence is None:
                deletion = await asyncio.to_thread(delete_source)
            else:
                raw_deletion = await source_fence(control_turn_id, delete_source)
                deletion = cast(InteractionDeletion | None, raw_deletion)
            if deletion is None:
                return None

            # 3. Session cache 只由它自己的 owner 失效。
            sessions.invalidate(deletion.session_key)
            return _public_result(deletion)


def _public_result(deletion: InteractionDeletion) -> InteractionUndoResult:
    return InteractionUndoResult(
        control_turn_id=deletion.control_turn_id,
        session_key=deletion.session_key,
        message_ids=deletion.message_ids,
        backup_path=deletion.backup_path,
        reconciliation_pending=False,
        old_last_consolidated=deletion.old_last_consolidated,
        new_last_consolidated=deletion.new_last_consolidated,
    )


T = TypeVar("T")


async def _finish_critical(task: asyncio.Task[T]) -> T:
    """等待临界任务完成，再把 caller cancellation 原样恢复。"""

    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancelled is None:
                cancelled = exc
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result


__all__ = ["InteractionUndoCoordinator"]
