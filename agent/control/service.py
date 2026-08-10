from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
import secrets
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from agent.control.errors import (
    PluginManagementError,
    ThreadBusyError,
    ThreadNotFoundError,
)
from agent.control.ids import new_thread_id
from agent.control.models import ThreadRecord, ThreadSource, TurnRequest
from agent.control.protocol.models import InitializeParams
from agent.control.protocol.errors import JsonRpcError, UNAUTHORIZED
from agent.control.runtime import ConversationRuntime, TurnHandle
from agent.restart import RestartCoordinator
from session.manager import SessionManager
from session.memory_policy import validate_session_memory_metadata

PluginInstall = Callable[..., Awaitable[dict[str, object]]]
PluginAction = Callable[[str], Awaitable[dict[str, object]]]
PluginStatus = Callable[[], dict[str, object]]
PluginUninstall = Callable[[str, str], Awaitable[dict[str, object]]]
PluginChildBinding = Callable[[str, bool], dict[str, str] | None]
PluginTurnBarrier = Callable[[], Awaitable[None]]
RuntimeSelector = Literal["stable", "latest"]
logger = logging.getLogger(__name__)


def _require_runtime_selector(value: object) -> RuntimeSelector:
    if value not in {"stable", "latest"}:
        raise ValueError("runtime 必须是 stable 或 latest")
    return cast(RuntimeSelector, value)


class ControlService:
    """把协议方法投影到唯一 ConversationRuntime 和 SessionManager。"""

    def __init__(
        self,
        runtime: ConversationRuntime,
        sessions: SessionManager,
        workspace: Path,
        *,
        plugin_drain: Callable[[str], Awaitable[str]] | None = None,
        plugin_uninstall: Callable[[str], Awaitable[dict[str, object]]] | None = None,
        plugin_uninstall_register: PluginUninstall | None = None,
        plugin_install: PluginInstall | None = None,
        plugin_revert: PluginAction | None = None,
        plugin_child_binding: PluginChildBinding | None = None,
        plugin_turn_barrier: PluginTurnBarrier | None = None,
        plugin_status: PluginStatus | None = None,
        plugin_promote: PluginAction | None = None,
        plugin_discard: PluginAction | None = None,
        workspace_token: str | None = None,
        restart_coordinator: RestartCoordinator | None = None,
        boot_id: str | None = None,
        ready: Callable[[], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.sessions = sessions
        self.workspace = workspace.resolve()
        self._plugin_drain = plugin_drain
        self._plugin_uninstall = plugin_uninstall
        self._plugin_uninstall_register = plugin_uninstall_register
        self._plugin_install = plugin_install
        self._plugin_revert = plugin_revert
        self._plugin_child_binding = plugin_child_binding
        self._plugin_turn_barrier = plugin_turn_barrier
        self._plugin_status = plugin_status
        self._plugin_promote = plugin_promote
        self._plugin_discard = plugin_discard
        self._workspace_token = workspace_token
        self._restart_coordinator = restart_coordinator
        self._boot_id = boot_id
        self._ready = ready
        self._operation_tasks: set[asyncio.Task[dict[str, object]]] = set()

    def initialize(self, params: InitializeParams) -> dict[str, object]:
        if self._workspace_token is not None and not secrets.compare_digest(
            params.workspaceToken or "",
            self._workspace_token,
        ):
            raise JsonRpcError(UNAUTHORIZED, "Invalid workspace token")
        return {
            "protocolVersion": "1.0",
            "serverInfo": {"name": "akashic-agent", "version": "0.1.0"},
            "workspace": str(self.workspace),
            "capabilities": {
                "reasoningEvents": False,
                "turnInterrupt": True,
                "turnSteer": False,
            },
        }

    def status(self) -> dict[str, object]:
        return {
            "ready": self._ready() if self._ready is not None else True,
            "bootId": self._boot_id,
            "workspace": str(self.workspace),
            "protocolVersion": "1.0",
        }

    def notify_turn_delivered(self, turn_id: str) -> None:
        if self._restart_coordinator is not None:
            self._restart_coordinator.mark_delivered(turn_id)

    def notify_turn_delivery_failed(self, turn_id: str, reason: str) -> None:
        if self._restart_coordinator is not None:
            self._restart_coordinator.mark_delivery_failed(turn_id, reason)

    def start_thread(
        self,
        metadata: dict[str, Any],
        runtime: str = "stable",
    ) -> dict[str, object]:
        # 1. 外部输入边界校验：非 boolean 的 skip_post_memory 拒绝创建 session。
        stored_metadata = dict(metadata)
        validate_session_memory_metadata(stored_metadata)
        _require_runtime_selector(runtime)
        if "runtime" in stored_metadata:
            raise ValueError("thread metadata 的 runtime 为协议保留字段")
        if runtime != "stable":
            stored_metadata["runtime"] = runtime
        thread_id = new_thread_id()
        session = self.sessions.get_or_create(thread_id)
        session.metadata.update(stored_metadata)
        self.sessions.save(session)
        return self._thread_record(thread_id).to_dict()

    def resume_thread(self, thread_id: str) -> dict[str, object]:
        return self._thread_record(thread_id).to_dict()

    def list_threads(self, cursor: str | None, limit: int) -> dict[str, object]:
        rows = self.sessions.list_sessions()
        start = 0
        if cursor is not None:
            matching = [index for index, row in enumerate(rows) if row["key"] == cursor]
            if not matching:
                raise ThreadNotFoundError(f"thread cursor 不存在: {cursor}")
            start = matching[0] + 1
        page = rows[start : start + limit]
        threads = [self._thread_record(str(row["key"])).to_dict() for row in page]
        next_cursor = (
            str(page[-1]["key"]) if start + limit < len(rows) and page else None
        )
        return {"data": threads, "nextCursor": next_cursor}

    def read_thread(self, thread_id: str, include_turns: bool) -> dict[str, object]:
        payload = self._thread_record(thread_id).to_dict()
        if include_turns:
            payload["turns"] = [
                turn.to_dict()
                for turn in self.sessions.control_store.list_turns(thread_id, limit=200)
            ]
        return payload

    def delete_thread(self, thread_id: str) -> dict[str, object]:
        if self.runtime.is_thread_active(thread_id):
            raise ThreadBusyError(f"thread 正在执行: {thread_id}")
        if not self.sessions.delete_session(thread_id):
            raise ThreadNotFoundError(f"thread 不存在: {thread_id}")
        return {"id": thread_id, "deleted": True}

    async def start_turn(
        self,
        thread_id: str,
        input_text: str,
        metadata: dict[str, Any],
        runtime: str | None = None,
        attached: bool = True,
    ) -> TurnHandle:
        if self._plugin_turn_barrier is not None:
            await self._plugin_turn_barrier()
        session_meta = self.sessions.control_store.get_session_meta(thread_id)
        if session_meta is None:
            raise ThreadNotFoundError(f"thread 不存在: {thread_id}")
        turn_metadata = dict(metadata)
        if "runtime" in turn_metadata:
            raise ValueError("turn metadata 的 runtime 为协议保留字段")
        session_metadata = session_meta["metadata"]
        assert isinstance(session_metadata, dict)
        selected = _require_runtime_selector(
            runtime
            if runtime is not None
            else session_metadata.get("runtime", "stable")
        )
        owner_turn_id = str(session_metadata.get("_pluginRolloutOwnerTurnId") or "")
        if owner_turn_id:
            if self._plugin_child_binding is None:
                raise RuntimeError("当前 runtime 不支持插件候选因果绑定")
            binding = self._plugin_child_binding(owner_turn_id, attached)
            if binding is not None:
                selected = "latest"
                turn_metadata.update(
                    {
                        "_pluginRolloutOwnerTurnId": binding["ownerTurnId"],
                        "_pluginRolloutPluginId": binding["pluginId"],
                        "_pluginRolloutGenerationId": binding["generationId"],
                        "_pluginRolloutSourceRevision": binding["sourceRevision"],
                    }
                )
        turn_metadata["runtime"] = selected
        return await self.runtime.start_turn(
            TurnRequest(thread_id, input_text, turn_metadata)
        )

    def read_turn(self, thread_id: str, turn_id: str) -> dict[str, object]:
        return self.runtime.read_turn(thread_id, turn_id).to_dict()

    async def interrupt_turn(self, thread_id: str, turn_id: str) -> dict[str, object]:
        return (await self.runtime.interrupt_turn(thread_id, turn_id)).to_dict()

    async def disable_and_drain_plugin(self, plugin_id: str) -> dict[str, object]:
        if self._plugin_drain is None:
            raise RuntimeError("当前 runtime 不支持插件 drain")
        message = await self._plugin_drain(plugin_id)
        return {"pluginId": plugin_id, "drained": True, "message": message}

    async def install_plugin(
        self,
        source: str,
        marketplace: str,
        ref: str,
        sparse: list[str],
        owner_turn_id: str = "",
    ) -> dict[str, object]:
        if self._plugin_install is None:
            raise RuntimeError("当前 runtime 不支持插件安装")
        try:
            if not owner_turn_id:
                return await self._plugin_install(source, marketplace, ref, sparse)
            return await self._plugin_install(
                source, marketplace, ref, sparse, owner_turn_id
            )
        except Exception as exc:
            logger.exception("runtime-owned plugin install failed")
            raise PluginManagementError(str(exc)) from exc

    def plugin_status(self) -> dict[str, object]:
        if self._plugin_status is None:
            raise RuntimeError("当前 runtime 不支持插件候选状态")
        try:
            return self._plugin_status()
        except Exception as exc:
            logger.exception("plugin candidate status failed")
            raise PluginManagementError(str(exc)) from exc

    async def promote_plugin(self, plugin_id: str) -> dict[str, object]:
        if self._plugin_promote is None:
            raise RuntimeError("当前 runtime 不支持插件 promote")
        try:
            return await self._plugin_promote(plugin_id)
        except Exception as exc:
            logger.exception("plugin promote failed plugin=%s", plugin_id)
            raise PluginManagementError(str(exc)) from exc

    async def discard_plugin(self, plugin_id: str) -> dict[str, object]:
        if self._plugin_discard is None:
            raise RuntimeError("当前 runtime 不支持插件 discard")
        try:
            return await self._plugin_discard(plugin_id)
        except Exception as exc:
            logger.exception("plugin discard failed plugin=%s", plugin_id)
            raise PluginManagementError(str(exc)) from exc

    async def register_plugin_uninstall(
        self,
        plugin_id: str,
        owner_turn_id: str,
    ) -> dict[str, object]:
        if self._plugin_uninstall_register is None:
            return self.start_plugin_uninstall(plugin_id).record()
        try:
            return await self._plugin_uninstall_register(plugin_id, owner_turn_id)
        except Exception as exc:
            logger.exception(
                "plugin uninstall registration failed plugin=%s", plugin_id
            )
            raise PluginManagementError(str(exc)) from exc

    async def revert_plugin(self, owner_turn_id: str) -> dict[str, object]:
        if self._plugin_revert is None:
            raise RuntimeError("当前 runtime 不支持 plugin revert")
        try:
            return await self._plugin_revert(owner_turn_id)
        except Exception as exc:
            logger.exception("plugin revert failed owner_turn=%s", owner_turn_id)
            raise PluginManagementError(str(exc)) from exc

    def start_plugin_uninstall(self, plugin_id: str) -> PluginOperationHandle:
        """启动服务端卸载任务，不阻塞调用 turn 持有的 snapshot lease。"""

        if self._plugin_uninstall is None:
            raise RuntimeError("当前 runtime 不支持插件卸载")
        from agent.control.ids import new_operation_id

        operation_id = new_operation_id()
        task = asyncio.create_task(
            self._run_plugin_uninstall(operation_id, plugin_id),
            name=f"control-plugin-uninstall:{operation_id}",
        )
        self._operation_tasks.add(task)
        task.add_done_callback(self._operation_tasks.discard)
        return PluginOperationHandle(operation_id, plugin_id, task)

    async def shutdown(self) -> None:
        for task in self._operation_tasks:
            task.cancel()
        if self._operation_tasks:
            await asyncio.gather(*self._operation_tasks, return_exceptions=True)
        self._operation_tasks.clear()

    async def _run_plugin_uninstall(
        self,
        operation_id: str,
        plugin_id: str,
    ) -> dict[str, object]:
        assert self._plugin_uninstall is not None
        try:
            result = await self._plugin_uninstall(plugin_id)
        except Exception as exc:
            return {
                "id": operation_id,
                "pluginId": plugin_id,
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        return {
            "id": operation_id,
            "pluginId": plugin_id,
            "status": "completed",
            "result": result,
        }

    def _thread_record(self, thread_id: str) -> ThreadRecord:
        meta = self.sessions.control_store.get_session_meta(thread_id)
        if meta is None:
            raise ThreadNotFoundError(f"thread 不存在: {thread_id}")
        source = (
            ThreadSource.PROGRAMMATIC
            if thread_id.startswith("programmatic:")
            else ThreadSource.CHANNEL if ":" in thread_id else ThreadSource.INTERNAL
        )
        created_at = datetime.fromisoformat(str(meta["created_at"]))
        updated_at = datetime.fromisoformat(str(meta["updated_at"]))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        else:
            created_at = created_at.astimezone(UTC)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        else:
            updated_at = updated_at.astimezone(UTC)
        return ThreadRecord(
            id=thread_id,
            source=source,
            created_at=created_at,
            updated_at=updated_at,
            metadata=dict(meta["metadata"]),
        )


@dataclass(frozen=True)
class PluginOperationHandle:
    id: str
    plugin_id: str
    task: asyncio.Task[dict[str, object]]

    def record(self) -> dict[str, object]:
        return {"id": self.id, "pluginId": self.plugin_id, "status": "in_progress"}
