from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.control_execution import execute_control_turn
from bootstrap.tools import build_core_runtime
from bootstrap.workspace_lock import WorkspaceInstanceLock
from core.net.http import SharedHttpResources
from infra.control.stdio import StdioAppServer


async def run_stdio_app_server(config: Config, workspace: Path) -> None:
    """启动无渠道 runtime，并把唯一控制连接托管在 stdio。"""

    http_resources = SharedHttpResources()
    workspace_lock = WorkspaceInstanceLock(workspace)
    workspace_lock.acquire()
    core = None
    runtime: ConversationRuntime | None = None
    service: ControlService | None = None
    try:
        # 1. 使用正式 core/provider wiring 建立应用服务。
        core = build_core_runtime(config, workspace, http_resources)
        await core.start()

        async def execute(request: TurnRequest):
            return await execute_control_turn(core.loop, core.event_bus, request)

        runtime = ConversationRuntime(core.session_manager.control_store, execute)
        service = ControlService(
            runtime,
            core.session_manager,
            workspace,
            consolidate=core.loop.trigger_memory_consolidation,
        )

        # 2. EOF 代表父进程关闭连接，随后按 owner 顺序收束 runtime。
        await StdioAppServer(service, max_message_bytes=config.app_server.max_message_bytes).run()
    finally:
        await run_cleanup_steps(
            ("control_service.shutdown", service.shutdown if service else _noop),
            ("conversation_runtime.shutdown", runtime.shutdown if runtime else _noop),
            ("core.stop", core.stop if core else _noop),
            (
                "memory_runtime.aclose",
                core.memory_runtime.aclose if core else _noop,
            ),
            ("http_resources.aclose", http_resources.aclose),
            ("workspace_lock.release", lambda: _release_lock(workspace_lock)),
        )


async def _noop() -> None:
    return None


async def _release_lock(lock: WorkspaceInstanceLock) -> None:
    lock.release()
