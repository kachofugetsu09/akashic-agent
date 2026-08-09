from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from agent.host_bridge.client import HostBridgeShellProcessManager
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.unified_exec import ExecutionResult
from agent.tools.unified_exec import ShellProcessManager

_SOCKET_ENV = "AKASHIC_HOST_BRIDGE_SOCKET"
_TOKEN_ENV = "AKASHIC_HOST_BRIDGE_TOKEN"
_BOOT_ID_ENV = "AKASHIC_BOOT_ID"
_MODE_ENV = "AKASHIC_EXECUTION_MODE"


class ShellProcessManagerProtocol(Protocol):
    async def exec_command(
        self,
        *,
        command: str,
        argv: list[str],
        cwd: Path | None,
        env: dict[str, str],
        tty: bool,
        yield_time_ms: int,
        max_output_tokens: int,
        hard_timeout_s: int,
        owner_session_key: str,
    ) -> ExecutionResult: ...
    async def write_stdin(
        self,
        *,
        execution_id: int,
        chars: str,
        yield_time_ms: int,
        max_output_tokens: int,
        owner_session_key: str,
    ) -> ExecutionResult: ...
    async def terminate_execution(
        self, execution_id: int, *, owner_session_key: str
    ) -> bool: ...
    async def terminate_owner(
        self, owner_session_key: str
    ) -> ExecutionCleanupReport: ...
    async def shutdown(self) -> ExecutionCleanupReport: ...
    async def active_execution_ids(self) -> list[int]: ...


def build_shell_process_manager() -> ShellProcessManagerProtocol:
    """Select the explicit local or host-bridge execution backend."""

    mode = os.environ.get(_MODE_ENV, "local")
    if mode == "local":
        return ShellProcessManager()
    if mode != "host-bridge":
        raise RuntimeError(f"{_MODE_ENV} 只能是 local 或 host-bridge")
    socket_text = os.environ.get(_SOCKET_ENV)
    if socket_text is None:
        raise RuntimeError(f"host-bridge 模式缺少 {_SOCKET_ENV}")
    token = os.environ.get(_TOKEN_ENV)
    boot_id = os.environ.get(_BOOT_ID_ENV)
    if not token or not boot_id:
        raise RuntimeError(
            f"配置 {_SOCKET_ENV} 时必须同时提供 {_TOKEN_ENV} 和 {_BOOT_ID_ENV}"
        )
    socket_path = Path(socket_text)
    if not socket_path.is_absolute():
        raise RuntimeError(f"{_SOCKET_ENV} 必须是绝对路径")
    return HostBridgeShellProcessManager(socket_path, boot_id, token)


def build_file_bridge() -> HostBridgeShellProcessManager | None:
    """Build a file RPC client only in explicit host-bridge mode."""

    manager = build_shell_process_manager()
    if isinstance(manager, HostBridgeShellProcessManager):
        return manager
    return None
