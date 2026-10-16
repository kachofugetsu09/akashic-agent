from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
import json
from pathlib import Path

from .bridge.factory import build_shell_process_manager, build_requirements_checker
from .shell_runtime import resolve_shell
from plugins.host_execution.contract import ResolvedShell, RequirementsChecker
from plugins.host_execution.contract import (
    PROCESSES, TERMINALS, Processes, Terminals, TerminalInfo, TerminalOutput,
)
from plugins.host_execution.contract import (
    ExecutionCleanupReport, ExecutionResult, ExecutionState, ShellProcessManagerProtocol
)
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey


class PluginProcesses:
    """实际进程由本 generation 保留；插件只操作自己命名的进程集合。"""

    def __init__(
        self, *, formal: bool = True,
        factory: Callable[[], ShellProcessManagerProtocol] = build_shell_process_manager,
    ):
        self._formal = formal
        self._factory = factory
        self._manager: ShellProcessManagerProtocol | None = None
        self._closed = False
        self._operations = 0
        self._terminal_writes: set[asyncio.Task[object]] = set()
        self._drained = asyncio.Event()
        self._drained.set()

    def resolve_shell(self, requested: str | None = None) -> ResolvedShell:
        return resolve_shell(requested)

    def requirements_checker(self) -> RequirementsChecker | None:
        return build_requirements_checker()

    @contextmanager
    def _operation(
        self, ctx: Context, key: str,
        service: tuple[ServiceKey[Processes] | ServiceKey[Terminals], object] | None = None,
        *, terminal_write: bool = False,
    ) -> Generator[str]:
        """同步接纳实际调用，关闭时先排空，防止已接纳 spawn 晚于 shutdown。"""
        if not self._formal or self._closed:
            raise RuntimeError("当前不能操作正式进程")
        service_key, provider = service or (PROCESSES, self)
        owner = ctx.require_runtime_identity(service_key, provider).plugin_id
        if not isinstance(key, str) or not key:
            raise ValueError("进程 owner key 不能为空")
        task = asyncio.current_task() if terminal_write else None
        if task is not None:
            self._terminal_writes.add(task)
        self._operations += 1
        self._drained.clear()
        try:
            yield json.dumps((owner, key), ensure_ascii=False, separators=(",", ":"))
        finally:
            if task is not None:
                self._terminal_writes.remove(task)
            self._operations -= 1
            if self._operations == 0:
                self._drained.set()

    def _backend(self) -> ShellProcessManagerProtocol:
        if self._manager is None:
            self._manager = self._factory()
        return self._manager

    async def exec_command(
        self, ctx: Context, owner_key: str, *, command: str, argv: list[str],
        cwd: Path | None, env: dict[str, str], tty: bool, yield_time_ms: int,
        max_output_tokens: int, hard_timeout_s: int, shell_snapshot: bool = False,
    ) -> ExecutionResult:
        with self._operation(ctx, owner_key) as owner:
            return await self._backend().exec_command(
                command=command, argv=argv, cwd=cwd, env=env, tty=tty,
                yield_time_ms=yield_time_ms, max_output_tokens=max_output_tokens,
                hard_timeout_s=hard_timeout_s, owner_session_key=owner, shell_snapshot=shell_snapshot,
            )

    async def write_stdin(
        self, ctx: Context, owner_key: str, *, execution_id: int, chars: str,
        yield_time_ms: int, max_output_tokens: int,
    ) -> ExecutionResult:
        with self._operation(ctx, owner_key) as owner:
            return await self._backend().write_stdin(
                execution_id=execution_id, chars=chars, yield_time_ms=yield_time_ms,
                max_output_tokens=max_output_tokens, owner_session_key=owner,
            )

    async def terminate_execution(self, ctx: Context, owner_key: str, execution_id: int) -> bool:
        with self._operation(ctx, owner_key) as owner:
            if self._manager is None:
                return False
            return await self._manager.terminate_execution(execution_id, owner_session_key=owner)

    async def terminate_owner(self, ctx: Context, owner_key: str) -> ExecutionCleanupReport:
        with self._operation(ctx, owner_key) as owner:
            if self._manager is None:
                return ExecutionCleanupReport((), (), ())
            return await self._manager.terminate_owner(owner)

    async def execution_states(self, ctx: Context, owner_key: str) -> list[ExecutionState]:
        with self._operation(ctx, owner_key) as owner:
            if self._manager is None:
                return []
            return await self._manager.execution_states(owner)

    async def close(self) -> None:
        """关闭准入并核对全部清理；失败保留原 manager 和进程证据。"""
        self._closed = True
        # 写满 PTY 的请求没有时限；先取消等待（远端同时收到取消），避免等它排空才关闭 PTY。
        for task in tuple(self._terminal_writes):
            _ = task.cancel()
        _ = await self._drained.wait()
        if self._manager is not None:
            report = await self._manager.shutdown()
            if report.failures:
                raise RuntimeError(f"进程清理未确认: {report.failures}")
            self._manager = None


class PluginTerminals:
    """交互终端与进程共用同一 manager 与关闭排空；owner 同样由调用插件身份限定。"""

    def __init__(self, processes: PluginProcesses) -> None:
        self._processes = processes

    @contextmanager
    def _operation(self, ctx: Context, key: str, *, terminal_write: bool = False) -> Generator[str]:
        with self._processes._operation(ctx, key, (TERMINALS, self), terminal_write=terminal_write) as owner:
            yield owner

    async def open(
        self, ctx: Context, owner_key: str, *, cwd: Path | None, cols: int, rows: int,
    ) -> TerminalInfo:
        with self._operation(ctx, owner_key) as owner:
            return await self._processes._backend().terminals.open_terminal(
                owner_session_key=owner, cwd=cwd, cols=cols, rows=rows,
            )

    async def write(self, ctx: Context, owner_key: str, terminal_id: int, data: bytes) -> None:
        with self._operation(ctx, owner_key, terminal_write=True) as owner:
            await self._processes._backend().terminals.write_terminal(
                owner_session_key=owner, terminal_id=terminal_id, data=data,
            )

    async def resize(self, ctx: Context, owner_key: str, terminal_id: int, *, cols: int, rows: int) -> None:
        with self._operation(ctx, owner_key) as owner:
            await self._processes._backend().terminals.resize_terminal(
                owner_session_key=owner, terminal_id=terminal_id, cols=cols, rows=rows,
            )

    async def read(
        self, ctx: Context, owner_key: str, terminal_id: int, *, offset: int, wait_ms: int,
    ) -> TerminalOutput:
        with self._operation(ctx, owner_key) as owner:
            return await self._processes._backend().terminals.read_terminal(
                owner_session_key=owner, terminal_id=terminal_id, offset=offset, wait_ms=wait_ms,
            )

    async def close(self, ctx: Context, owner_key: str, terminal_id: int) -> bool:
        with self._operation(ctx, owner_key) as owner:
            return await self._processes._backend().terminals.close_terminal(
                owner_session_key=owner, terminal_id=terminal_id,
            )

    async def list(self, ctx: Context, owner_key: str) -> list[TerminalInfo]:
        with self._operation(ctx, owner_key) as owner:
            return await self._processes._backend().terminals.list_terminals(owner)
