from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
import json
from pathlib import Path

from agent.host_bridge.factory import ShellProcessManagerProtocol, build_shell_process_manager
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey
from agent.tools.unified_exec import ExecutionCleanupReport, ExecutionResult


class ProcessCleanupError(RuntimeError):
    def __init__(self, report: ExecutionCleanupReport):
        self.report = report
        super().__init__(f"进程清理未确认: {report.failures}")


class PluginProcesses:
    """实际进程由宿主统一保留；插件只操作自己命名的进程集合。"""

    def __init__(
        self, *, formal: bool = True,
        factory: Callable[[], ShellProcessManagerProtocol] = build_shell_process_manager,
    ):
        self._formal = formal
        self._factory = factory
        self._manager: ShellProcessManagerProtocol | None = None
        self._closed = False
        self._operations = 0
        self._drained = asyncio.Event()
        self._drained.set()

    def start(self) -> None:
        if self._closed and (self._manager is not None or self._operations):
            raise RuntimeError("旧进程资源尚未确认清理")
        self._closed = False

    @contextmanager
    def _operation(self, ctx: Context, key: str) -> Generator[str]:
        """同步接纳实际调用，关闭时先排空，防止已接纳 spawn 晚于 shutdown。"""
        if not self._formal or self._closed:
            raise RuntimeError("当前不能操作正式进程")
        owner = ctx.require_runtime_owner(PROCESSES, self)
        if not isinstance(key, str) or not key:
            raise ValueError("进程 owner key 不能为空")
        self._operations += 1
        self._drained.clear()
        try:
            yield json.dumps((owner, key), ensure_ascii=False, separators=(",", ":"))
        finally:
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
        max_output_tokens: int, hard_timeout_s: int,
    ) -> ExecutionResult:
        with self._operation(ctx, owner_key) as owner:
            return await self._backend().exec_command(
                command=command, argv=argv, cwd=cwd, env=env, tty=tty,
                yield_time_ms=yield_time_ms, max_output_tokens=max_output_tokens,
                hard_timeout_s=hard_timeout_s, owner_session_key=owner,
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

    async def close(self) -> None:
        """关闭准入并核对全部清理；失败保留原 manager 和进程证据。"""
        self._closed = True
        _ = await self._drained.wait()
        if self._manager is not None:
            report = await self._manager.shutdown()
            if report.failures:
                raise ProcessCleanupError(report)
            self._manager = None


PROCESSES = ServiceKey[PluginProcesses]("core.processes")
