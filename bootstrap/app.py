from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Awaitable, Callable

from bootstrap.web_shell import WebShellServer

from agent.restart import RestartGate
from agent.config_models import Config
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.tools import CoreRuntime, build_core_runtime
from bootstrap.workspace_lock import WorkspaceInstanceLock
from agent.plugins.watcher import PluginWatcher
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from core.common.diagnostic_log import configure_logging

configure_logging()
logging.getLogger("agent.plugins.manager").setLevel(
    os.environ.get("AKASHIC_PLUGIN_LOG_LEVEL", "INFO").upper()
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


_run_cleanup_steps = run_cleanup_steps


def _clear_readiness(
    readiness: RuntimeReadiness | None,
) -> Callable[[], Awaitable[None]]:
    async def clear() -> None:
        if readiness is not None:
            readiness.clear()

    return clear


def _raise_unexpected_task_errors(name: str, results: Iterable[object]) -> None:
    """记录并重新抛出任务停止时的首个非取消异常。"""

    first_error: BaseException | None = None
    for result in results:
        if not isinstance(result, BaseException) or isinstance(
            result, asyncio.CancelledError
        ):
            continue
        logger.error(
            "%s failed while stopping",
            name,
            exc_info=(type(result), result, result.__traceback__),
        )
        if first_error is None:
            first_error = result
    if first_error is not None:
        raise first_error


def _stop_plugin_watcher(
    watcher: PluginWatcher | None,
    task: asyncio.Task[None] | None,
) -> Callable[[], Awaitable[None]]:
    async def stop() -> None:
        if watcher is not None:
            watcher.stop()
            await watcher.wait_stopped()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                return

    return stop


def _wait_server_task(
    task: asyncio.Task[None] | None,
) -> Callable[[], Awaitable[None]]:
    async def wait() -> None:
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            return

    return wait


class AppRuntime:
    def __init__(
        self,
        config: Config,
        workspace: Path,
        *,
        restart_gate: RestartGate | None = None,
        readiness: RuntimeReadiness | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        if restart_gate is None and readiness is not None:
            # fixture/嵌入式 host 没有 supervisor 时，readiness 仍是该 host
            # 的启动边界；让 Core、Channel Host 和 readiness 使用同一身份。
            restart_gate = RestartGate(
                boot_id=readiness.boot_id,
                supervised=False,
            )
        self.restart_gate = restart_gate
        self.readiness = readiness
        self.http_resources = SharedHttpResources()
        self.core: CoreRuntime | None = None
        self.web_shell: WebShellServer | None = None
        self.web_shell_task: asyncio.Task[None] | None = None
        self.plugin_watcher: PluginWatcher | None = None
        self.plugin_watcher_task: asyncio.Task[None] | None = None
        self._shutdown = False
        self._started = False
        self._plugin_candidate_tasks: set[asyncio.Task[Any]] = set()
        self._plugin_reload_signal_installed = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._workspace_lock = WorkspaceInstanceLock(workspace)

    async def start(self) -> None:
        if self._started:
            return
        self._workspace_lock.acquire()
        if self.readiness is not None:
            self.readiness.mark_stage("workspace.locked")
        try:
            configure_default_shared_http_resources(self.http_resources)
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
                restart_gate=self.restart_gate,
                host_ready=(lambda: self.readiness.ready) if self.readiness else None,
            )
            await self.core.start()
            if self.readiness is not None:
                self.readiness.mark_stage("core.ready")
            plugin_manager = getattr(self.core, "plugin_manager", None)
            if self.readiness is not None:
                self.readiness.mark_stage("services.ready")

            if plugin_manager is None:
                raise RuntimeError("插件 Runtime 不可用")
            if os.environ.get("AKASHIC_SUPERVISED") != "1" and "AKASHIC_WEB_PORT" in os.environ:
                from bootstrap.web_shell import create_web_shell_server
                host = os.environ.get("AKASHIC_WEB_HOST", "127.0.0.1")
                if host != "127.0.0.1" and os.environ.get("AKASHIC_WEB_ALLOW_NON_LOOPBACK") != "1":
                    raise ValueError("公网 Web Shell 需要 AKASHIC_WEB_ALLOW_NON_LOOPBACK=1")
                port = int(os.environ["AKASHIC_WEB_PORT"])
                if not 1 <= port <= 65535:
                    raise ValueError("AKASHIC_WEB_PORT 必须是 1 到 65535")
                shell = create_web_shell_server(self.workspace, host=host, port=port)
                self.web_shell = shell
                self.web_shell_task = asyncio.create_task(shell.serve(), name="web_shell")
                await asyncio.to_thread(shell.startup_event.wait, 5)
                if not shell.started:
                    raise RuntimeError("Web Shell 未完成启动")
            if plugin_manager is not None:
                self.plugin_watcher = PluginWatcher(
                    plugin_manager,
                    accepting=lambda: self.restart_gate is None or self.restart_gate.accepting,
                )
                self.plugin_watcher_task = asyncio.create_task(
                    self.plugin_watcher.run(),
                    name="plugin_watcher",
                )

            self._install_plugin_reload_signal()
            if self.readiness is not None:
                self.readiness.mark_stage("runtime.started")
            self._started = True
        except (asyncio.CancelledError, Exception) as startup_error:
            try:
                await self.shutdown()
            except (asyncio.CancelledError, Exception) as rollback_error:
                raise startup_error from rollback_error
            raise

    async def run(self) -> None:
        """监督宿主任务，退出时完成清理并保留原始错误。"""
        run_error: BaseException | None = None
        try:
            # 1. 完整启动后监督实际宿主任务与进程停止请求。
            await self.start()
            assert self.core is not None
            shutdown_task = asyncio.create_task(self.core.restart_gate.wait_shutdown(), name="process_shutdown")
            self._shutdown_task = shutdown_task
            watched_tasks = {
                task
                for task in (
                    self.web_shell_task,
                    self.plugin_watcher_task,
                    shutdown_task,
                )
                if task is not None
            }
            # 实际宿主监督任务获得一次调度机会后仍存活，才对外发布 ready。
            done, _ = await asyncio.wait(watched_tasks, timeout=0)
            if not done:
                if self.readiness is not None:
                    self.readiness.mark_ready()
                done, _ = await asyncio.wait(
                    watched_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            if shutdown_task in done:
                try:
                    await shutdown_task
                finally:
                    self._shutdown_task = None
            else:
                if self.web_shell_task is not None and self.web_shell_task in done:
                    watched_task = self.web_shell_task
                    self.web_shell_task = None
                elif (
                    self.plugin_watcher_task is not None
                    and self.plugin_watcher_task in done
                ):
                    watched_task = self.plugin_watcher_task
                    self.plugin_watcher_task = None
                else:
                    raise RuntimeError("未知 runtime watcher task")
                await watched_task
        except (asyncio.CancelledError, Exception) as error:
            run_error = error

        # 2. 清理全部资源；关闭失败作为原始错误的原因保留。
        shutdown_error: BaseException | None = None
        try:
            await self.shutdown()
        except (asyncio.CancelledError, Exception) as error:
            shutdown_error = error

        if run_error is not None:
            if shutdown_error is not None and shutdown_error is not run_error:
                raise run_error from shutdown_error
            raise run_error
        if shutdown_error is not None:
            raise shutdown_error

    async def _cancel_shutdown_task(self) -> None:
        """收束进程停止等待；取消正常结束，其他错误继续暴露。"""
        task = self._shutdown_task
        if task is None:
            return
        _ = task.cancel()
        try:
            results = await asyncio.gather(task, return_exceptions=True)
        finally:
            self._shutdown_task = None

        _raise_unexpected_task_errors("process shutdown task", results)

    async def _cancel_plugin_candidate_tasks(self) -> None:
        for task in self._plugin_candidate_tasks:
            _ = task.cancel()
        results: list[object] = []
        try:
            if self._plugin_candidate_tasks:
                results = await asyncio.gather(
                    *self._plugin_candidate_tasks,
                    return_exceptions=True,
                )
        finally:
            self._plugin_candidate_tasks.clear()
        _raise_unexpected_task_errors("plugin candidate task", results)

    async def _request_server_shutdown(self) -> None:
        if self.web_shell is not None:
            self.web_shell.should_exit = True

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        closing_root = None if self.core is None else self.core.plugin_manager.live_root
        try:
            self._remove_plugin_reload_signal()
            # 已排空业务后先关闭入站连接；插件释放前保留数据库和 HTTP provider。
            await _run_cleanup_steps(
                ("plugin_candidate_tasks.cancel", self._cancel_plugin_candidate_tasks),
                ("process_shutdown.cancel", self._cancel_shutdown_task),
                ("servers.request_shutdown", self._request_server_shutdown),
                ("web_shell.wait", _wait_server_task(self.web_shell_task)),
                (
                    "plugin_watcher.stop",
                    _stop_plugin_watcher(
                        self.plugin_watcher,
                        self.plugin_watcher_task,
                    ),
                ),
            )
            if self.core is not None:
                await self.core.stop()
            await _run_cleanup_steps(
                ("http_resources.aclose", self.http_resources.aclose),
                (
                    "runtime_readiness.clear",
                    _clear_readiness(self.readiness),
                ),
            )
            if self._started and self.readiness is not None:
                self.readiness.mark_closed(None if closing_root is None else closing_root.generation_id)
        finally:
            self._workspace_lock.release()
            clear_default_shared_http_resources(self.http_resources)

    def _install_plugin_reload_signal(self) -> None:
        if not hasattr(signal, "SIGHUP"):
            return
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            return
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGHUP, self._schedule_plugin_candidate_scan)
        self._plugin_reload_signal_installed = True

    def _remove_plugin_reload_signal(self) -> None:
        if not self._plugin_reload_signal_installed:
            return
        _ = asyncio.get_running_loop().remove_signal_handler(signal.SIGHUP)
        self._plugin_reload_signal_installed = False

    def _schedule_plugin_candidate_scan(self) -> None:
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None or self._shutdown or (self.restart_gate is not None and not self.restart_gate.accepting):
            return
        if self.plugin_watcher is not None:
            self.plugin_watcher.request_scan()
            return
        task = asyncio.create_task(
            manager.reconcile_changed(),
            name="plugin_reload_scan",
        )
        self._plugin_candidate_tasks.add(task)
        task.add_done_callback(self._plugin_candidate_scan_done)

    def _plugin_candidate_scan_done(self, task: asyncio.Task[Any]) -> None:
        self._plugin_candidate_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "plugin candidate scan failed",
                exc_info=(type(error), error, error.__traceback__),
            )


def build_app_runtime(
    config: Config,
    workspace: Path,
    *,
    restart_gate: RestartGate | None = None,
    readiness: RuntimeReadiness | None = None,
) -> AppRuntime:
    return AppRuntime(
        config,
        workspace,
        restart_gate=restart_gate,
        readiness=readiness,
    )
