from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import stat
from pathlib import Path
from typing import Any, Awaitable, Callable

import uvicorn

from agent.config import resolve_app_server_endpoint
from agent.control.service import ControlService
from agent.host_bridge.monitor import HostBridgeStatus, build_host_bridge_monitor
from agent.host_bridge.monitor import claim_host_bridge_boot
from agent.restart import RestartGate
from agent.config_models import Config
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.web_runtime import dashboard_socket_path, prepare_runtime_socket
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.tools import CoreRuntime, build_core_runtime
from bootstrap.workspace_lock import WorkspaceInstanceLock
from bootstrap.workspace_token import ensure_workspace_token
from bus.event_bus import EventBus
from bus.queue import MessageBus
from agent.plugins.watcher import PluginWatcher
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from core.common.diagnostic_log import configure_logging
from infra.control.socket import SocketAppServer, is_tcp_endpoint

configure_logging()
logging.getLogger("agent.plugins.manager").setLevel(
    os.environ.get("AKASHIC_PLUGIN_LOG_LEVEL", "INFO").upper()
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


_run_cleanup_steps = run_cleanup_steps


async def _noop_async() -> None:
    return None


def _release_workspace_lock(
    lock: WorkspaceInstanceLock,
) -> Callable[[], Awaitable[None]]:
    async def release() -> None:
        lock.release()

    return release


def _clear_readiness(
    readiness: RuntimeReadiness | None,
) -> Callable[[], Awaitable[None]]:
    async def clear() -> None:
        if readiness is not None:
            readiness.clear()

    return clear


def _close_message_bus(bus: object | None) -> Callable[[], Awaitable[None]]:
    """返回已初始化 MessageBus 的异步关闭动作。"""

    if isinstance(bus, MessageBus):
        return bus.aclose
    return _noop_async


def _raise_unexpected_task_errors(name: str, results: list[object]) -> None:
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


async def _run_primary_tasks(tasks: list[asyncio.Future[Any]]) -> None:
    """监督 runtime tasks，并在失败或取消时等待兄弟任务收束。"""

    try:
        _ = await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        # gather 已把取消传播给子任务；再次 cancel 会打断子任务的 finally。
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except Exception:
        for task in tasks:
            if not task.done():
                _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
        raise


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


def _remove_dashboard_socket(
    server: uvicorn.Server | None,
    task: asyncio.Task[None] | None,
) -> Callable[[], Awaitable[None]]:
    """Remove this host's Unix socket after its dashboard listener has stopped."""

    async def remove() -> None:
        if server is None:
            return
        if task is not None and not task.done():
            raise RuntimeError("Dashboard server task is still running")
        if not server.started:
            return
        if any(listener.is_serving() for listener in server.servers):
            raise RuntimeError("Dashboard Unix socket is still serving")
        uds = server.config.uds
        if uds is None:
            return
        path = Path(uds)
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"Dashboard socket path is not a socket: {path}")
        # Python 3.12 closes the listener but leaves its Unix socket pathname.
        path.unlink()

    return remove


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
        self.host_bridge_status = HostBridgeStatus()
        self.http_resources = SharedHttpResources()
        self.app_server: SocketAppServer | None = None
        self.control_service: ControlService | None = None
        self.core: CoreRuntime | None = None
        self.bus = None
        self.event_bus: EventBus | None = None
        self.dashboard_server: uvicorn.Server | None = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.plugin_watcher: PluginWatcher | None = None
        self.plugin_watcher_task: asyncio.Task[None] | None = None
        self.tasks: list[Awaitable[None]] = []
        self._shutdown = False
        self._started = False
        self._plugin_candidate_tasks: set[asyncio.Task[Any]] = set()
        self._plugin_reload_signal_installed = False
        self._runtime_tasks: set[asyncio.Future[Any]] = set()
        self._primary_task: asyncio.Future[Any] | None = None
        self._workspace_lock = WorkspaceInstanceLock(workspace)

    async def start(self) -> None:
        if self._started:
            return
        self._workspace_lock.acquire()
        if self.readiness is not None:
            self.readiness.mark_stage("workspace.locked")
        try:
            claim = await claim_host_bridge_boot()
            if claim is not None and self.readiness is not None:
                self.readiness.mark_stage("host_bridge.owner")
            configure_default_shared_http_resources(self.http_resources)
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
                restart_gate=self.restart_gate,
                clear_stale_session_admissions=True,
            )
            self.bus = self.core.bus
            event_bus = self.core.event_bus
            self.event_bus = event_bus
            manager = self.core.plugin_manager
            manager.bind_endpoint_switcher(self._swap_plugin_endpoints)
            self.dashboard_server = build_dashboard_server(
                workspace=self.workspace,
                plugin_manager=manager,
                host_bridge_status=self.host_bridge_status.snapshot,
            )
            await self.core.start()
            if self.readiness is not None:
                self.readiness.mark_stage("core.ready")
            app_server_endpoint: str | None = None
            workspace_token: str | None = None
            if self.config.app_server.enabled:
                app_server_endpoint = resolve_app_server_endpoint(self.config.app_server.listen, self.workspace)
                if is_tcp_endpoint(app_server_endpoint):
                    workspace_token = ensure_workspace_token(self.workspace)
            from bootstrap.app_server import build_control_service

            self.control_service = build_control_service(
                self.core, workspace_token=workspace_token,
                boot_id=self.readiness.boot_id if self.readiness else None,
                ready=(lambda: self.readiness.ready) if self.readiness else None,
            )
            if self.config.app_server.enabled:
                assert app_server_endpoint is not None
                self.app_server = SocketAppServer(
                    app_server_endpoint,
                    self.control_service,
                    max_connections=self.config.app_server.max_connections,
                    max_pending_requests=self.config.app_server.ingress_queue_size,
                    max_message_bytes=self.config.app_server.max_message_bytes,
                    outbound_queue_size=self.config.app_server.outbound_queue_size,
                )
                await self.app_server.start()

            plugin_manager = getattr(self.core, "plugin_manager", None)
            if self.readiness is not None:
                self.readiness.mark_stage("services.ready")

            # provider 已开放实际 binding；这里只触发传输 owner 的 pending 恢复。
            await self.bus.recover_durable_inbounds()
            if self.readiness is not None:
                self.readiness.mark_stage("channels.ready")
            if plugin_manager is None:
                raise RuntimeError("插件 Runtime 不可用")
            host_bridge_monitor = build_host_bridge_monitor(self.host_bridge_status)
            self.tasks = []
            if host_bridge_monitor is not None:
                self.tasks.append(host_bridge_monitor)
            self.dashboard_server.config.uds = prepare_runtime_socket(
                dashboard_socket_path(self.workspace)
            )
            self.dashboard_task = asyncio.create_task(
                self.dashboard_server.serve(),
                name="dashboard_server",
            )
            if plugin_manager is not None:
                self.plugin_watcher = PluginWatcher(
                    plugin_manager,
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
        run_error: BaseException | None = None
        try:
            await self.start()
            runtime_tasks = self._schedule_runtime_tasks()
            if runtime_tasks:
                self._primary_task = asyncio.create_task(
                    _run_primary_tasks(runtime_tasks),
                    name="primary_runtime",
                )
            else:
                self._primary_task = None
            self._runtime_tasks.clear()
            watched_tasks = {
                task
                for task in (
                    self.dashboard_task,
                    self.plugin_watcher_task,
                )
                if task is not None
            }
            supervised_tasks = set(watched_tasks)
            if self._primary_task is not None:
                supervised_tasks.add(self._primary_task)
            if not supervised_tasks:
                raise RuntimeError("没有可监督的宿主任务")

            # 实际宿主监督任务获得一次调度机会后仍存活，才对外发布 ready。
            done, _ = await asyncio.wait(supervised_tasks, timeout=0)
            if not done:
                if self.readiness is not None:
                    self.readiness.mark_ready()
                done, _ = await asyncio.wait(
                    supervised_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            if self._primary_task is not None and self._primary_task in done:
                await self._primary_task
            else:
                if self.dashboard_task is not None and self.dashboard_task in done:
                    watched_task = self.dashboard_task
                    self.dashboard_task = None
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

    def _schedule_runtime_tasks(self) -> list[asyncio.Future[Any]]:
        pending = self.tasks
        self.tasks = []
        scheduled: list[asyncio.Future[Any]] = []
        try:
            for awaitable in pending:
                task = asyncio.ensure_future(awaitable)
                scheduled.append(task)
        except (asyncio.CancelledError, Exception):
            self._runtime_tasks = set(scheduled)
            self.tasks = pending[len(scheduled) :]
            for awaitable in self.tasks:
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
            raise
        self._runtime_tasks = set(scheduled)
        return scheduled

    async def _cancel_runtime_tasks(self) -> None:
        results: list[object] = []
        primary_task = self._primary_task
        try:
            if primary_task is not None:
                _ = primary_task.cancel()
                try:
                    await primary_task
                except (asyncio.CancelledError, Exception) as error:
                    results.append(error)
            elif self._runtime_tasks:
                for task in self._runtime_tasks:
                    _ = task.cancel()
                results = await asyncio.gather(
                    *self._runtime_tasks,
                    return_exceptions=True,
                )
        finally:
            self._runtime_tasks.clear()
            for awaitable in self.tasks:
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
            self.tasks.clear()
            self._primary_task = None

        _raise_unexpected_task_errors("primary runtime task", results)

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
        if self.dashboard_server is not None:
            self.dashboard_server.should_exit = True

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._remove_plugin_reload_signal()
            await _run_cleanup_steps(
                ("plugin_candidate_tasks.cancel", self._cancel_plugin_candidate_tasks),
                ("runtime_tasks.cancel", self._cancel_runtime_tasks),
                ("servers.request_shutdown", self._request_server_shutdown),
                (
                    "dashboard_server.wait",
                    _wait_server_task(self.dashboard_task),
                ),
                (
                    "dashboard_socket.remove",
                    _remove_dashboard_socket(self.dashboard_server, self.dashboard_task),
                ),
                ("message_bus.aclose", _close_message_bus(self.bus)),
                (
                    "plugin_watcher.stop",
                    _stop_plugin_watcher(
                        self.plugin_watcher,
                        self.plugin_watcher_task,
                    ),
                ),
                (
                    "app_server.stop",
                    self.app_server.stop if self.app_server else _noop_async,
                ),
                (
                    "control_service.shutdown",
                    (
                        self.control_service.shutdown
                        if self.control_service
                        else _noop_async
                    ),
                ),
                ("core.stop", self.core.stop if self.core else _noop_async),
                ("http_resources.aclose", self.http_resources.aclose),
                (
                    "runtime_readiness.clear",
                    _clear_readiness(self.readiness),
                ),
                (
                    "workspace_lock.release",
                    _release_workspace_lock(self._workspace_lock),
                ),
            )
        finally:
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
        if manager is None or self._shutdown:
            return
        if self.plugin_watcher is not None:
            self.plugin_watcher.wake()
            return
        task = asyncio.create_task(
            manager.reconcile_changed(),
            name="plugin_reload_scan",
        )
        self._plugin_candidate_tasks.add(task)
        task.add_done_callback(self._plugin_candidate_scan_done)

    async def _disable_and_drain_plugin(self, plugin_id: str) -> str:
        plugin_id = plugin_id.strip()
        if not plugin_id:
            raise ValueError("缺少插件 ID")
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")
        await manager.reconcile_disabled_and_drain(plugin_id)
        return f"插件已停用并排空: {plugin_id}"

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

    async def _swap_plugin_endpoints(
        self,
        old_commands: tuple[tuple[str, str], ...],
        new_commands: tuple[tuple[str, str], ...],
    ) -> None:
        # Installed channel plugins resolve COMMANDS inside each exact
        # request scope.  There is no Core endpoint registry to mutate.
        _ = old_commands, new_commands


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
