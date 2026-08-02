from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agent.config import resolve_app_server_endpoint
from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from agent.restart import RestartCoordinator
from agent.config_models import Config
from bootstrap.channel_host import ChannelHost
from bootstrap.channels import start_channels
from bootstrap.chat_api import build_chat_server
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.control_execution import execute_control_turn
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.passive_worker import PassiveMessageWorker
from bootstrap.tools import CoreRuntime, build_core_runtime
from bootstrap.workspace_lock import WorkspaceInstanceLock
from bootstrap.workspace_token import ensure_workspace_token
from bus.event_bus import EventBus
from bus.queue import MessageBus
from agent.plugins.jobs import PluginJobRuntime
from agent.plugins.service_host import PluginServiceHost
from agent.plugins.watcher import PluginWatcher
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from infra.control.socket import SocketAppServer, is_tcp_endpoint

if TYPE_CHECKING:
    from proactive_v2.loop import ProactiveLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
logging.getLogger("agent.plugins.manager").setLevel(
    os.environ.get("AKASHIC_PLUGIN_LOG_LEVEL", "INFO").upper()
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

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


def _stop_plugin_jobs(runtime: PluginJobRuntime | None) -> Callable[[], Awaitable[None]]:
    async def stop() -> None:
        if runtime is not None:
            runtime.stop()
            await runtime.wait_stopped()

    return stop


def _stop_proactive(runtime: ProactiveLoop | None) -> Callable[[], Awaitable[None]]:
    async def stop() -> None:
        if runtime is not None:
            try:
                runtime.stop()
                await runtime.wait_stopped()
            finally:
                runtime.close()

    return stop


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


def _close_mobile_gateway(runtime: Any | None) -> Callable[[], Awaitable[None]]:
    async def close() -> None:
        if runtime is not None:
            await asyncio.to_thread(runtime.close)

    return close


def _dashboard_bind_address() -> tuple[str, int]:
    """Resolve and validate the dashboard listener from environment config."""

    # 1. Normalize the process boundary before any runtime service starts.
    host = os.environ.get("AKASHIC_DASHBOARD_HOST", "0.0.0.0").strip()
    if not host:
        raise ValueError("AKASHIC_DASHBOARD_HOST 不能为空")

    # 2. Reject invalid ports instead of falling back to the formal listener.
    raw_port = os.environ.get("AKASHIC_DASHBOARD_PORT", "2236")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise ValueError("AKASHIC_DASHBOARD_PORT 必须是 1 到 65535 的整数") from error
    if not 1 <= port <= 65_535:
        raise ValueError("AKASHIC_DASHBOARD_PORT 必须是 1 到 65535 的整数")
    return host, port


class AppRuntime:
    def __init__(
        self,
        config: Config,
        workspace: Path,
        *,
        restart_coordinator: RestartCoordinator | None = None,
        readiness: RuntimeReadiness | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.restart_coordinator = restart_coordinator
        self.readiness = readiness
        self.dashboard_host, self.dashboard_port = _dashboard_bind_address()
        self.http_resources = SharedHttpResources()
        self.app_server: SocketAppServer | None = None
        self.conversation_runtime: ConversationRuntime | None = None
        self.control_service: ControlService | None = None
        self.passive_worker: PassiveMessageWorker | None = None
        self.channel_host: ChannelHost | None = None
        self.core: CoreRuntime | None = None
        self.agent_loop = None
        self.bus = None
        self.event_bus: EventBus | None = None
        self.tools = None
        self.push_tool = None
        self.session_manager = None
        self.scheduler = None
        self.provider = None
        self.light_provider = None
        self.memory_runtime = None
        self.presence = None
        self.proactive_loop = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.chat_server = None
        self.chat_task: asyncio.Task[None] | None = None
        self.web_chat_channel = None
        self.mobile_gateway_runtime = None
        self.mobile_gateway_server = None
        self.mobile_gateway_task: asyncio.Task[None] | None = None
        self.plugin_job_runtime: PluginJobRuntime | None = None
        self.plugin_service_host: PluginServiceHost | None = None
        self.plugin_watcher: PluginWatcher | None = None
        self.plugin_watcher_task: asyncio.Task[None] | None = None
        self.workspace_mcp_watcher_task: asyncio.Task[None] | None = None
        self.tasks: list[Awaitable[None]] = []
        self._memory_optimizer = None
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
            configure_default_shared_http_resources(self.http_resources)
            core_kwargs = (
                {"restart_coordinator": self.restart_coordinator}
                if self.restart_coordinator is not None
                else {}
            )
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
                **core_kwargs,
                clear_stale_session_admissions=True,
            )
            self.agent_loop = self.core.loop
            self.bus = self.core.bus
            event_bus = self.core.event_bus
            self.event_bus = event_bus
            self.tools = self.core.tools
            self.push_tool = self.core.push_tool
            self.session_manager = self.core.session_manager
            self.scheduler = self.core.scheduler
            self.provider = self.core.provider
            self.light_provider = self.core.light_provider
            self.memory_runtime = self.core.memory_runtime
            self.presence = self.core.presence
            await self.core.start()
            if self.readiness is not None:
                self.readiness.mark_stage("core.ready")
            self.workspace_mcp_watcher_task = (
                self.core.workspace_mcp_watcher_task
            )

            async def _execute_control_request(request: TurnRequest):
                assert self.agent_loop is not None
                return await execute_control_turn(
                    self.agent_loop,
                    event_bus,
                    request,
                )

            self.conversation_runtime = ConversationRuntime(
                self.session_manager.control_store,
                _execute_control_request,
                restart_coordinator=self.restart_coordinator,
            )
            if self.restart_coordinator is not None:
                self.restart_coordinator.bind_admission(
                    quiesce=self.conversation_runtime.quiesce_for_restart,
                    resume=self.conversation_runtime.resume_after_restart_cancel,
                )
            app_server_endpoint: str | None = None
            workspace_token: str | None = None
            if self.config.app_server.enabled:
                app_server_endpoint = resolve_app_server_endpoint(
                    self.config.app_server.listen,
                    self.workspace,
                )
                if is_tcp_endpoint(app_server_endpoint):
                    workspace_token = ensure_workspace_token(self.workspace)
            self.control_service = ControlService(
                self.conversation_runtime,
                self.session_manager,
                self.workspace,
                plugin_drain=self._disable_and_drain_plugin,
                plugin_uninstall=self._uninstall_plugin,
                consolidate=(
                    self.agent_loop.trigger_memory_consolidation
                    if self.config.app_server.enabled
                    else None
                ),
                workspace_token=workspace_token,
                restart_coordinator=self.restart_coordinator,
                boot_id=self.readiness.boot_id if self.readiness else None,
                ready=(lambda: self.readiness.ready) if self.readiness else None,
            )
            self.passive_worker = PassiveMessageWorker(
                self.bus,
                self.conversation_runtime,
                self.agent_loop,
            )
            if self.restart_coordinator is not None:
                coordinator = self.restart_coordinator

                async def observe_delivery(
                    message: Any,
                    delivered: bool,
                ) -> None:
                    turn_id = str(message.control_turn_id or "")
                    if not turn_id:
                        return
                    if delivered:
                        coordinator.mark_delivered(turn_id)
                    else:
                        coordinator.mark_delivery_failed(
                            turn_id,
                            "channel callback did not deliver original response",
                        )

                self.bus.bind_outbound_delivery_observer(observe_delivery)
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
            if plugin_manager is not None:
                self.plugin_service_host = PluginServiceHost()
                snapshot = plugin_manager.current_snapshot
                service_bindings = {
                    plugin_id: {
                        service_id: dict(spec)
                        for service_id, spec in services.items()
                    }
                    for plugin_id, services in (
                        snapshot.managed_services.items() if snapshot is not None else ()
                    )
                }
                self.plugin_service_host.bind_plugin_services(service_bindings)
                await self.plugin_service_host.start_all()
                plugin_manager.bind_service_switcher(
                    self.plugin_service_host.swap_plugin_services
                )
            if self.readiness is not None:
                self.readiness.mark_stage("services.ready")
            if self.config.mobile_realtime.enabled:
                from infra.mobile_realtime.gateway import (
                    build_mobile_gateway_runtime,
                )

                self.mobile_gateway_runtime, _ = build_mobile_gateway_runtime(
                    self.config.mobile_realtime,
                    self.workspace,
                )
                self.bus.bind_durable_inbound_store(
                    self.session_manager.control_store
                )
                from infra.mobile_realtime.runtime_inspection import (
                    RuntimeInspectionService,
                )

                self.mobile_gateway_runtime.channel.bind_runtime_inspection(
                    RuntimeInspectionService(
                        workspace=self.workspace,
                        scheduler=self.scheduler,
                        snapshot_store=(
                            plugin_manager.snapshot_store
                            if plugin_manager is not None
                            else None
                        ),
                    )
                )
                if plugin_manager is not None:
                    from agent.plugins.mobile_ui import PluginMobileUiProvider

                    self.mobile_gateway_runtime.channel.bind_mobile_ui_provider(
                        PluginMobileUiProvider(plugin_manager)
                    )
            plugin_channels = list(plugin_manager.channels) if plugin_manager else []
            if self.mobile_gateway_runtime is not None:
                plugin_channels.append(self.mobile_gateway_runtime.channel)
            if self.config.channels.chat.enabled:
                from infra.channels.web_chat_channel import WebChatChannel

                self.web_chat_channel = WebChatChannel(
                    channel_name=self.config.channels.chat.channel_name,
                )
                plugin_channels.append(self.web_chat_channel)
            self.channel_host = await start_channels(
                self.config,
                bus=self.bus,
                session_manager=self.session_manager,
                push_tool=self.push_tool,
                http_resources=self.http_resources,
                event_bus=event_bus,
                telegram_bot_commands=(
                    plugin_manager.telegram_bot_commands
                    if plugin_manager
                    else None
                ),
                mobile_bot_commands=(
                    plugin_manager.mobile_bot_commands
                    if plugin_manager
                    else None
                ),
                interrupt_controller=self.conversation_runtime,
                plugin_channels=plugin_channels,
            )
            await self.channel_host.start_all()
            if self.readiness is not None:
                self.readiness.mark_stage("channels.ready")
            if plugin_manager is not None:
                channel_bindings = {
                    plugin_id: generation.contributions.channels
                    for plugin_id, generation in plugin_manager.current_snapshot.generations.items()
                } if plugin_manager.current_snapshot is not None else {}
                self.channel_host.bind_plugin_channels(channel_bindings)
                plugin_manager.bind_channel_switcher(
                    self.channel_host.swap_plugin_channels
                )
                plugin_manager.bind_endpoint_switcher(
                    self._swap_plugin_endpoints
                )

            self.tasks = [
                self.passive_worker.run(),
                self.bus.dispatch_outbound(),
                self.scheduler.run(),
            ]
            if plugin_manager is not None:
                assert self.core.plugin_manager is not None
                llm = self.core.plugin_manager.llm
                if llm is not None:
                    self.plugin_job_runtime = PluginJobRuntime(
                        event_bus=event_bus,
                        llm=llm,
                        snapshot_store=plugin_manager.snapshot_store,
                    )
                    self.tasks.append(self.plugin_job_runtime.run())
            optimizer_tasks, self._memory_optimizer = build_memory_optimizer_task(
                self.config,
                provider=self.provider,
                memory_store=self.memory_runtime.markdown.store,
            )
            self.tasks.extend(optimizer_tasks)
            self.dashboard_server = build_dashboard_server(
                workspace=self.workspace,
                host=self.dashboard_host,
                port=self.dashboard_port,
                manual_consolidator=self.agent_loop,
                manual_memory_optimizer=self._memory_optimizer,
                memory_admin=self.memory_runtime.engine,
                memory_store=self.memory_runtime.markdown.store,
                plugin_manager=plugin_manager,
            )
            self.dashboard_task = asyncio.create_task(
                self.dashboard_server.serve(),
                name="dashboard_server",
            )
            if self.config.mobile_realtime.enabled:
                from infra.mobile_realtime.gateway import (
                    build_mobile_gateway_server,
                )

                assert self.mobile_gateway_runtime is not None
                mobile_keyset = self.mobile_gateway_runtime.keyset
                self.mobile_gateway_server = build_mobile_gateway_server(
                    self.mobile_gateway_runtime,
                    mobile_keyset,
                )
                self.mobile_gateway_task = asyncio.create_task(
                    self.mobile_gateway_server.serve(),
                    name="mobile_gateway_server",
                )
            if self.web_chat_channel is not None:
                self.chat_server = build_chat_server(
                    workspace=self.workspace,
                    channel=self.web_chat_channel,
                    host=self.config.channels.chat.host,
                    port=self.config.channels.chat.port,
                    mobile_pairing_admin=(
                        self.mobile_gateway_runtime.admin
                        if self.mobile_gateway_runtime is not None
                        else None
                    ),
                )
                self.chat_task = asyncio.create_task(
                    self.chat_server.serve(),
                    name="chat_server",
                )
            proactive_tasks, self.proactive_loop = build_proactive_runtime(
                self.config,
                self.workspace,
                session_manager=self.session_manager,
                provider=self.provider,
                push_tool=self.push_tool,
                memory_store=self.memory_runtime,
                presence=self.presence,
                agent_loop=self.agent_loop,
                event_bus=event_bus,
                tool_hooks=list(plugin_manager.tool_hooks) if plugin_manager else None,
                proactive_modules=(
                    list(plugin_manager.proactive_modules)
                    if plugin_manager
                    else None
                ),
                proactive_lifecycles=(
                    list(plugin_manager.proactive_lifecycles)
                    if plugin_manager
                    else None
                ),
                proactive_module_factories=(
                    list(plugin_manager.proactive_module_factories)
                    if plugin_manager
                    else None
                ),
                proactive_runtime_factories=(
                    list(plugin_manager.proactive_runtime_factories)
                    if plugin_manager
                    else None
                ),
                proactive_sources=(
                    list(plugin_manager.proactive_sources)
                    if plugin_manager
                    else None
                ),
                runtime_snapshot_store=(
                    plugin_manager.snapshot_store if plugin_manager else None
                ),
            )
            self.tasks.extend(proactive_tasks)
            if self.proactive_loop is not None:
                if plugin_manager is not None:
                    plugin_manager.bind_endpoint_admission(
                        quiesce=self.proactive_loop.quiesce_for_reload,
                        resume=self.proactive_loop.resume_after_reload,
                    )

            if plugin_manager is not None:
                mobile_ui_refresh = (
                    self.mobile_gateway_runtime.channel.refresh_mobile_ui_catalog
                    if self.mobile_gateway_runtime is not None
                    else None
                )
                self.plugin_watcher = PluginWatcher(
                    plugin_manager,
                    baseline_revision="",
                    after_reconcile=mobile_ui_refresh,
                )
                self.plugin_watcher_task = asyncio.create_task(
                    self.plugin_watcher.run(),
                    name="plugin_watcher",
                )
                self.plugin_watcher.wake()

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
            self._primary_task = asyncio.create_task(
                _run_primary_tasks(runtime_tasks),
                name="primary_runtime",
            )
            self._runtime_tasks.clear()
            watched_tasks = {
                task
                for task in (
                    self.dashboard_task,
                    self.chat_task,
                    self.mobile_gateway_task,
                    self.plugin_watcher_task,
                    self.workspace_mcp_watcher_task,
                )
                if task is not None
            }
            supervised_tasks = {self._primary_task, *watched_tasks}

            # runtime task 获得一次调度机会后仍存活，才对外发布 ready。
            done, _ = await asyncio.wait(supervised_tasks, timeout=0)
            if not done:
                if self.readiness is not None:
                    self.readiness.mark_ready()
                done, _ = await asyncio.wait(
                    supervised_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            if self._primary_task in done:
                await self._primary_task
            else:
                if self.dashboard_task is not None and self.dashboard_task in done:
                    watched_task = self.dashboard_task
                    self.dashboard_task = None
                elif self.chat_task is not None and self.chat_task in done:
                    watched_task = self.chat_task
                    self.chat_task = None
                elif (
                    self.mobile_gateway_task is not None
                    and self.mobile_gateway_task in done
                ):
                    watched_task = self.mobile_gateway_task
                    self.mobile_gateway_task = None
                elif (
                    self.plugin_watcher_task is not None
                    and self.plugin_watcher_task in done
                ):
                    watched_task = self.plugin_watcher_task
                    self.plugin_watcher_task = None
                else:
                    assert self.workspace_mcp_watcher_task is not None
                    watched_task = self.workspace_mcp_watcher_task
                    self.workspace_mcp_watcher_task = None
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
            self.tasks = pending[len(scheduled):]
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
        if self.chat_server is not None:
            self.chat_server.should_exit = True
        if self.mobile_gateway_server is not None:
            self.mobile_gateway_server.should_exit = True

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
                    "chat_server.wait",
                    _wait_server_task(self.chat_task),
                ),
                (
                    "mobile_gateway_server.wait",
                    _wait_server_task(self.mobile_gateway_task),
                ),
                ("message_bus.aclose", _close_message_bus(self.bus)),
                (
                    "mobile_gateway.close",
                    _close_mobile_gateway(self.mobile_gateway_runtime),
                ),
                (
                    "plugin_watcher.stop",
                    _stop_plugin_watcher(
                        self.plugin_watcher,
                        self.plugin_watcher_task,
                    ),
                ),
                (
                    "proactive.stop",
                    _stop_proactive(self.proactive_loop),
                ),
                (
                    "plugin_jobs.stop",
                    _stop_plugin_jobs(self.plugin_job_runtime),
                ),
                (
                    "app_server.stop",
                    self.app_server.stop if self.app_server else _noop_async,
                ),
                (
                    "control_service.shutdown",
                    self.control_service.shutdown
                    if self.control_service
                    else _noop_async,
                ),
                (
                    "conversation_runtime.shutdown",
                    self.conversation_runtime.shutdown
                    if self.conversation_runtime
                    else _noop_async,
                ),
                (
                    "channels.stop",
                    self.channel_host.stop_all if self.channel_host else _noop_async,
                ),
                (
                    "plugin_services.stop",
                    self.plugin_service_host.stop_all
                    if self.plugin_service_host
                    else _noop_async,
                ),
                ("core.stop", self.core.stop if self.core else _noop_async),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose
                    if self.memory_runtime
                    else _noop_async,
                ),
                ("http_resources.aclose", self.http_resources.aclose),
                (
                    "runtime_readiness.clear",
                    _clear_readiness(self.readiness),
                ),
                ("workspace_lock.release", _release_workspace_lock(self._workspace_lock)),
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

    async def _uninstall_plugin(self, plugin_id: str) -> dict[str, object]:
        """Disable, drain, and remove plugin code while retaining workspace data."""

        # 1. 先更新安装清单，新请求不再取得该插件 generation。
        plugin_id = plugin_id.strip()
        if not plugin_id:
            raise ValueError("缺少插件 ID")
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")
        from agent.plugins.install import (
            finalize_uninstall_plugin,
            set_installed_plugin_enabled,
        )

        _ = set_installed_plugin_enabled(
            plugin_id,
            enabled=False,
            plugins_home=manager.installed_plugins_home,
        )

        # 2. 等待旧 turn 释放 lease，再删除 cache 和 manifest entry。
        await manager.reconcile_disabled_and_drain(plugin_id)
        cache_path, data_path = finalize_uninstall_plugin(
            plugin_id,
            workspace=self.workspace,
            plugins_home=manager.installed_plugins_home,
        )
        return {
            "pluginId": plugin_id,
            "cachePath": str(cache_path),
            "dataPath": str(data_path),
        }

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
        plugin_id: str,
        old_services: dict[str, dict[str, Any]],
        new_services: dict[str, dict[str, Any]],
        old_channels: tuple[Any, ...],
        new_channels: tuple[Any, ...],
    ) -> None:
        assert self.channel_host is not None
        assert self.plugin_service_host is not None
        swap = (
            self.channel_host.prepare_plugin_swap(
                plugin_id,
                old_channels,
                new_channels,
            )
            if old_channels != new_channels
            else None
        )
        if swap is not None:
            await self.channel_host.stop_plugin_swap(swap)
        services_switched = False
        try:
            if old_services != new_services:
                await self.plugin_service_host.swap_plugin_services(
                    plugin_id,
                    old_services,
                    new_services,
                )
                services_switched = True
            if swap is not None:
                await self.channel_host.start_plugin_swap(swap)
        except BaseException as error:
            service_restore_error: BaseException | None = None
            if services_switched:
                try:
                    await self.plugin_service_host.swap_plugin_services(
                        plugin_id,
                        new_services,
                        old_services,
                    )
                except BaseException as restore_error:
                    service_restore_error = restore_error
            channel_restore_error: BaseException | None = None
            if swap is not None:
                try:
                    await self.channel_host.restore_plugin_swap(swap)
                except BaseException as restore_error:
                    channel_restore_error = restore_error
            if service_restore_error is not None or channel_restore_error is not None:
                details: list[str] = []
                if service_restore_error is not None:
                    details.append(f"managed service: {service_restore_error}")
                if channel_restore_error is not None:
                    details.append(f"Channel: {channel_restore_error}")
                raise RuntimeError(
                    "插件旧端点恢复失败: " + "; ".join(details)
                ) from error
            raise
        if swap is not None:
            self.channel_host.commit_plugin_swap(swap)


def build_app_runtime(
    config: Config,
    workspace: Path,
    *,
    restart_coordinator: RestartCoordinator | None = None,
    readiness: RuntimeReadiness | None = None,
) -> AppRuntime:
    return AppRuntime(
        config,
        workspace,
        restart_coordinator=restart_coordinator,
        readiness=readiness,
    )
