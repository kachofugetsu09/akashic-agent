from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agent.config import resolve_app_server_endpoint
from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from agent.host_bridge.monitor import build_host_bridge_monitor
from agent.host_bridge.monitor import claim_host_bridge_boot
from agent.restart import RestartCoordinator
from agent.config_models import Config
from bootstrap.channel_host import ChannelHost
from bootstrap.channel_presentation import ChannelTurnPresentationBridge
from bootstrap.channels import start_channels
from bootstrap.chat_api import build_chat_server
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.control_execution import execute_control_turn
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.memory import build_memory_optimizer_task
from bootstrap.web_runtime import (
    chat_socket_path,
    dashboard_socket_path,
    prepare_runtime_socket,
)
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.passive_worker import PassiveMessageWorker
from bootstrap.tools import CoreRuntime, build_core_runtime
from bootstrap.workspace_lock import WorkspaceInstanceLock
from bootstrap.workspace_token import ensure_workspace_token
from bus.event_bus import EventBus
from bus.queue import MessageBus
from agent.plugins.turn_rollout import TurnPluginRollout
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
        self.http_resources = SharedHttpResources()
        self.app_server: SocketAppServer | None = None
        self.conversation_runtime: ConversationRuntime | None = None
        self.control_service: ControlService | None = None
        self.plugin_turn_rollout: TurnPluginRollout | None = None
        self.passive_worker: PassiveMessageWorker | None = None
        self.channel_host: ChannelHost | None = None
        self.channel_presentation: ChannelTurnPresentationBridge | None = None
        self.core: CoreRuntime | None = None
        self.agent_loop = None
        self.bus = None
        self.event_bus: EventBus | None = None
        self.tools = None
        self.push_tool = None
        self.session_manager = None
        self.memory_runtime = None
        self.presence = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.chat_server = None
        self.chat_task: asyncio.Task[None] | None = None
        self.web_chat_channel = None
        self.mobile_gateway_runtime = None
        self.mobile_gateway_server = None
        self.mobile_gateway_task: asyncio.Task[None] | None = None
        self.plugin_watcher: PluginWatcher | None = None
        self.plugin_watcher_task: asyncio.Task[None] | None = None
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
            claim = await claim_host_bridge_boot()
            if claim is not None and self.readiness is not None:
                self.readiness.mark_stage("host_bridge.owner")
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
            self.channel_presentation = ChannelTurnPresentationBridge(event_bus)
            self.tools = self.core.tools
            self.push_tool = self.core.push_tool
            self.session_manager = self.core.session_manager
            self.memory_runtime = self.core.memory_runtime
            self.presence = self.core.presence

            async def _execute_control_request(request: TurnRequest):
                assert self.agent_loop is not None
                return await execute_control_turn(
                    self.agent_loop,
                    event_bus,
                    request,
                )

            manager = getattr(self.core, "plugin_manager", None)
            if manager is None:
                await self.core.start()
                raise RuntimeError("插件 Runtime 不可用")
            self.plugin_turn_rollout = TurnPluginRollout(
                manager,
                workspace=self.workspace,
                uninstall=self._uninstall_plugin,
            )
            assert self.agent_loop is not None
            self.agent_loop.bind_plugin_rollout_fact_provider(
                self.plugin_turn_rollout.consume_fact
            )
            self.conversation_runtime = ConversationRuntime(
                self.session_manager.control_store,
                _execute_control_request,
                restart_coordinator=self.restart_coordinator,
                turn_terminal=self.plugin_turn_rollout.turn_terminal,
            )
            if self.restart_coordinator is not None:
                self.restart_coordinator.bind_admission(
                    quiesce=self.conversation_runtime.quiesce_for_restart,
                    resume=self.conversation_runtime.resume_after_restart_cancel,
                )
            self.core.bind_conversation_runtime(self.conversation_runtime)
            await self.core.start()
            if self.readiness is not None:
                self.readiness.mark_stage("core.ready")
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
                plugin_uninstall_register=self._register_plugin_uninstall,
                plugin_install=self._install_plugin,
                plugin_revert=self._revert_plugin_operation,
                plugin_turn_barrier=self.plugin_turn_rollout.wait_for_turn_boundary,
                plugin_child_binding=lambda capability, consume: (
                    self.plugin_turn_rollout.child_binding(
                        capability,
                        consume,
                    )
                    if self.plugin_turn_rollout is not None
                    else None
                ),
                plugin_status=self._plugin_status,
                plugin_promote=self._promote_plugin,
                plugin_discard=self._discard_plugin,
                workspace_token=workspace_token,
                restart_coordinator=self.restart_coordinator,
                boot_id=self.readiness.boot_id if self.readiness else None,
                ready=(lambda: self.readiness.ready) if self.readiness else None,
            )
            channel_attachment_store = self.core.channel_attachment_store
            if channel_attachment_store is None:
                raise RuntimeError("Core channel attachment store 未初始化")
            self.passive_worker = PassiveMessageWorker(
                self.bus,
                self.conversation_runtime,
                self.agent_loop,
                attachment_store=channel_attachment_store,
                channel_dispatcher=(
                    lambda message, passive: self.push_tool.dispatch(
                        message,
                        commit_role="passive" if passive else "",
                    )
                ),
            )

            # 1. v3 control 始终通过 exact Channel binding 中断并回送收据。
            async def interrupt_v3_channel(raw: Any) -> str:
                assert self.conversation_runtime is not None
                result = self.conversation_runtime.request_interrupt(
                    raw.session_key,
                    sender=raw.message.sender,
                    command=raw.message.content,
                )
                return result.status

            async def dispatch_v3_control_response(
                envelope: Any,
                binding: Any,
            ) -> Any:
                return await self.bus.publish_channel_outbound_awaited(
                    envelope,
                    binding,
                    passive=True,
                )

            manager.channel_generation_host.bind_control_interrupter(
                interrupt_v3_channel
            )
            manager.channel_generation_host.bind_control_response_dispatcher(
                dispatch_v3_control_response
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
            from infra.mobile_realtime.runtime_inspection import (
                RuntimeInspectionService,
            )

            runtime_inspection = RuntimeInspectionService(
                workspace=self.workspace,
                snapshot_store=(
                    plugin_manager.snapshot_store
                    if plugin_manager is not None
                    else None
                ),
            )
            plugin_ui_provider = None
            web_ui_provider = None
            model_catalog_reader = None
            model_control = None
            if plugin_manager is not None:
                from agent.plugins.mobile_ui import PluginMobileUiProvider
                from agent.plugins.model_control import RuntimeModelControl
                from agent.plugins.web_ui import PluginWebUiProvider

                plugin_ui_provider = PluginMobileUiProvider(plugin_manager)
                web_ui_provider = PluginWebUiProvider(plugin_manager.snapshot_store)
                model_control = RuntimeModelControl(plugin_manager.snapshot_store)
                model_catalog_reader = model_control.catalog
            if self.config.mobile_realtime.enabled:
                from infra.mobile_realtime.gateway import (
                    build_mobile_gateway_runtime,
                )

                self.mobile_gateway_runtime, _ = build_mobile_gateway_runtime(
                    self.config.mobile_realtime,
                    self.workspace,
                )
                self.bus.bind_durable_inbound_store(self.session_manager.control_store)
                self.mobile_gateway_runtime.channel.bind_runtime_inspection(
                    runtime_inspection
                )
                self.mobile_gateway_runtime.channel.bind_channel_attachment_store(
                    channel_attachment_store
                )
                if model_catalog_reader is None:
                    raise RuntimeError("Mobile Gateway 启动需要模型目录")
                self.mobile_gateway_runtime.channel.bind_model_catalog(
                    model_catalog_reader
                )
                if plugin_ui_provider is not None:
                    self.mobile_gateway_runtime.channel.bind_mobile_ui_provider(
                        plugin_ui_provider
                    )
            extra_channels = []
            if (
                self.config.channels.chat.enabled
                or self.mobile_gateway_runtime is not None
            ):
                from infra.channels.akashic_channel import AkashicChannel

                if self.config.channels.chat.enabled:
                    from infra.channels.web_chat_channel import WebChatChannel

                    self.web_chat_channel = WebChatChannel()
                    self.web_chat_channel.bind_artifact_store(channel_attachment_store)
                extra_channels.append(
                    AkashicChannel(
                        web=self.web_chat_channel,
                        mobile=(
                            None
                            if self.mobile_gateway_runtime is None
                            else self.mobile_gateway_runtime.channel
                        ),
                    )
                )
            self.channel_host = await start_channels(
                self.config,
                bus=self.bus,
                session_manager=self.session_manager,
                push_tool=self.push_tool,
                http_resources=self.http_resources,
                event_bus=event_bus,
                telegram_command_catalog_provider=(
                    plugin_manager.stable_telegram_command_catalog
                    if plugin_manager is not None
                    else None
                ),
                mobile_command_catalog_provider=(
                    plugin_manager.stable_mobile_command_catalog
                    if plugin_manager is not None
                    else None
                ),
                interrupt_controller=self.conversation_runtime,
                extra_channels=extra_channels,
            )
            if plugin_manager is not None:
                from bootstrap.core_channel_adapter import build_core_channel_definition

                await plugin_manager.bind_core_channel_definitions(
                    tuple(
                        build_core_channel_definition(channel)
                        for channel in self.channel_host.channels
                    )
                )
            await self.channel_host.start_all()
            if self.readiness is not None:
                self.readiness.mark_stage("channels.ready")
            if plugin_manager is None:
                raise RuntimeError("插件 Runtime 不可用")
            plugin_manager.bind_endpoint_switcher(self._swap_plugin_endpoints)

            self.tasks = [
                self.passive_worker.run(),
                self.bus.dispatch_outbound(),
                plugin_manager.run_runtime_services(),
            ]
            host_bridge_monitor = build_host_bridge_monitor()
            if host_bridge_monitor is not None:
                self.tasks.append(host_bridge_monitor)
            optimizer_tasks, self._memory_optimizer = build_memory_optimizer_task(
                self.config,
                runtime_snapshot_store=plugin_manager.snapshot_store,
                memory_store=self.memory_runtime.markdown.store,
            )
            self.tasks.extend(optimizer_tasks)
            self.dashboard_server = build_dashboard_server(
                workspace=self.workspace,
                uds=prepare_runtime_socket(dashboard_socket_path(self.workspace)),
                manual_memory_optimizer=self._memory_optimizer,
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
                    uds=prepare_runtime_socket(chat_socket_path(self.workspace)),
                    mobile_pairing_admin=(
                        self.mobile_gateway_runtime.admin
                        if self.mobile_gateway_runtime is not None
                        else None
                    ),
                    runtime_inspection=runtime_inspection,
                    plugin_ui_provider=plugin_ui_provider,
                    web_ui_provider=web_ui_provider,
                    model_catalog_reader=model_catalog_reader,
                    model_control=model_control,
                )
                self.chat_task = asyncio.create_task(
                    self.chat_server.serve(),
                    name="chat_server",
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
                (
                    "conversation_runtime.shutdown",
                    (
                        self.conversation_runtime.shutdown
                        if self.conversation_runtime
                        else _noop_async
                    ),
                ),
                (
                    "channel_presentation.aclose",
                    (
                        self.channel_presentation.aclose
                        if self.channel_presentation
                        else _noop_async
                    ),
                ),
                (
                    "plugin_turn_rollout.shutdown",
                    (
                        self.plugin_turn_rollout.shutdown
                        if self.plugin_turn_rollout
                        else _noop_async
                    ),
                ),
                (
                    "channels.stop",
                    self.channel_host.stop_all if self.channel_host else _noop_async,
                ),
                (
                    "mobile_gateway.close",
                    _close_mobile_gateway(self.mobile_gateway_runtime),
                ),
                ("core.stop", self.core.stop if self.core else _noop_async),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose if self.memory_runtime else _noop_async,
                ),
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

    async def _install_plugin(
        self,
        source: str,
        marketplace: str,
        ref: str,
        sparse: list[str],
        owner_turn_id: str = "",
    ) -> dict[str, object]:
        """安装 immutable artifact，并等待 runtime latest 已可租用。"""

        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")

        # 1. PluginManager 与 watcher 共用一个 candidate 发布 owner。
        rollout = getattr(self, "plugin_turn_rollout", None)
        if rollout is None:
            result, status = await manager.install_candidate(
                source=source,
                marketplace=marketplace,
                ref_name=ref,
                sparse_paths=sparse,
            )
        elif not owner_turn_id:
            raise ValueError("plugin-install 必须由当前 active turn 发起")
        else:
            result, status = await rollout.install(
                owner_turn_id,
                source=source,
                marketplace=marketplace,
                ref_name=ref,
                sparse_paths=sparse,
            )

        # 2. 返回 manager 在 candidate owner 锁内冻结的发布结果。
        plugin_id = f"{result.plugin_name}@{result.marketplace}"
        publication = status["candidate_state"] if result.staged_candidate else "stable"
        message = (
            f"{plugin_id} 候选版本安装成功。当前 turn 仍使用原版本；"
            "本 turn 启动的 attached programmatic 验证会自动使用新版本。"
            "验证正确后请正常结束当前 turn，系统会在本轮结束后自动切换，"
            "下一 turn 生效；如果结果或轨迹不正确，请先执行 plugin-revert。"
            if result.staged_candidate
            else f"{plugin_id} 已经是当前安装版本；没有创建候选，也不需要重启。"
        )
        return {
            "pluginId": plugin_id,
            "version": result.plugin_version,
            "sourceRevision": result.source_revision,
            "installedPath": str(result.installed_path),
            "dataPath": str(result.data_path),
            "publicationState": publication,
            "candidate": self._plugin_status(status),
            "message": message,
        }

    async def _register_plugin_uninstall(
        self,
        plugin_id: str,
        owner_turn_id: str,
    ) -> dict[str, object]:
        rollout = self.plugin_turn_rollout
        if rollout is None:
            raise RuntimeError("插件 turn rollout owner 不可用")
        result = await rollout.uninstall(owner_turn_id, plugin_id)
        result["message"] = (
            f"{plugin_id} 卸载已确认。当前 turn 的已有操作可以完成；"
            "本轮结束后系统会自动停止插件并删除已安装代码，plugin-data 会保留。"
            "下一 turn 不再加载该插件。如需取消，请在本轮结束前执行 plugin-revert。"
        )
        return result

    async def _revert_plugin_operation(self, owner_turn_id: str) -> dict[str, object]:
        rollout = self.plugin_turn_rollout
        if rollout is None:
            raise RuntimeError("插件 turn rollout owner 不可用")
        result = await rollout.revert(owner_turn_id)
        result["message"] = (
            "已撤销当前 turn 最近一次插件操作；已发布版本和 plugin-data 均未改变。"
        )
        return result

    def _plugin_status(
        self,
        status: dict[str, object] | None = None,
    ) -> dict[str, object]:
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")
        resolved_status = manager.candidate_status() if status is None else status
        return {
            "stableSnapshotId": resolved_status["stable_snapshot_id"],
            "latestSnapshotId": resolved_status["latest_snapshot_id"],
            "candidatePluginId": resolved_status["candidate_plugin_id"],
            "candidateGenerationId": resolved_status["candidate_generation_id"],
            "candidateState": resolved_status["candidate_state"],
            "candidateRuntimeRevision": resolved_status["candidate_source_revision"],
            "candidateReloadTransactionId": resolved_status["candidate_reload_tx_id"],
            "candidateError": resolved_status["candidate_error"],
        }

    async def _promote_plugin(self, plugin_id: str) -> dict[str, object]:
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")
        return await manager.switch_ready(plugin_id)

    async def _discard_plugin(self, plugin_id: str) -> dict[str, object]:
        manager = getattr(self.core, "plugin_manager", None)
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")
        return await manager.drop_candidate(plugin_id)

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
        old_commands: tuple[tuple[str, str], ...],
        new_commands: tuple[tuple[str, str], ...],
    ) -> None:
        assert self.channel_host is not None
        if old_commands != new_commands:
            await self.channel_host.swap_command_catalog(
                old_commands,
                new_commands,
            )


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
