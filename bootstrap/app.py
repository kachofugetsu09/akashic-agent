from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Awaitable, Callable

from agent.config_models import Config
from bootstrap.channel_host import ChannelHost
from bootstrap.channels import start_channels
from bootstrap.chat_api import build_chat_server
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime
from bootstrap.tools import CoreRuntime, build_core_runtime
from bus.event_bus import EventBus
from agent.plugins.jobs import PluginJobRuntime
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _run_cleanup_steps(*steps: tuple[str, Callable[[], Awaitable[None]]]) -> None:
    first_error: Exception | None = None
    for name, step in steps:
        try:
            await step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.warning("shutdown step failed: %s: %s", name, exc)
    if first_error is not None:
        raise first_error


async def _noop_async() -> None:
    return None


def _stop_plugin_jobs(runtime: PluginJobRuntime | None) -> Callable[[], Awaitable[None]]:
    async def stop() -> None:
        if runtime is not None:
            runtime.stop()

    return stop


class AppRuntime:
    def __init__(self, config: Config, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self.http_resources = SharedHttpResources()
        self.ipc = None
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
        self.mcp_registry = None
        self.memory_runtime = None
        self.presence = None
        self.proactive_loop = None
        self.peer_process_manager = None
        self.peer_poller = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.chat_server = None
        self.chat_task: asyncio.Task[None] | None = None
        self.web_chat_channel = None
        self.plugin_job_runtime: PluginJobRuntime | None = None
        self.tasks: list[Awaitable[None]] = []
        self._memory_optimizer = None
        self._shutdown = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        configure_default_shared_http_resources(self.http_resources)
        try:
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
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
            self.mcp_registry = self.core.mcp_registry
            self.memory_runtime = self.core.memory_runtime
            self.presence = self.core.presence
            self.peer_process_manager = self.core.peer_process_manager
            self.peer_poller = self.core.peer_poller
            await self.core.start()

            plugin_manager = getattr(self.core, "plugin_manager", None)
            plugin_channels = list(plugin_manager.channels) if plugin_manager else []
            if self.config.channels.chat.enabled:
                from infra.channels.web_chat_channel import WebChatChannel

                self.web_chat_channel = WebChatChannel(
                    channel_name=self.config.channels.chat.channel_name,
                )
                plugin_channels.append(self.web_chat_channel)
            self.ipc, self.channel_host = await start_channels(
                self.config,
                bus=self.bus,
                session_manager=self.session_manager,
                push_tool=self.push_tool,
                http_resources=self.http_resources,
                event_bus=event_bus,
                bot_commands=(
                    plugin_manager.telegram_bot_commands
                    if plugin_manager
                    else None
                ),
                interrupt_controller=self.agent_loop,
                plugin_channels=plugin_channels,
            )
            await self.channel_host.start_all()

            self.tasks = [
                self.agent_loop.run(),
                self.bus.dispatch_outbound(),
                self.scheduler.run(),
            ]
            plugin_jobs = plugin_manager.jobs if plugin_manager else []
            if plugin_jobs:
                assert self.core.plugin_manager is not None
                llm = self.core.plugin_manager.llm
                if llm is not None:
                    self.plugin_job_runtime = PluginJobRuntime(
                        event_bus=event_bus,
                        llm=llm,
                        jobs=plugin_jobs,
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
                manual_consolidator=self.agent_loop,
                manual_memory_optimizer=self._memory_optimizer,
                memory_admin=self.memory_runtime.engine,
                memory_store=self.memory_runtime.markdown.store,
            )
            self.dashboard_task = asyncio.create_task(
                self.dashboard_server.serve(),
                name="dashboard_server",
            )
            if self.web_chat_channel is not None:
                self.chat_server = build_chat_server(
                    workspace=self.workspace,
                    channel=self.web_chat_channel,
                    host=self.config.channels.chat.host,
                    port=self.config.channels.chat.port,
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
            )
            self.tasks.extend(proactive_tasks)
            if self.proactive_loop is not None:
                self.ipc.set_proactive_loop(self.proactive_loop)

            self._started = True
        except Exception:
            await self.shutdown()
            raise

    async def run(self) -> None:
        try:
            await self.start()
            await asyncio.gather(*self.tasks)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self.dashboard_server is not None:
                self.dashboard_server.should_exit = True
            if self.chat_server is not None:
                self.chat_server.should_exit = True
            if self.dashboard_task is not None:
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            if self.chat_task is not None:
                try:
                    await self.chat_task
                except asyncio.CancelledError:
                    pass
            await _run_cleanup_steps(
                (
                    "plugin_jobs.stop",
                    _stop_plugin_jobs(self.plugin_job_runtime),
                ),
                ("core.stop", self.core.stop if self.core else _noop_async),
                ("ipc.stop", self.ipc.stop if self.ipc else _noop_async),
                (
                    "channels.stop",
                    self.channel_host.stop_all if self.channel_host else _noop_async,
                ),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose if self.memory_runtime else _noop_async,
                ),
                ("http_resources.aclose", self.http_resources.aclose),
            )
        finally:
            clear_default_shared_http_resources(self.http_resources)


def build_app_runtime(config: Config, workspace: Path | None = None) -> AppRuntime:
    return AppRuntime(config, workspace or (Path.home() / ".akashic" / "workspace"))
