from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from pathlib import Path
from typing import Awaitable, Callable

from agent.config_models import Config
from bootstrap.channels import start_channels
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.memory import build_memory_runtime
from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime
from bootstrap.providers import build_providers
from bootstrap.tools import CoreRuntime, build_core_runtime
from bus.event_bus import EventBus
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from core.observe.writer import TraceWriter
from core.observe.retention import run_retention_if_needed

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


class AppRuntime:
    def __init__(self, config: Config, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self.http_resources = SharedHttpResources()
        self.ipc = None
        self.tg_channel = None
        self.qq_channel = None
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
        self.observe_writer: TraceWriter | None = None
        self.observe_task: asyncio.Task[None] | None = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.tasks: list[Awaitable[None]] = []
        self._shutdown = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        configure_default_shared_http_resources(self.http_resources)
        try:
            # 初始化 observe writer
            observe_db_path = self.workspace / "observe" / "observe.db"
            self.observe_writer = TraceWriter(observe_db_path)
            asyncio.create_task(
                run_retention_if_needed(observe_db_path), name="observe_retention"
            )

            build_core_kwargs = {}
            if "observe_writer" in inspect.signature(build_core_runtime).parameters:
                build_core_kwargs["observe_writer"] = self.observe_writer
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
                **build_core_kwargs,
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

            self.ipc, self.tg_channel, self.qq_channel = await start_channels(
                self.config,
                bus=self.bus,
                session_manager=self.session_manager,
                push_tool=self.push_tool,
                http_resources=self.http_resources,
                event_bus=event_bus,
                bot_commands=(
                    self.core.plugin_manager.telegram_bot_commands
                    if self.core.plugin_manager
                    else None
                ),
                interrupt_controller=self.agent_loop,
            )

            self.tasks = [
                self.agent_loop.run(),
                self.bus.dispatch_outbound(),
                self.scheduler.run(),
            ]
            if self.observe_writer is not None:
                self.observe_task = asyncio.create_task(
                    self.observe_writer.run(), name="observe_writer"
                )
            self.dashboard_server = build_dashboard_server(
                workspace=self.workspace,
                manual_consolidator=self.agent_loop,
            )
            self.dashboard_task = asyncio.create_task(
                self.dashboard_server.serve(),
                name="dashboard_server",
            )
            plugin_manager = getattr(self.core, "plugin_manager", None)
            proactive_tasks, self.proactive_loop = build_proactive_runtime(
                self.config,
                self.workspace,
                session_manager=self.session_manager,
                provider=self.provider,
                light_provider=self.light_provider,
                push_tool=self.push_tool,
                memory_store=self.memory_runtime.facade,
                presence=self.presence,
                agent_loop=self.agent_loop,
                observe_writer=self.observe_writer,
                tool_hooks=list(plugin_manager.tool_hooks) if plugin_manager else None,
            )
            self.tasks.extend(proactive_tasks)
            if self.proactive_loop is not None:
                self.ipc.set_proactive_loop(self.proactive_loop)

            self.tasks.extend(
                build_memory_optimizer_task(
                    self.config,
                    provider=self.provider,
                    memory_store=(
                        self.memory_runtime.profile_maint or self.memory_runtime.port
                    ),
                )
            )

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
            if self.dashboard_task is not None:
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            await _run_cleanup_steps(
                ("core.stop", self.core.stop if self.core else _noop_async),
                ("ipc.stop", self.ipc.stop if self.ipc else _noop_async),
                (
                    "telegram.stop",
                    self.tg_channel.stop if self.tg_channel else _noop_async,
                ),
                ("qq.stop", self.qq_channel.stop if self.qq_channel else _noop_async),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose if self.memory_runtime else _noop_async,
                ),
                ("http_resources.aclose", self.http_resources.aclose),
            )
            if self.observe_task is not None:
                _ = self.observe_task.cancel()
                try:
                    await self.observe_task
                except asyncio.CancelledError:
                    pass
        finally:
            clear_default_shared_http_resources(self.http_resources)


def build_app_runtime(config: Config, workspace: Path | None = None) -> AppRuntime:
    return AppRuntime(config, workspace or (Path.home() / ".akashic" / "workspace"))
