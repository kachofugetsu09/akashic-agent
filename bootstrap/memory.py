from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from agent.config_models import Config
from agent.provider import LLMProvider
from agent.tools.meta import register_memory_meta_tools
from agent.tools.registry import ToolRegistry
from core.memory.engine import MemoryEngine
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources

if TYPE_CHECKING:
    from bus.event_bus import EventBus


# 统一 engine 构造入口，正常 runtime 和 dashboard 复用同一套 wiring。
def _build_memory_engine(
    *,
    config: Config,
    workspace: Path,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    event_publisher: "EventBus | None" = None,
) -> MemoryEngine:
    from bootstrap.wiring import MemoryEngineBuildDeps, resolve_memory_engine_builder

    # 1. 根据配置选择 engine builder。
    engine_builder = resolve_memory_engine_builder(
        getattr(getattr(config, "wiring", None), "memory_engine", "default")
    )

    # 2. builder 只接收 engine 自建依赖需要的基础资源。
    return cast(
        MemoryEngine,
        engine_builder(
            MemoryEngineBuildDeps(
                config=config,
                workspace=workspace,
                provider=provider,
                light_provider=light_provider,
                http_resources=http_resources,
                event_publisher=event_publisher,
            )
        ),
    )


def build_memory_runtime(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    event_publisher: "EventBus | None" = None,
) -> MemoryRuntime:
    from agent.tools.filesystem import EditFileTool, WriteFileTool
    from agent.tools.forget_memory import ForgetMemoryTool
    from agent.tools.memorize import MemorizeTool
    from agent.tools.recall_memory import RecallMemoryTool

    # 1. 先构造 engine，工具层只拿协议对象。
    engine = _build_memory_engine(
        config=config,
        workspace=workspace,
        provider=provider,
        light_provider=light_provider,
        http_resources=http_resources,
        event_publisher=event_publisher,
    )

    # 2. memory_v2 关闭时仍注册文件工具，但不暴露记忆读写工具。
    memorize_tool: MemorizeTool | None = None
    forget_tool: ForgetMemoryTool | None = None
    recall_tool: RecallMemoryTool | None = None
    if config.memory_v2.enabled:
        memorize_tool = MemorizeTool(engine)
        forget_tool = ForgetMemoryTool(engine)
        recall_tool = RecallMemoryTool(facade=engine)

    # 3. 工具执行时再通过 engine API 读写记忆。
    register_memory_meta_tools(
        tools,
        memorize_tool=memorize_tool,
        forget_tool=forget_tool,
        recall_tool=recall_tool,
        write_file_tool=WriteFileTool(),
        edit_file_tool=EditFileTool(),
    )
    return MemoryRuntime(
        engine=engine,
        closeables=list(getattr(engine, "closeables", [])),
    )


def build_memory_admin_runtime(
    config: Config,
    workspace: Path,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    event_publisher: "EventBus | None" = None,
) -> MemoryRuntime:
    # dashboard 不注册工具，只需要同一套 engine admin 能力和关闭生命周期。
    engine = _build_memory_engine(
        config=config,
        workspace=workspace,
        provider=provider,
        light_provider=light_provider,
        http_resources=http_resources,
        event_publisher=event_publisher,
    )
    return MemoryRuntime(
        engine=engine,
        closeables=[*list(getattr(engine, "closeables", [])), http_resources],
    )
