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
    from bus.publisher import EventPublisher

# TODO(memory-engine-cleanup): 旧测试 monkeypatch 完成迁移后删除这个占位名。
PostResponseMemoryWorker = object


def build_memory_runtime(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    event_publisher: "EventPublisher | None" = None,
) -> MemoryRuntime:
    from agent.tools.filesystem import EditFileTool, WriteFileTool
    from agent.tools.forget_memory import ForgetMemoryTool
    from agent.tools.memorize import MemorizeTool
    from agent.tools.recall_memory import RecallMemoryTool
    from bootstrap.wiring import MemoryEngineBuildDeps, resolve_memory_engine_builder

    engine_builder = resolve_memory_engine_builder(
        getattr(getattr(config, "wiring", None), "memory_engine", "default")
    )
    engine = cast(
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

    memory_tools = {}
    if config.memory_v2.enabled:
        memory_tools = {
            "memorize_tool": MemorizeTool(engine),
            "forget_tool": ForgetMemoryTool(engine),
            "recall_tool": RecallMemoryTool(facade=engine),
        }

    register_memory_meta_tools(
        tools,
        **memory_tools,
        write_file_tool=WriteFileTool(),
        edit_file_tool=EditFileTool(),
    )
    return MemoryRuntime(
        engine=engine,
        closeables=list(getattr(engine, "closeables", [])),
    )
