from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from agent.config_models import Config
from agent.provider import LLMProvider
from agent.tools.meta import register_memory_meta_tools
from agent.tools.registry import ToolRegistry
from core.memory.engine import MemoryEngine
from core.memory.markdown import build_markdown_memory_runtime
from core.memory.plugin import (
    DisabledMemoryEngine,
    MemoryPluginBuildDeps,
    MemoryPluginRuntime,
)
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources

if TYPE_CHECKING:
    from bus.event_bus import EventBus
    from core.memory.markdown import MarkdownMemoryRuntime


# 统一插件构造入口，正常 runtime 和 dashboard 复用同一套路由。
def _build_memory_plugin_runtime(
    *,
    config: Config,
    workspace: Path,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    markdown: "MarkdownMemoryRuntime",
    event_publisher: "EventBus | None" = None,
) -> MemoryPluginRuntime:
    from bootstrap.wiring import resolve_memory_plugin

    engine_name = (config.memory.engine or "").strip() or "default"
    plugin = resolve_memory_plugin(engine_name)
    return plugin.build(
        MemoryPluginBuildDeps(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            event_publisher=event_publisher,
            markdown=markdown,
        )
    )


def _memory_plugin_enabled(config: Config) -> bool:
    return bool(config.memory.enabled)


def ensure_memory_plugin_storage(
    config: Config,
    workspace: Path,
) -> list[tuple[Path, bool]]:
    if not _memory_plugin_enabled(config):
        return []
    engine_name = (config.memory.engine or "").strip() or "default"
    from bootstrap.wiring import resolve_memory_plugin

    plugin = resolve_memory_plugin(engine_name)
    initializer = getattr(plugin, "ensure_workspace_storage", None)
    if not callable(initializer):
        return []
    result = initializer(config=config, workspace=workspace)
    if isinstance(result, list):
        normalized: list[tuple[Path, bool]] = []
        for item in result:
            if isinstance(item, tuple) and len(item) == 2:
                raw_path, raw_existed = item
                path = Path(str(raw_path))
                normalized.append((path, bool(raw_existed)))
            elif isinstance(item, str | Path):
                path = Path(item)
                normalized.append((path, path.exists()))
        return normalized
    return []


def build_memory_runtime(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    event_publisher: "EventBus | None" = None,
) -> MemoryRuntime:
    # 1. markdown 是默认记忆层，任何 engine 都共用。
    markdown = build_markdown_memory_runtime(
        workspace=workspace,
        provider=provider,
        model=config.model,
        keep_count=_memory_keep_count(config.memory_window),
        event_bus=event_publisher,
        recent_context_provider=light_provider or provider,
        recent_context_model=config.light_model or config.model,
    )

    closeables: list[object] = []
    if _memory_plugin_enabled(config):
        plugin_runtime = _build_memory_plugin_runtime(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            markdown=markdown,
            event_publisher=event_publisher,
        )
        engine = plugin_runtime.engine
        closeables.extend(plugin_runtime.closeables)
        register_memory_meta_tools(
            tools,
            memorize_tool=plugin_runtime.tools.memorize,
            forget_tool=plugin_runtime.tools.forget_memory,
            recall_tool=plugin_runtime.tools.recall_memory,
        )
    else:
        engine = cast(MemoryEngine, DisabledMemoryEngine())

    return MemoryRuntime(
        markdown=markdown,
        engine=engine,
        closeables=closeables,
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
    markdown = build_markdown_memory_runtime(
        workspace=workspace,
        provider=provider,
        model=config.model,
        keep_count=_memory_keep_count(config.memory_window),
        event_bus=event_publisher,
        recent_context_provider=light_provider or provider,
        recent_context_model=config.light_model or config.model,
    )
    closeables: list[object] = [http_resources]
    if _memory_plugin_enabled(config):
        plugin_runtime = _build_memory_plugin_runtime(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            markdown=markdown,
            event_publisher=event_publisher,
        )
        engine = plugin_runtime.engine
        closeables[:0] = plugin_runtime.closeables
    else:
        engine = cast(MemoryEngine, DisabledMemoryEngine())
    return MemoryRuntime(
        markdown=markdown,
        engine=engine,
        closeables=closeables,
    )


def _memory_keep_count(window: int) -> int:
    aligned_window = max(6, ((max(1, window) + 5) // 6) * 6)
    return aligned_window // 2
