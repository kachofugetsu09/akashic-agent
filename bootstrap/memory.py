from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.config_models import Config
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from core.memory.markdown import build_markdown_memory_runtime
from core.memory.optimizer import MemoryOptimizer, MemoryOptimizerLoop
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources

if TYPE_CHECKING:
    from bus.event_bus import EventBus
    from core.memory.markdown import MarkdownMemoryStore


# TODO(memory-plugin): Move this Markdown/PENDING/SELF owner into an ordinary
# lifecycle plugin. Keep it separate from embedded-memory providers such as Akasha.
def build_memory_runtime(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    event_publisher: "EventBus | None" = None,
) -> MemoryRuntime:
    markdown = build_markdown_memory_runtime(
        workspace=workspace,
        provider=provider,
        model=config.model,
        event_bus=event_publisher,
    )
    _ = tools, light_provider, http_resources
    return MemoryRuntime(markdown=markdown)


def build_memory_optimizer_task(
    config: Config,
    *,
    provider: LLMProvider,
    memory_store: "MarkdownMemoryStore",
) -> tuple[list, MemoryOptimizer | None]:
    if not config.memory_optimizer_enabled:
        print("MemoryOptimizerLoop 已禁用（memory_optimizer_enabled=false）")
        return [], None

    optimizer = MemoryOptimizer(
        memory=memory_store,
        provider=provider,
        model=config.model,
    )
    interval = config.memory_optimizer_interval_seconds
    print(f"MemoryOptimizerLoop 已启动，间隔={interval}s ({interval / 3600:.1f}h)")
    return [MemoryOptimizerLoop(optimizer, interval_seconds=interval).run()], optimizer
