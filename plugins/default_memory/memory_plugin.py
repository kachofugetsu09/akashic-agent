from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.tools.forget_memory import ForgetMemoryTool
from agent.tools.memorize import MemorizeTool
from agent.tools.recall_memory import RecallMemoryTool
from core.memory.plugin import (
    MemoryPluginBuildDeps,
    MemoryPluginRuntime,
    MemoryToolBundle,
)
from plugins.default_memory.config import (
    ensure_default_memory_config_file,
    load_default_memory_config,
    resolve_memory_db_path,
)
from plugins.default_memory.engine import DefaultMemoryEngine


class MemoryPlugin:
    plugin_id = "default"

    def ensure_workspace_storage(
        self,
        *,
        config: Config,
        workspace: Path,
    ) -> list[tuple[Path, bool]]:
        _ = ensure_default_memory_config_file()
        default_config = load_default_memory_config()
        db_path = resolve_memory_db_path(
            workspace=workspace,
            default_config=default_config,
        )
        existed = db_path.exists()
        DefaultMemoryEngine.ensure_workspace_storage(
            default_config=default_config,
            workspace=workspace,
        )
        return [(db_path, existed)]

    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime:
        default_config = load_default_memory_config()
        engine = DefaultMemoryEngine(
            config=deps.config,
            default_config=default_config,
            workspace=deps.workspace,
            provider=deps.provider,
            light_provider=deps.light_provider,
            http_resources=deps.http_resources,
            event_publisher=deps.event_publisher,
        )
        return MemoryPluginRuntime(
            engine=engine,
            tools=MemoryToolBundle(
                recall_memory=RecallMemoryTool(facade=engine),
                memorize=MemorizeTool(engine),
                forget_memory=ForgetMemoryTool(engine),
            ),
            closeables=list(engine.closeables),
            admin=engine,
        )
