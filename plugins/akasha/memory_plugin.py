from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from core.memory.plugin import MemoryPluginBuildDeps, MemoryPluginRuntime
from plugins.akasha.config import (
    ensure_akasha_config_file,
    load_akasha_config,
    resolve_akasha_db_path,
)
from plugins.akasha.engine import AkashaMemoryEngine


class MemoryPlugin:
    plugin_id = "akasha"

    # 准备 Akasha sidecar 存储。
    def ensure_workspace_storage(
        self,
        *,
        config: Config,
        workspace: Path,
    ) -> list[tuple[Path, bool]]:
        # 1. 确保插件配置存在，并按配置解析数据库路径。
        _ = config
        _ = ensure_akasha_config_file()
        akasha_config = load_akasha_config()
        db_path = resolve_akasha_db_path(
            workspace=workspace,
            akasha_config=akasha_config,
        )
        existed = db_path.exists()

        # 2. 创建 schema 后返回给启动日志展示。
        AkashaMemoryEngine.ensure_workspace_storage(
            akasha_config=akasha_config,
            workspace=workspace,
        )
        return [(db_path, existed)]

    # 构造 Akasha memory runtime。
    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime:
        # 1. Akasha 是独立 memory engine，不继承 default_memory 的 store/retriever。
        akasha_config = load_akasha_config()
        engine = AkashaMemoryEngine(
            config=deps.config,
            akasha_config=akasha_config,
            workspace=deps.workspace,
            http_resources=deps.http_resources,
            event_publisher=deps.event_publisher,
        )
        return MemoryPluginRuntime(
            engine=engine,
            closeables=list(engine.closeables),
            admin=engine,
        )
