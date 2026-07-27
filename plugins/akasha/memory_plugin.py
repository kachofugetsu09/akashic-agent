"""Akasic Agent MemoryPlugin factory for Akasha V2."""

from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.plugins.manifest import (
    builtin_plugin_data_dir,
    ensure_workspace_plugin_data_dir,
)
from core.memory.plugin import MemoryPluginBuildDeps, MemoryPluginRuntime
from infra.persistence.json_store import atomic_write_text

from .config import (
    load_akasha_config,
    render_akasha_config,
    resolve_workspace_path,
)
from .engine import AkashaMemoryEngine


class MemoryPlugin:
    """Build the Akasha V2 runtime with host-owned infrastructure."""

    plugin_id = "akasha"

    def ensure_workspace_storage(
        self,
        *,
        config: Config,
        workspace: Path,
    ) -> list[tuple[Path, bool]]:
        """Create plugin config and report derived storage locations."""

        # 1. Create only the plugin data directory and versioned config.
        _ = config
        plugin_dir = builtin_plugin_data_dir("akasha", workspace)
        ensure_workspace_plugin_data_dir(plugin_dir, workspace)
        config_path = plugin_dir / "config.local.toml"
        if not config_path.exists():
            atomic_write_text(
                config_path,
                render_akasha_config(),
                domain="akasha-v2.config",
            )

        # 2. Report storage paths without creating invalid empty databases.
        akasha_config = load_akasha_config(config_path)
        paths = (
            resolve_workspace_path(workspace, akasha_config.db_path),
            resolve_workspace_path(workspace, akasha_config.index_path),
        )
        return [(path, path.exists()) for path in paths]

    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime:
        """Load configuration and return the complete plugin runtime."""

        # 1. Ensure and read the versioned plugin configuration.
        self.ensure_workspace_storage(
            config=deps.config,
            workspace=deps.workspace,
        )
        plugin_dir = builtin_plugin_data_dir(
            "akasha",
            deps.workspace,
        )
        akasha_config = load_akasha_config(
            plugin_dir / "config.local.toml"
        )

        # 2. Build the engine and expose its host lifecycle resources.
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
            embedding_api=engine.embedding_api,
        )
