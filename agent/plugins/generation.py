from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.plugins.scope import PluginScope
    from agent.plugins.static_manifest import StaticPluginManifest
    from agent.plugins.snapshot import RuntimeSnapshot


@dataclass
class PluginGeneration:
    plugin_id: str
    generation_id: str
    module_path: str
    source_revision: str
    config_revision: str
    plugin_dir: Path
    data_dir: Path
    instance: object
    scope: PluginScope
    config_projection: dict[str, object] = field(default_factory=dict)
    source_type: Literal["builtin", "installed"] = "builtin"
    static_manifest: StaticPluginManifest | None = None
    runtime_snapshot: RuntimeSnapshot | None = None
    retire_started: bool = False
    state: str = "active"
    reload_tx_id: str | None = None
    validation_workspace: Path | None = None
    archive_ref: str | None = None

    @property
    def code_dir(self) -> Path:
        """代码和资源沿实际入口定位；plugin_dir 只记录安装来源。"""
        import sys

        module = sys.modules[self.module_path]
        if module.__file__ is None:
            raise RuntimeError("插件入口缺少文件路径")
        return Path(module.__file__).resolve().parent
