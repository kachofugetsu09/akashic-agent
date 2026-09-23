from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.plugin_composition.context import Fiber
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
    instance: object | None
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
    code_dir_path: Path | None = None
    fiber: Fiber | None = None
    load_error: BaseException | None = None

    @property
    def code_dir(self) -> Path:
        """Return the immutable archived code directory for this generation."""
        if self.code_dir_path is None:
            raise RuntimeError("generation 缺少固定归档代码目录")
        return self.code_dir_path.resolve()
