from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent.plugin_composition.model import CompositionError


@dataclass(frozen=True, slots=True)
class DashboardContext:
    """向一个 v3 Dashboard 暴露 Core 分配的窄运行边界。"""

    plugin_id: str
    plugin_dir: Path
    data_root: Path
    validation: bool
    _workspace_roots: tuple[tuple[str, Path], ...] = field(
        default=(),
        repr=False,
    )
    _workspace_files: tuple[tuple[str, Path], ...] = field(
        default=(),
        repr=False,
    )

    def workspace_root(self, name: str) -> Path:
        """返回与当前 Dashboard generation 相同的声明式 workspace root。"""

        for declared, path in self._workspace_roots:
            if declared == name:
                return path
        raise CompositionError(
            "WORKSPACE_ROOT_UNDECLARED",
            f"{self.plugin_id} 未声明 workspace root: {name}",
        )

    def workspace_file(self, name: str) -> Path:
        """返回当前 Dashboard generation 声明过的 workspace 顶层文件。"""

        for declared, path in self._workspace_files:
            if declared == name:
                return path
        raise CompositionError(
            "WORKSPACE_FILE_UNDECLARED",
            f"{self.plugin_id} 未声明 workspace file: {name}",
        )
