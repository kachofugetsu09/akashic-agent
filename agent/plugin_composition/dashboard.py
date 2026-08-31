from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

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
    _workload_urls: Mapping[tuple[str, str], str] = field(
        default_factory=lambda: MappingProxyType({}),
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
        """返回当前 Dashboard generation 声明过的 workspace 文件。"""

        for declared, path in self._workspace_files:
            if declared == name:
                return path
        raise CompositionError(
            "WORKSPACE_FILE_UNDECLARED",
            f"{self.plugin_id} 未声明 workspace file: {name}",
        )

    def workload_url(self, workload: str, port: str) -> str:
        """Return one ready endpoint owned by this plugin generation."""

        try:
            return self._workload_urls[(workload, port)]
        except KeyError as error:
            raise CompositionError(
                "WORKLOAD_PORT_UNDECLARED",
                f"{self.plugin_id} 未声明 Workload port: {workload}:{port}",
            ) from error
