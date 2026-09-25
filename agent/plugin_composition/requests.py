from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from agent.plugin_composition.model import CompositionError, ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context


@dataclass(frozen=True, slots=True)
class RequestContext:
    """向插件请求暴露声明能力和该 generation 的资源路径。"""

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

    _resolve: Callable[[ServiceKey[Any]], object] | None = field(default=None, repr=False)
    _context: Context | None = field(default=None, repr=False, compare=False)

    def _require_context(self, key: ServiceKey[Any], service: object) -> Context:
        """Core 在当前请求许可内取得原 owner，不向插件开放完整 Context API。"""
        if self.require(key) is not service:
            raise CompositionError("SERVICE_SCOPE_MISMATCH", "授权服务不属于当前请求")
        if self._context is None:
            raise CompositionError("REQUEST_SCOPE_MISSING", "插件没有请求 owner")
        return self._context

    def require[T](self, key: ServiceKey[T]) -> T:
        """在 async 路由的当前请求租约内取得声明能力，不暴露宿主 Root。"""
        if self._resolve is None:
            raise CompositionError("REQUEST_SCOPE_MISSING", "插件没有请求能力入口")
        return cast(T, self._resolve(cast(ServiceKey[Any], key)))

    def workspace_root(self, name: str) -> Path:
        """返回与当前插件 generation 相同的声明式 workspace root。"""

        for declared, path in self._workspace_roots:
            if declared == name:
                return path
        raise CompositionError(
            "WORKSPACE_ROOT_UNDECLARED",
            f"{self.plugin_id} 未声明 workspace root: {name}",
        )

    def workspace_file(self, name: str) -> Path:
        """返回当前插件 generation 声明过的 workspace 文件。"""

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
