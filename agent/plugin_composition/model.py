from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Mapping
from types import MappingProxyType
from typing import Generic, TypeVar, cast

T = TypeVar("T", covariant=True)


class CompositionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FiberState(str, Enum):
    PENDING = "pending"
    LOADING = "loading"
    ACTIVE = "active"
    FAILED = "failed"
    UNLOADING = "unloading"
    DISPOSED = "disposed"


@dataclass(frozen=True, slots=True)
class ServiceKey(Generic[T]):
    name: str

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("ServiceKey.name 必须是非空且无首尾空白的字符串")


@dataclass(frozen=True, slots=True)
class ServiceView:
    """暴露一组由 Core 冻结的 composition service。"""

    _values: Mapping[ServiceKey[object], object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    @classmethod
    def freeze(
        cls,
        values: Mapping[ServiceKey[object], object],
    ) -> ServiceView:
        return cls(values)

    def get(self, key: ServiceKey[T]) -> T | None:
        return cast(T | None, self._values.get(key))


@dataclass(frozen=True, slots=True)
class FiberView:
    fiber_id: int
    name: str
    state: FiberState
    required_for_readiness: bool
    missing_services: tuple[str, ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class HealthView:
    owner: str
    name: str
    required: bool
    healthy: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class IncidentView:
    sequence: int
    owner: str
    kind: str
    message: str
    error_type: str | None


@dataclass(frozen=True, slots=True)
class TopologyFiberView:
    name: str
    parent: str | None
    required_for_readiness: bool
    dependencies: tuple[str, ...]
    static_active: bool


@dataclass(frozen=True, slots=True)
class TopologyView:
    generation_id: str
    identity: str
    composition_revision: int
    fibers: tuple[TopologyFiberView, ...]
    services: tuple[str, ...]
    effects: tuple[str, ...]
    listeners: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PluginRuntime:
    """Expose the Core-assigned paths and config to one mounted plugin tree."""

    plugin_id: str
    generation_id: str
    plugin_dir: Path
    data_dir: Path
    workspace: Path
    config: object
    workspace_roots: tuple[str, ...] = ()
    workspace_files: tuple[str, ...] = ()

    def workspace_root(self, name: str) -> Path:
        """解析插件声明过的产品级 workspace 顶层目录。"""

        if name not in self.workspace_roots:
            raise CompositionError(
                "WORKSPACE_ROOT_UNDECLARED",
                f"{self.plugin_id} 未声明 workspace root: {name}",
            )
        return resolve_declared_workspace_root(self.workspace, name)

    def workspace_file(self, name: str) -> Path:
        """Resolve one declared product file without broad directory access."""

        if name not in self.workspace_files:
            raise CompositionError(
                "WORKSPACE_FILE_UNDECLARED",
                f"{self.plugin_id} 未声明 workspace file: {name}",
            )
        return resolve_declared_workspace_file(self.workspace, name)


def resolve_declared_workspace_root(workspace: Path, name: str) -> Path:
    """解析 Core 分配的顶层 workspace root，并拒绝符号链接越界。"""

    root = workspace.resolve(strict=False)
    declared = root / name
    if declared.is_symlink():
        raise CompositionError(
            "WORKSPACE_ROOT_SYMLINK",
            f"workspace root 不能是符号链接: {declared}",
        )
    resolved = declared.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise CompositionError(
            "WORKSPACE_ROOT_ESCAPE",
            f"workspace root 越界: {declared}",
        )
    if declared.exists() and not declared.is_dir():
        raise CompositionError(
            "WORKSPACE_ROOT_NOT_DIRECTORY",
            f"workspace root 不是目录: {declared}",
        )
    return resolved


def resolve_declared_workspace_file(workspace: Path, name: str) -> Path:
    """Resolve one relative file and reject symlinks or workspace escape."""

    root = workspace.resolve(strict=False)
    declared = root / name
    try:
        relative = declared.relative_to(root)
    except ValueError as error:
        raise CompositionError(
            "WORKSPACE_FILE_ESCAPE",
            f"workspace file 越界: {declared}",
        ) from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise CompositionError(
                "WORKSPACE_FILE_SYMLINK",
                f"workspace file 路径不能包含符号链接: {current}",
            )
    resolved = declared.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise CompositionError(
            "WORKSPACE_FILE_ESCAPE",
            f"workspace file 越界: {declared}",
        )
    if declared.exists() and not declared.is_file():
        raise CompositionError(
            "WORKSPACE_FILE_NOT_FILE",
            f"workspace file 不是普通文件: {declared}",
        )
    return resolved


@dataclass(frozen=True, slots=True)
class WriteObservation:
    plugin_id: str
    operation: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ExternalEffectObservation:
    kind: str
    target: str
    outcome: str


@dataclass(frozen=True, slots=True)
class CompositionReceipt:
    generation_id: str
    ready: bool
    fibers: tuple[FiberView, ...]
    services: tuple[str, ...]
    effects: tuple[str, ...]
    required_pending: tuple[str, ...]
    optional_pending: tuple[str, ...]
    health: tuple[HealthView, ...]
    required_degraded: tuple[str, ...]
    incidents: tuple[IncidentView, ...]
    incident_sequence: int
    incident_counts: tuple[tuple[str, int], ...]
    incident_overflowed: bool
    writes: tuple[WriteObservation, ...]
    external_effects: tuple[ExternalEffectObservation, ...]
