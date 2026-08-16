from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

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
class FiberView:
    fiber_id: int
    name: str
    state: FiberState
    required_for_readiness: bool
    missing_services: tuple[str, ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class TopologyFiberView:
    name: str
    parent: str | None
    required_for_readiness: bool
    dependencies: tuple[str, ...]


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
    plugin_dir: Path
    data_dir: Path
    workspace: Path
    config: object


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
    errors: tuple[str, ...]
    writes: tuple[WriteObservation, ...]
    external_effects: tuple[ExternalEffectObservation, ...]
