"""Models-owned settings commands and the generation-local service key."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import (
    CapabilitySources,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
)


@dataclass(frozen=True, slots=True)
class AddConnection:
    expected_revision: int
    connection_id: str
    name: str
    driver_id: str
    endpoint: str
    auth_identity: str
    credential: Mapping[str, str]
    driver_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UpdateConnection:
    expected_revision: int
    connection_id: str
    name: str
    auth_identity: str
    endpoint: str | None = None
    credential: Mapping[str, str] | None = None
    driver_config: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DisableConnection:
    expected_revision: int
    connection_id: str


@dataclass(frozen=True, slots=True)
class AddModel:
    expected_revision: int
    model_id: str
    connection_id: str
    kind: ModelKind
    model: str
    capabilities: ModelCapabilities
    capability_sources: CapabilitySources
    default_reasoning_effort: str | None = None
    driver_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SetDefaultModel:
    expected_revision: int
    role: str | None
    model_id: str


@dataclass(frozen=True, slots=True)
class SyncModels:
    expected_revision: int
    connection_id: str


@dataclass(frozen=True, slots=True)
class StartConnectionAuth:
    driver_id: str
    connection_id: str
    input: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FinishConnectionAuth:
    expected_revision: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class CancelConnectionAuth:
    attempt_id: str


@dataclass(frozen=True, slots=True)
class CreateConnectionWithModel:
    """Probe and commit one new connection and its first model atomically."""

    connection: AddConnection
    model: AddModel


ModelChange: TypeAlias = (
    AddConnection
    | UpdateConnection
    | DisableConnection
    | AddModel
    | SetDefaultModel
    | SyncModels
    | StartConnectionAuth
    | FinishConnectionAuth
    | CancelConnectionAuth
    | CreateConnectionWithModel
)


@dataclass(frozen=True, slots=True)
class SettingsReceipt:
    revision: int
    status: str
    attempt_id: str | None = None
    challenge: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.challenge is not None:
            object.__setattr__(
                self,
                "challenge",
                MappingProxyType(dict(self.challenge)),
            )


class ModelSettings(Protocol):
    async def discover(self, connection: AddConnection) -> tuple[DiscoveredModel, ...]: ...

    async def apply(self, command: ModelChange) -> SettingsReceipt: ...


MODEL_SETTINGS = ServiceKey[ModelSettings]("models.settings.v1")


__all__ = [
    "AddConnection",
    "AddModel",
    "CancelConnectionAuth",
    "CreateConnectionWithModel",
    "DisableConnection",
    "FinishConnectionAuth",
    "MODEL_SETTINGS",
    "ModelChange",
    "ModelSettings",
    "SetDefaultModel",
    "SettingsReceipt",
    "StartConnectionAuth",
    "SyncModels",
    "UpdateConnection",
]
