"""UI 提供方公开合同；登记、资产和请求执行由 provider 实现。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import WebSocketRoute

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

DashboardRoute = APIRoute | WebSocketRoute

@dataclass(frozen=True)
class WebModuleAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    contract_digests: tuple[tuple[str, str], ...] = ()
    contract_sha256: str = ""

@dataclass(frozen=True)
class WebModuleDescriptor:
    plugin_id: str
    registration_uuid: str
    generation_id: str
    asset: WebModuleAsset

@dataclass(frozen=True)
class WebUiCatalog:
    identity: str
    modules: tuple[WebModuleDescriptor, ...]


@dataclass(frozen=True)
class DashboardBinding:
    plugin_id: str
    app: FastAPI
    routes: tuple[DashboardRoute, ...]
    context: Context
    generation_id: str
    has_web: bool
    runtime_workspace: Path
    runtime_data_root: Path
    module_name: str


class UiRegistry(Protocol):
    @property
    def root_instance_token(self) -> object: ...

    async def register(
        self, ctx: Context, *, web: str | None = None, dashboard: Callable[[], ModuleType] | None = None,
        requires: tuple[str, ...] = (), provides: tuple[str, ...] = (),
        contract_digests: Mapping[str, str] | None = None,
    ) -> Effect: ...

    def catalog(self) -> WebUiCatalog: ...

    def bindings(self) -> tuple[DashboardBinding, ...]: ...

class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...

    async def state(self) -> dict[str, str | bool]: ...


UI = ServiceKey[UiRegistry]("ui.v1")
WEB_UI = ServiceKey[WebUiProvider]("ui.web.v1")

@dataclass(frozen=True)
class PluginUiAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]


PluginUiSlot = Literal[
    "turn.before_reasoning",
    "turn.before_tool",
    "turn.after_answer",
    "drawer.panel",
]

PLUGIN_UI_SLOTS = frozenset(
    {
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    }
)

class PluginUiQueryHandler(Protocol):
    def __call__(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> object | Awaitable[object]:
        """Sync handlers run in bounded workers; async handlers keep the owner scope."""
        ...

class PluginUiRpcInvalidRequest(ValueError):
    """Signal a request rejected by the plugin-owned UI projection."""

class PluginUiPluginUnavailable(LookupError):
    """Signal that the requested Plugin UI owner is not available."""

class PluginUiStaleRevision(LookupError):
    """Signal that a Plugin UI request names an old registration revision."""

class PluginUiQueryTimeout(TimeoutError):
    """Signal that a Plugin UI query exceeded its caller-visible deadline."""

class PluginUiQueryOverloaded(RuntimeError):
    """Signal that the bounded Plugin UI query admission is full."""

class PluginUiRpcExecutionError(RuntimeError):
    """Signal that a Plugin UI handler failed while executing its RPC."""

@dataclass(frozen=True, slots=True)
class PluginUiNavigation:
    label: str
    description: str

@dataclass(frozen=True, slots=True)
class PluginUiDefinition:
    module: str
    stylesheet: str | None = None
    navigation: PluginUiNavigation | None = None
    slots: tuple[PluginUiSlot, ...] = ()

@dataclass(frozen=True, slots=True)
class PluginUiDescriptor:
    """Describe immutable plugin assets without retaining executable handlers."""

    owner: str
    module_sha256: str
    module_bytes: int
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class PluginUiBinding:
    """Bind one descriptor and its handlers to one exact contributor Context."""

    descriptor: PluginUiDescriptor
    asset: PluginUiAsset
    query: PluginUiQueryHandler
    available: Callable[[], bool]
    context: Context
    registration_uuid: str

class UiSlots(Protocol):
    """Expose the current plugin registrations owned by the UI provider."""

    @property
    def root_instance_token(self) -> object: ...

    async def register_plugin_ui(
        self, ctx: Context, definition: PluginUiDefinition, *,
        query: PluginUiQueryHandler, available: Callable[[], bool] | None = None,
    ) -> Effect: ...

    def bindings(self) -> tuple[PluginUiBinding, ...]: ...


UI_SLOTS = ServiceKey[UiSlots]("ui.slots.v1")
