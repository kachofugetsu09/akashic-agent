from __future__ import annotations

from collections.abc import Callable, Awaitable
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import (
    ServiceKey,
)


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


UI_SLOTS = ServiceKey[UiSlots]("core.ui_slots")
