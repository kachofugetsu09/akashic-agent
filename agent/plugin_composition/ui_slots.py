from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import (
    ServiceKey,
)


@dataclass(frozen=True)
class MobileUiAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]


MobileUiSlot = Literal[
    "turn.before_reasoning",
    "turn.before_tool",
    "turn.after_answer",
    "drawer.panel",
]

MOBILE_UI_SLOTS = frozenset(
    {
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    }
)


class MobileUiQueryHandler(Protocol):
    def __call__(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> object: ...


class MobileUiRpcInvalidRequest(ValueError):
    """Signal a request rejected by the plugin-owned mobile projection."""


class MobileUiPluginUnavailable(LookupError):
    """Signal that the requested Mobile UI owner is not available."""


class MobileUiStaleRevision(LookupError):
    """Signal that a Mobile UI request names an old registration revision."""


class MobileUiQueryTimeout(TimeoutError):
    """Signal that a Mobile UI query exceeded its caller-visible deadline."""


class MobileUiQueryOverloaded(RuntimeError):
    """Signal that the bounded Mobile UI query admission is full."""


class MobileUiRpcExecutionError(RuntimeError):
    """Signal that a Mobile UI handler failed while executing its RPC."""


@dataclass(frozen=True, slots=True)
class MobileUiNavigation:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class MobileUiDefinition:
    module: str
    stylesheet: str | None = None
    navigation: MobileUiNavigation | None = None
    slots: tuple[MobileUiSlot, ...] = ()


@dataclass(frozen=True, slots=True)
class MobileUiDescriptor:
    """Describe immutable mobile assets without retaining executable handlers."""

    owner: str
    module_sha256: str
    module_bytes: int
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class MobileUiBinding:
    """Bind one descriptor and its handlers to one exact contributor Context."""

    descriptor: MobileUiDescriptor
    asset: MobileUiAsset
    query: MobileUiQueryHandler
    available: Callable[[], bool]
    context: Context
    registration_uuid: str


class UiSlots(Protocol):
    """Expose the current Mobile registrations owned by the UI provider."""

    @property
    def root_instance_token(self) -> object: ...

    async def register_mobile(
        self, ctx: Context, definition: MobileUiDefinition, *,
        query: MobileUiQueryHandler, available: Callable[[], bool] | None = None,
    ) -> Effect: ...

    def bindings(self) -> tuple[MobileUiBinding, ...]: ...


UI_SLOTS = ServiceKey[UiSlots]("core.ui_slots")
