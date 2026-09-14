from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition.context import Context, FiberHandle
from agent.plugin_composition.model import (
    FiberState,
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
    """Bind one descriptor and its handlers to one exact snapshot Root."""

    descriptor: MobileUiDescriptor
    asset: MobileUiAsset
    query: MobileUiQueryHandler
    available: Callable[[], bool]
    owner_fiber: FiberHandle | None = None
    activation_token: object | None = None

    def is_live(self) -> bool:
        """Return whether this binding still belongs to its active Fiber activation."""

        if self.owner_fiber is None:
            return True
        return (
            self.activation_token is not None
            and self.owner_fiber.state is FiberState.ACTIVE
            and self.owner_fiber.activation_token is self.activation_token
        )


class MobileUiRegistry(Protocol):
    """一个实际 Root 的封存目录，只提供读取能力。"""

    @property
    def root_instance_token(self) -> object: ...
    @property
    def descriptors(self) -> tuple[MobileUiDescriptor, ...]: ...
    def binding(self, plugin_id: str) -> MobileUiBinding | None: ...
    def descriptor(self, plugin_id: str) -> MobileUiDescriptor | None: ...
    def __getitem__(self, plugin_id: str) -> MobileUiBinding: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...


class UiSlots(Protocol):
    """Mobile 贡献经实际 Context 注册，封存由所选 provider 拥有。"""

    @property
    def root_instance_token(self) -> object: ...

    async def register_mobile(
        self, ctx: Context, definition: MobileUiDefinition, *,
        query: MobileUiQueryHandler, available: Callable[[], bool] | None = None,
    ) -> None: ...

    def catalog(self) -> MobileUiRegistry: ...


UI_SLOTS = ServiceKey[UiSlots]("core.ui_slots")
