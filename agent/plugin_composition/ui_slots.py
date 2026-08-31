from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

from agent.plugin_composition.context import Context, FiberHandle
from agent.plugin_composition.model import (
    CompositionError,
    FiberState,
    ServiceKey,
)
from agent.plugins.generation import MobileUiAsset


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


class MobileUiRegistry(Mapping[str, MobileUiBinding]):
    """Expose immutable mobile descriptors and exact-snapshot handlers."""

    def __init__(self, bindings: Mapping[str, MobileUiBinding]) -> None:
        if any(key != binding.descriptor.owner for key, binding in bindings.items()):
            raise ValueError("Mobile UI registry key 与 descriptor owner 不一致")
        self._bindings = MappingProxyType(
            {plugin_id: bindings[plugin_id] for plugin_id in sorted(bindings)}
        )
        self._descriptors = tuple(
            sorted(
                (binding.descriptor for binding in self._bindings.values()),
                key=lambda descriptor: descriptor.owner,
            )
        )
        identity_payload = [
            {
                "owner": descriptor.owner,
                "module_sha256": descriptor.module_sha256,
                "module_bytes": descriptor.module_bytes,
                "stylesheet_sha256": descriptor.stylesheet_sha256,
                "stylesheet_bytes": descriptor.stylesheet_bytes,
                "navigation_label": descriptor.navigation_label,
                "navigation_description": descriptor.navigation_description,
                "slots": list(descriptor.slots),
            }
            for descriptor in self._descriptors
        ]
        self._identity = hashlib.sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def descriptors(self) -> tuple[MobileUiDescriptor, ...]:
        return self._descriptors

    @property
    def identity(self) -> str:
        return self._identity

    def binding(self, plugin_id: str) -> MobileUiBinding | None:
        return self._bindings.get(plugin_id)

    def descriptor(self, plugin_id: str) -> MobileUiDescriptor | None:
        binding = self.binding(plugin_id)
        return None if binding is None else binding.descriptor

    def __getitem__(self, plugin_id: str) -> MobileUiBinding:
        return self._bindings[plugin_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


UI_SLOTS = ServiceKey["PluginUiSlots"]("core.ui_slots")


@dataclass(frozen=True, slots=True)
class _MobileUiRegistration:
    token: int
    plugin_id: str
    binding: MobileUiBinding


class PluginUiSlots:
    """Collect plugin-owned mobile UI declarations for one composition Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _MobileUiRegistration] = {}
        self._frozen: MobileUiRegistry | None = None

    async def register_mobile(
        self,
        ctx: Context,
        definition: MobileUiDefinition,
        *,
        query: MobileUiQueryHandler,
        available: Callable[[], bool] | None = None,
    ) -> None:
        """Register one mobile UI declaration as an Effect of the calling Fiber."""

        # 1. Validate the public ABI before any registration becomes visible.
        if not isinstance(definition, MobileUiDefinition):
            raise TypeError("插件 Mobile UI 声明必须是 MobileUiDefinition")
        _validate_sync_callable(query, "query")
        if available is not None:
            _validate_sync_callable(available, "available")
        navigation = definition.navigation
        if navigation is not None and not isinstance(navigation, MobileUiNavigation):
            raise TypeError("插件 Mobile UI navigation 必须是 MobileUiNavigation")
        runtime = ctx.runtime
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError(
                "INACTIVE_FIBER",
                f"{runtime.plugin_id} 当前 Fiber 没有 active activation",
            )
        asset = resolve_mobile_ui_asset(
            runtime.plugin_dir,
            module=definition.module,
            stylesheet=definition.stylesheet,
            navigation_label=None if navigation is None else navigation.label,
            navigation_description=(
                None if navigation is None else navigation.description
            ),
            slots=tuple(definition.slots),
        )
        binding = MobileUiBinding(
            descriptor=MobileUiDescriptor(
                owner=runtime.plugin_id,
                module_sha256=asset.module_sha256,
                module_bytes=asset.module_bytes,
                stylesheet_sha256=asset.stylesheet_sha256,
                stylesheet_bytes=asset.stylesheet_bytes,
                navigation_label=asset.navigation_label,
                navigation_description=asset.navigation_description,
                slots=asset.slots,
            ),
            asset=asset,
            query=cast(MobileUiQueryHandler, query),
            available=_always_available if available is None else available,
            owner_fiber=owner_fiber,
            activation_token=activation_token,
        )
        _ = await ctx.effect(
            lambda: self._register(runtime.plugin_id, binding),
            label=f"ui-slot:mobile:{definition.module}",
        )

    def freeze(self) -> MobileUiRegistry:
        """Freeze registrations into the immutable snapshot registry."""

        if self._frozen is None:
            registrations = sorted(
                self._registrations.values(),
                key=lambda registration: registration.token,
            )
            bindings: dict[str, MobileUiBinding] = {}
            for registration in registrations:
                owner = registration.plugin_id
                if owner in bindings:
                    raise CompositionError(
                        "DUPLICATE_PLUGIN_MOBILE_UI",
                        f"插件只能声明一个 Mobile UI: {owner}",
                    )
                bindings[owner] = registration.binding
            self._frozen = MobileUiRegistry(bindings)
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        binding: MobileUiBinding,
    ) -> Callable[[], None]:
        """Add one declaration and return its exact inverse."""

        # 1. Freeze closes the admission boundary for this Root.
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_UI_SLOTS_FROZEN",
                "插件 UI Slot 声明已冻结，不能在 snapshot 发布后新增",
            )
        if any(
            registration.plugin_id == plugin_id
            for registration in self._registrations.values()
        ):
            raise CompositionError(
                "DUPLICATE_PLUGIN_MOBILE_UI",
                f"插件只能声明一个 Mobile UI: {plugin_id}",
            )

        # 2. Effect cleanup removes only this registration token.
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _MobileUiRegistration(
            token=token,
            plugin_id=plugin_id,
            binding=binding,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup


def resolve_mobile_ui_asset(
    plugin_dir: Path,
    *,
    module: str,
    stylesheet: str | None,
    navigation_label: str | None,
    navigation_description: str | None,
    slots: tuple[str, ...],
) -> MobileUiAsset:
    """Validate and freeze plugin-owned mobile UI static assets."""

    # 1. Validate metadata and the protocol slot namespace.
    _validate_mobile_ui_metadata(
        module=module,
        stylesheet=stylesheet,
        navigation_label=navigation_label,
        navigation_description=navigation_description,
        slots=slots,
    )

    # 2. Resolve symlinks before enforcing plugin-source containment.
    plugin_root = plugin_dir.resolve(strict=True)
    module_path = _resolve_asset_path(plugin_root, module, suffix=".js", kind="module")
    stylesheet_path = (
        None
        if stylesheet is None
        else _resolve_asset_path(
            plugin_root,
            stylesheet,
            suffix=".css",
            kind="stylesheet",
        )
    )
    return _build_mobile_ui_asset(
        module_path,
        stylesheet_path,
        navigation_label=navigation_label,
        navigation_description=navigation_description,
        slots=slots,
    )


def _validate_mobile_ui_metadata(
    *,
    module: str,
    stylesheet: str | None,
    navigation_label: str | None,
    navigation_description: str | None,
    slots: tuple[str, ...],
) -> None:
    if (
        not isinstance(module, str)
        or not module
        or module != module.strip()
        or Path(module).is_absolute()
    ):
        raise RuntimeError("插件 mobile UI module 必须是非空相对路径")
    if stylesheet is not None and (
        not isinstance(stylesheet, str)
        or not stylesheet
        or stylesheet != stylesheet.strip()
        or Path(stylesheet).is_absolute()
    ):
        raise RuntimeError("插件 mobile UI stylesheet 必须是非空相对路径")
    if (navigation_label is None) != (navigation_description is None):
        raise RuntimeError("插件 mobile UI navigation 无效")
    if navigation_label is not None and (
        not isinstance(navigation_label, str)
        or not navigation_label.strip()
        or len(navigation_label) > 64
        or not isinstance(navigation_description, str)
        or not navigation_description.strip()
        or len(navigation_description) > 160
    ):
        raise RuntimeError("插件 mobile UI navigation 无效")
    if not isinstance(slots, tuple) or any(
        not isinstance(slot, str) or slot not in MOBILE_UI_SLOTS for slot in slots
    ):
        raise RuntimeError("插件 mobile UI slots 无效")
    if len(set(slots)) != len(slots):
        raise RuntimeError("插件 mobile UI slots 无效")


def _build_mobile_ui_asset(
    module_path: Path,
    stylesheet_path: Path | None,
    *,
    navigation_label: str | None,
    navigation_description: str | None,
    slots: tuple[str, ...],
) -> MobileUiAsset:
    """Read validated assets and attach content hashes and byte sizes."""

    module_content = module_path.read_text(encoding="utf-8")
    stylesheet_content = (
        "" if stylesheet_path is None else stylesheet_path.read_text(encoding="utf-8")
    )
    module_encoded = module_content.encode("utf-8")
    stylesheet_encoded = stylesheet_content.encode("utf-8")
    if len(module_encoded) + len(stylesheet_encoded) > 240 * 1024:
        raise RuntimeError("插件 mobile UI 资产超过协议安全预算")
    return MobileUiAsset(
        module=module_content,
        module_sha256=hashlib.sha256(module_encoded).hexdigest(),
        module_bytes=len(module_encoded),
        stylesheet=stylesheet_content,
        stylesheet_sha256=(
            hashlib.sha256(stylesheet_encoded).hexdigest()
            if stylesheet_content
            else None
        ),
        stylesheet_bytes=len(stylesheet_encoded),
        navigation_label=(
            None if navigation_label is None else navigation_label.strip()
        ),
        navigation_description=(
            None
            if navigation_description is None
            else navigation_description.strip()
        ),
        slots=slots,
    )


def _resolve_asset_path(
    plugin_root: Path,
    relative_path: str,
    *,
    suffix: str,
    kind: str,
) -> Path:
    """Resolve one asset and reject missing, wrong-type, and escaped paths."""

    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"插件 mobile UI {kind} 无效: {relative_path}") from error
    if (
        not path.is_relative_to(plugin_root)
        or path.suffix != suffix
        or not path.is_file()
    ):
        raise RuntimeError(f"插件 mobile UI {kind} 无效: {relative_path}")
    return path


def _always_available() -> bool:
    return True


def _validate_sync_callable(value: object, field_name: str) -> None:
    if not callable(value):
        raise TypeError(f"插件 Mobile UI {field_name} 必须可调用")
    if inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(
        getattr(value, "__call__", None)
    ):
        raise TypeError(f"插件 Mobile UI {field_name} 必须是同步函数")
