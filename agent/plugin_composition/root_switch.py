from __future__ import annotations

# pyright: reportPrivateUsage=false

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import CompositionError, ServiceKey

_SwitchCall = Callable[[], Awaitable[None]]
_RecoverCall = Callable[[bool], Awaitable[None]]
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class SwitchPart:
    """Own one shared resource that cannot stay live across two Roots."""

    name: str
    stop: _SwitchCall
    leave: _SwitchCall
    enter: _SwitchCall
    start: _SwitchCall
    recover: _RecoverCall

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(
                "SwitchPart.name 必须是小写字母开头的简单名称"
            )
        for field_name in ("stop", "leave", "enter", "start", "recover"):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"SwitchPart.{field_name} 必须可调用")


@dataclass(frozen=True, slots=True)
class _PartNeed:
    """Identify one exact plugin needed to build a switch part."""

    owner: str
    generation: str
    artifact: str


@dataclass(frozen=True, slots=True)
class _PartRef:
    """Identify one part from one exact plugin artifact and generation."""

    name: str
    owner: str
    generation: str
    artifact: str
    needs: tuple[_PartNeed, ...] = ()


@dataclass(frozen=True, slots=True)
class _PartEntry:
    """Bind one exact part implementation to its durable identity."""

    ref: _PartRef
    part: SwitchPart


class _PartSet(Mapping[str, _PartEntry]):
    """Expose the immutable switch parts selected for one snapshot."""

    def __init__(self, bindings: Mapping[str, _PartEntry]) -> None:
        ordered = {name: bindings[name] for name in sorted(bindings)}
        if any(name != binding.ref.name for name, binding in ordered.items()):
            raise RuntimeError("Switch registry key 与 part name 不一致")
        self._bindings = MappingProxyType(ordered)
        payload: list[dict[str, object]] = [
            {
                "name": binding.ref.name,
                "owner": binding.ref.owner,
                "generation": binding.ref.generation,
                "artifact": binding.ref.artifact,
                "needs": [
                    {
                        "owner": need.owner,
                        "generation": need.generation,
                        "artifact": need.artifact,
                    }
                    for need in binding.ref.needs
                ],
            }
            for binding in ordered.values()
        ]
        self._identity = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def identity(self) -> str:
        return self._identity

    def __getitem__(self, name: str) -> _PartEntry:
        return self._bindings[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


@dataclass(slots=True)
class _SwitchRegistration:
    token: int
    owner: str
    generation: str
    part: SwitchPart


class _SwitchParts:
    """Own one Root-local set until snapshot sealing freezes it."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _SwitchRegistration] = {}
        self._frozen: _PartSet | None = None

    async def add(self, ctx: Context, part: SwitchPart) -> Effect:
        if not isinstance(part, SwitchPart):
            raise TypeError("RootSwitch.add 只接受 SwitchPart")
        runtime = ctx.runtime

        def setup() -> Callable[[], None]:
            return self._register(
                owner=runtime.plugin_id,
                generation=runtime.generation_id,
                part=part,
            )

        return await ctx.effect(setup, label=f"root-switch:{part.name}")

    def freeze(
        self,
        artifacts: Mapping[str, str],
        needs: Mapping[str, tuple[_PartNeed, ...]],
        plugin_ids: frozenset[str] | None = None,
    ) -> _PartSet:
        if self._frozen is None:
            bindings: dict[str, _PartEntry] = {}
            owners: set[str] = set()
            for item in self._registrations.values():
                artifact = artifacts.get(item.owner)
                if artifact is None:
                    raise RuntimeError(
                        "Root switch part owner 缺少 exact artifact: "
                        f"{item.owner}:{item.part.name}"
                    )
                if item.part.name in bindings:
                    raise CompositionError(
                        "DUPLICATE_ROOT_SWITCH",
                        "Root switch part 名称重复: " + item.part.name,
                    )
                if item.owner in owners:
                    raise CompositionError(
                        "DUPLICATE_OWNER",
                        "Root switch owner 重复: " + item.owner,
                    )
                owners.add(item.owner)
                bindings[item.part.name] = _PartEntry(
                    ref=_PartRef(
                        name=item.part.name,
                        owner=item.owner,
                        generation=item.generation,
                        artifact=artifact,
                        needs=needs.get(item.owner, ()),
                    ),
                    part=item.part,
                )
            self._frozen = _PartSet(bindings)
        if plugin_ids is None:
            return self._frozen
        return _PartSet(
            {
                name: binding
                for name, binding in self._frozen.items()
                if binding.ref.owner in plugin_ids
            }
        )

    def _register(
        self,
        *,
        owner: str,
        generation: str,
        part: SwitchPart,
    ) -> Callable[[], None]:
        if self._frozen is not None:
            raise CompositionError(
                "ROOT_SWITCH_FROZEN",
                "Root switch registry 已冻结",
            )
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _SwitchRegistration(
            token=token,
            owner=owner,
            generation=generation,
            part=part,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup


ROOT_SWITCH = ServiceKey["RootSwitch"]("core.root_switch")


class RootSwitch:
    """Let ordinary plugins register one shared-owner switch part."""

    def __init__(self, root_token: object) -> None:
        self._root_token = root_token
        self._parts = _SwitchParts()

    async def add(self, ctx: Context, part: SwitchPart) -> Effect:
        """Register one Fiber-owned part on the exact Root."""

        if (
            ctx._root_instance_token() is not self._root_token
            or ctx.require(ROOT_SWITCH) is not self
        ):
            raise CompositionError(
                "ROOT_SWITCH_MISMATCH",
                "RootSwitch Service 不属于当前 Root",
            )
        return await self._parts.add(ctx, part)


def _freeze_root_switch(
    value: object,
    root_token: object,
    *,
    artifacts: Mapping[str, str],
    needs: Mapping[str, tuple[_PartNeed, ...]] | None = None,
    plugin_ids: frozenset[str] | None = None,
) -> _PartSet:
    """Freeze the exact Core-created registration facade."""

    if not isinstance(value, RootSwitch):
        raise RuntimeError("RuntimeSnapshot RootSwitch Service 类型无效")
    if value._root_token is not root_token:
        raise RuntimeError("RuntimeSnapshot RootSwitch Service 不属于 exact Root")
    return value._parts.freeze(artifacts, needs or {}, plugin_ids)


def _merge_root_switch(
    base: _PartSet | None,
    delta: _PartSet | None,
    replaced: frozenset[str],
) -> _PartSet | None:
    """Replace selected owners without replaying unchanged plugin code."""

    if base is None and delta is None:
        return None
    bindings: dict[str, _PartEntry] = {}
    for registry in (base, delta):
        if registry is None:
            continue
        for name, binding in registry.items():
            if registry is base and binding.ref.owner in replaced:
                continue
            if name in bindings:
                raise CompositionError(
                    "DUPLICATE_ROOT_SWITCH",
                    f"candidate 与 stable 重复注册 Root switch part: {name}",
                )
            bindings[name] = binding
    return _PartSet(bindings)


__all__ = [
    "ROOT_SWITCH",
    "RootSwitch",
    "SwitchPart",
]
