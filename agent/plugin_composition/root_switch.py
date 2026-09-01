from __future__ import annotations

# pyright: reportPrivateUsage=false

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect, _effect_setup_owner
from agent.plugin_composition.model import CompositionError, ServiceKey

_SwitchCall = Callable[[str], Awaitable[None]]
_RecoverCall = Callable[[str, bool], Awaitable[None]]
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class SwitchInput:
    """Carry one direct RootSwitch contribution until its Effect ends."""

    __slots__ = ("_state",)

    def __new__(cls) -> SwitchInput:
        raise TypeError("SwitchInput 只能由 RootSwitch.input 创建")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("SwitchInput 不能序列化")

    def __copy__(self) -> SwitchInput:
        raise TypeError("SwitchInput 不能复制")

    def __deepcopy__(self, memo: object) -> SwitchInput:
        del memo
        raise TypeError("SwitchInput 不能复制")


@dataclass(frozen=True, slots=True)
class SwitchPart:
    """Own one shared resource that cannot stay live across two Roots."""

    name: str
    stop: _SwitchCall
    leave: _SwitchCall
    enter: _SwitchCall
    start: _SwitchCall
    recover: _RecoverCall
    inputs: tuple[SwitchInput, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(
                "SwitchPart.name 必须是小写字母开头的简单名称"
            )
        for field_name in ("stop", "leave", "enter", "start", "recover"):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"SwitchPart.{field_name} 必须可调用")
        if not isinstance(self.inputs, tuple) or any(
            not isinstance(item, SwitchInput) for item in self.inputs
        ):
            raise TypeError("SwitchPart.inputs 必须是 SwitchInput tuple")


@dataclass(frozen=True, slots=True)
class _PartNeed:
    """Identify one exact plugin needed to build a switch part."""

    owner: str
    generation: str
    artifact: str


@dataclass(frozen=True, slots=True)
class _SwitchInputRef:
    """Identify one direct contributor without persisting token order."""

    owner: str
    generation: str
    artifact: str
    fiber: str


@dataclass(frozen=True, slots=True)
class _PartRef:
    """Identify one part from one exact plugin artifact and generation."""

    name: str
    owner: str
    generation: str
    artifact: str
    fiber: str
    inputs: tuple[_SwitchInputRef, ...] = ()
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
                "fiber": binding.ref.fiber,
                "inputs": [
                    {
                        "owner": item.owner,
                        "generation": item.generation,
                        "artifact": item.artifact,
                        "fiber": item.fiber,
                    }
                    for item in binding.ref.inputs
                ],
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
    fiber: str
    part: SwitchPart


@dataclass(slots=True)
class _SwitchInputState:
    """Own one affine input until its creating Effect is cleaned up."""

    token: int
    secret: object
    root_token: object
    owner: str
    generation: str
    fiber: str
    activation_token: object
    context: Context
    effect: Effect
    active: bool = True


class _SwitchParts:
    """Own one Root-local set until snapshot sealing freezes it."""

    def __init__(self, root_token: object) -> None:
        self._root_token = root_token
        self._secret = object()
        self._next_token = 1
        self._next_input_token = 1
        self._registrations: dict[int, _SwitchRegistration] = {}
        self._inputs: dict[int, _SwitchInputState] = {}
        self._frozen: _PartSet | None = None

    def input(self, ctx: Context) -> SwitchInput:
        """Create one input linked to the current Effect setup."""

        owner = self._setup_owner(ctx)
        if self._frozen is not None:
            raise CompositionError(
                "ROOT_SWITCH_FROZEN",
                "Root switch registry 已冻结",
            )
        if owner.activation_token is None:
            raise CompositionError(
                "INACTIVE_INPUT",
                "SwitchInput 需要 active plugin Fiber",
            )
        token = self._next_input_token
        self._next_input_token += 1
        state = _SwitchInputState(
            token=token,
            secret=self._secret,
            root_token=self._root_token,
            owner=owner.plugin_id,
            generation=owner.generation_id,
            fiber=owner.fiber,
            activation_token=owner.activation_token,
            context=ctx,
            effect=owner.effect,
        )
        self._inputs[token] = state

        def cleanup() -> None:
            state.active = False
            if self._inputs.get(token) is state:
                del self._inputs[token]

        owner.effect._add_setup_cleanup(cleanup)
        value = object.__new__(SwitchInput)
        value._state = state
        return value

    async def add(self, ctx: Context, part: SwitchPart) -> Effect:
        if not isinstance(part, SwitchPart):
            raise TypeError("RootSwitch.add 只接受 SwitchPart")

        def setup() -> Callable[[], None]:
            return self._register(ctx, part)

        return await ctx.effect(setup, label=f"root-switch:{part.name}")

    def freeze(
        self,
        artifacts: Mapping[str, str],
        needs: Mapping[tuple[str, str], tuple[_PartNeed, ...]],
        plugin_ids: frozenset[str] | None = None,
    ) -> _PartSet:
        if self._frozen is None:
            bindings: dict[str, _PartEntry] = {}
            owners: set[str] = set()
            consumed: set[int] = set()
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
                input_refs: list[_SwitchInputRef] = []
                for value in item.part.inputs:
                    state = value._state
                    if (
                        state.secret is not self._secret
                        or state.root_token is not self._root_token
                        or not state.active
                        or self._inputs.get(state.token) is not state
                        or not state.context._owns_effect(state.effect)
                        or state.activation_token
                        is not state.context.fiber.activation_token
                    ):
                        raise CompositionError(
                            "INVALID_INPUT",
                            f"Root switch input 不属于 active exact Root: {item.part.name}",
                        )
                    if state.token in consumed:
                        raise CompositionError(
                            "DUPLICATE_INPUT",
                            f"Root switch input 被重复消费: {item.part.name}",
                        )
                    input_artifact = artifacts.get(state.owner)
                    if input_artifact is None:
                        raise RuntimeError(
                            "Root switch input owner 缺少 exact artifact: "
                            f"{state.owner}:{item.part.name}"
                        )
                    consumed.add(state.token)
                    input_refs.append(
                        _SwitchInputRef(
                            owner=state.owner,
                            generation=state.generation,
                            artifact=input_artifact,
                            fiber=state.fiber,
                        )
                    )
                frozen_inputs = tuple(
                    sorted(
                        input_refs,
                        key=lambda value: (
                            value.owner,
                            value.fiber,
                            value.generation,
                            value.artifact,
                        ),
                    )
                )
                bindings[item.part.name] = _PartEntry(
                    ref=_PartRef(
                        name=item.part.name,
                        owner=item.owner,
                        generation=item.generation,
                        artifact=artifact,
                        fiber=item.fiber,
                        inputs=frozen_inputs,
                        needs=_part_needs(
                            item.owner,
                            item.fiber,
                            frozen_inputs,
                            artifacts,
                            needs,
                        ),
                    ),
                    part=item.part,
                )
            missing = set(self._inputs).difference(consumed)
            if missing:
                raise CompositionError(
                    "UNUSED_INPUT",
                    "Root switch input 没有被任何 SwitchPart 消费",
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

    def _register(self, ctx: Context, part: SwitchPart) -> Callable[[], None]:
        owner = self._setup_owner(ctx)
        if self._frozen is not None:
            raise CompositionError(
                "ROOT_SWITCH_FROZEN",
                "Root switch registry 已冻结",
            )
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _SwitchRegistration(
            token=token,
            owner=owner.plugin_id,
            generation=owner.generation_id,
            fiber=owner.fiber,
            part=part,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup

    def _setup_owner(self, ctx: Context):
        """Require the current Context and Effect setup to be the same owner."""

        owner = _effect_setup_owner()
        runtime = ctx.runtime
        if (
            owner is None
            or not ctx._owns_effect(owner.effect)
            or owner.root_token is not self._root_token
            or owner.root_token is not ctx._root_instance_token()
            or owner.plugin_id != runtime.plugin_id
            or owner.generation_id != runtime.generation_id
            or owner.fiber != ctx.fiber.name
            or owner.activation_token is not ctx.fiber.activation_token
        ):
            raise CompositionError(
                "INPUT_MISMATCH",
                "RootSwitch 资源只能在当前 Context 的 Effect setup 中创建",
            )
        return owner


ROOT_SWITCH = ServiceKey["RootSwitch"]("core.root_switch")


class RootSwitch:
    """Let ordinary plugins register one shared-owner switch part."""

    def __init__(self, root_token: object) -> None:
        self._root_token = root_token
        self._parts = _SwitchParts(root_token)

    def input(self, ctx: Context) -> SwitchInput:
        """Create one affine input in the caller's current Effect setup."""

        self._check_context(ctx)
        return self._parts.input(ctx)

    async def add(self, ctx: Context, part: SwitchPart) -> Effect:
        """Register one Fiber-owned part on the exact Root."""

        self._check_context(ctx)
        return await self._parts.add(ctx, part)

    def _check_context(self, ctx: Context) -> None:
        if (
            ctx._root_instance_token() is not self._root_token
            or ctx.require(ROOT_SWITCH) is not self
        ):
            raise CompositionError(
                "ROOT_SWITCH_MISMATCH",
                "RootSwitch Service 不属于当前 Root",
            )


def _part_needs(
    owner: str,
    fiber: str,
    inputs: tuple[_SwitchInputRef, ...],
    artifacts: Mapping[str, str],
    closures: Mapping[tuple[str, str], tuple[_PartNeed, ...]],
) -> tuple[_PartNeed, ...]:
    """Join the part and input contributor closures without duplicate owners."""

    selected: dict[str, _PartNeed] = {}

    def add(need: _PartNeed) -> None:
        if need.owner == owner:
            return
        current = selected.get(need.owner)
        if current is not None and current != need:
            raise RuntimeError(f"Root switch dependency identity 冲突: {need.owner}")
        selected[need.owner] = need

    for need in closures.get((owner, fiber), ()):
        add(need)
    for item in inputs:
        if item.owner != owner:
            artifact = artifacts.get(item.owner)
            if artifact is None:
                raise RuntimeError(
                    f"Root switch input owner 缺少 exact artifact: {item.owner}"
                )
            if artifact != item.artifact:
                raise RuntimeError(f"Root switch input artifact 不一致: {item.owner}")
            add(_PartNeed(item.owner, item.generation, artifact))
        for need in closures.get((item.owner, item.fiber), ()):
            add(need)
    return tuple(selected[name] for name in sorted(selected))


def _freeze_root_switch(
    value: object,
    root_token: object,
    *,
    artifacts: Mapping[str, str],
    needs: Mapping[tuple[str, str], tuple[_PartNeed, ...]] | None = None,
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
    "SwitchInput",
    "SwitchPart",
]
