from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

from agent.plugin_composition.context import Context, FiberHandle, HealthHandle
from agent.plugin_composition.model import CompositionError, FiberState, ServiceKey
from agent.plugin_composition.models import ModelRole

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_EXPORT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:]*$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} 必须是字符串")
    if not allow_empty and not value:
        raise ValueError(f"{field} 不能为空")
    if value.strip() != value:
        raise ValueError(f"{field} 不能有首尾空白")
    return value


def _tuple(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} 必须是 tuple")
    return value


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    items = _tuple(value, field)
    result = tuple(_text(item, f"{field}[]") for item in items)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} 不能包含重复值")
    return result


def _identifier(value: object, field: str) -> str:
    text = _text(value, field)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValueError(f"{field} 无效: {text}")
    return text


def _export(value: object) -> str:
    text = _text(value, "handler_export")
    if _EXPORT.fullmatch(text) is None or ".." in text or text.endswith((".", ":")):
        raise ValueError(f"handler_export 无效: {text}")
    return text


@dataclass(frozen=True, slots=True)
class IntervalTrigger:
    """Trigger a job on a positive Core-owned interval."""

    seconds: int

    def __post_init__(self) -> None:
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, int):
            raise TypeError("interval seconds 必须是整数")
        if self.seconds <= 0:
            raise ValueError("interval seconds 必须是正整数")


BackgroundJobTrigger: TypeAlias = IntervalTrigger


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Freeze the retry policy fields that participate in catalog identity."""

    max_attempts: int = 1
    base_delay_seconds: float = 0.0
    max_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("retry max_attempts 必须是整数")
        if self.max_attempts <= 0:
            raise ValueError("retry max_attempts 必须是正整数")
        for field, value in (
            ("base_delay_seconds", self.base_delay_seconds),
            ("max_delay_seconds", self.max_delay_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"retry {field} 必须是数字")
            if value < 0:
                raise ValueError(f"retry {field} 不能为负数")
            if not math.isfinite(value):
                raise ValueError(f"retry {field} 必须是有限数字")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("retry max_delay_seconds 不能小于 base_delay_seconds")


BackgroundJobRetryPolicy = RetryPolicy


@dataclass(frozen=True, slots=True)
class ProgrammaticTurnReceipt:
    """Identify a Turn admitted through one invocation-scoped port."""

    session_id: str
    turn_id: str


class ProgrammaticTurnPreAdmissionError(RuntimeError):
    """Report a failure proven to happen before Turn admission."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class ProgrammaticTurnUncertainError(RuntimeError):
    """Report a failure after Turn admission may already have happened."""


class ProgrammaticTurnPort(Protocol):
    """Expose only Core-owned programmatic Turn admission to one job."""

    async def create_session(self, *, metadata: Mapping[str, object]) -> str: ...

    async def submit(
        self,
        session_id: str,
        content: str,
    ) -> ProgrammaticTurnReceipt: ...


@dataclass(frozen=True, slots=True)
class BackgroundJobDefinition:
    """Describe one interval job without retaining a handler callable."""

    name: str
    triggers: tuple[BackgroundJobTrigger, ...]
    handler_export: str
    debounce_seconds: int = 0
    coalesce: bool = True
    retry_policy: RetryPolicy = RetryPolicy()
    model_role: str | None = None
    programmatic_turns: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(f"background job name 无效: {self.name}")
        triggers = _tuple(self.triggers, "triggers")
        if not triggers:
            raise ValueError("background job 至少需要一个 trigger")
        for trigger in triggers:
            if not isinstance(trigger, IntervalTrigger):
                raise TypeError("background job trigger 类型无效")
        typed_triggers = tuple(
            cast(BackgroundJobTrigger, trigger) for trigger in triggers
        )
        if len({_trigger_identity(trigger) for trigger in typed_triggers}) != len(
            triggers
        ):
            raise ValueError("background job trigger 不能重复")
        _export(self.handler_export)
        if (
            isinstance(self.debounce_seconds, bool)
            or not isinstance(self.debounce_seconds, int)
            or self.debounce_seconds < 0
        ):
            raise ValueError("debounce_seconds 必须是非负整数")
        if not isinstance(self.coalesce, bool):
            raise TypeError("coalesce 必须是 bool")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("retry_policy 必须是 RetryPolicy")
        if self.model_role is not None:
            _identifier(self.model_role, "model_role")
            try:
                ModelRole(self.model_role)
            except ValueError as error:
                raise ValueError(
                    f"background job model_role 无效: {self.model_role}"
                ) from error
        if not isinstance(self.programmatic_turns, bool):
            raise TypeError("programmatic_turns 必须是 bool")


@dataclass(frozen=True, slots=True)
class BackgroundJobDescriptor:
    """Store only immutable job metadata used by snapshot identity."""

    owner: str
    name: str
    triggers: tuple[BackgroundJobTrigger, ...]
    debounce_seconds: int
    coalesce: bool
    handler_export: str
    retry_policy: RetryPolicy
    model_role: str | None = None
    programmatic_turns: bool = False

    def __post_init__(self) -> None:
        _text(self.owner, "owner")
        definition = BackgroundJobDefinition(
            name=self.name,
            triggers=self.triggers,
            handler_export=self.handler_export,
            debounce_seconds=self.debounce_seconds,
            coalesce=self.coalesce,
            retry_policy=self.retry_policy,
            model_role=self.model_role,
            programmatic_turns=self.programmatic_turns,
        )
        object.__setattr__(self, "triggers", definition.triggers)


@dataclass(frozen=True, slots=True)
class BackgroundJobBinding:
    """Bind one descriptor to an exact generation, Fiber activation and Health."""

    generation_id: str
    plugin_id: str
    name: str
    descriptor: BackgroundJobDescriptor
    definition: BackgroundJobDefinition
    owner_fiber: FiberHandle
    activation_token: object
    required_health: HealthHandle

    @property
    def owner(self) -> str:
        return self.plugin_id

    @property
    def handler_export(self) -> str:
        return self.descriptor.handler_export

    def is_owned(self) -> bool:
        return (
            self.owner_fiber.state is FiberState.ACTIVE
            and self.owner_fiber.activation_token is self.activation_token
        )

    def is_live(self) -> bool:
        return self.is_owned() and self.required_health.healthy


class BackgroundJobCatalog(Mapping[str, BackgroundJobBinding]):
    """Expose one immutable Root-local background-job catalog."""

    __slots__ = (
        "_root_instance_token",
        "_bindings",
        "_descriptors",
        "_identity",
    )

    def __init__(
        self,
        bindings: Mapping[str, BackgroundJobBinding],
        *,
        root_instance_token: object,
    ) -> None:
        self._root_instance_token = root_instance_token
        self._bindings = MappingProxyType(dict(sorted(bindings.items())))
        self._descriptors = tuple(
            sorted(
                (item.descriptor for item in self._bindings.values()),
                key=lambda item: (item.owner, item.name),
            )
        )
        self._identity = _digest(
            [_descriptor_identity(item) for item in self._descriptors]
        )

    @property
    def root_instance_token(self) -> object:
        return self._root_instance_token

    @property
    def descriptors(self) -> tuple[BackgroundJobDescriptor, ...]:
        return self._descriptors

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def catalog_digest(self) -> str:
        return self._identity

    def job(self, name: str) -> BackgroundJobBinding | None:
        """Resolve a canonical owner/name key or a unique semantic name."""

        binding = self._bindings.get(name)
        if binding is not None:
            return binding
        matches = tuple(item for item in self._bindings.values() if item.name == name)
        if len(matches) == 1:
            return matches[0]
        return None

    def __getitem__(self, key: str) -> BackgroundJobBinding:
        return self._bindings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


BACKGROUND_JOBS = ServiceKey["PluginBackgroundJobs"]("core.background_jobs")


@dataclass(slots=True)
class _Registration:
    token: int
    plugin_id: str
    generation_id: str
    definition: BackgroundJobDefinition
    descriptor: BackgroundJobDescriptor
    owner_fiber: FiberHandle
    activation_token: object
    required_health: HealthHandle | None = None


class _BackgroundJobDeclarations:
    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _Registration] = {}
        self._names: dict[tuple[str, str], int] = {}
        self._frozen: BackgroundJobCatalog | None = None

    async def register(self, ctx: Context, definition: BackgroundJobDefinition) -> None:
        normalized = _normalize_definition(definition)
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError(
                "INACTIVE_FIBER", "当前 Fiber 没有 active activation"
            )
        registration: _Registration | None = None

        def setup() -> object:
            nonlocal registration
            registration, cleanup = self._register(
                ctx.runtime.plugin_id,
                ctx.generation_id,
                normalized,
                owner_fiber,
                activation_token,
            )
            return cleanup

        effect = await ctx.effect(
            setup,
            label=f"background-job:{normalized.name}",
        )
        try:
            health = await ctx.health(
                f"background-job:{normalized.name}",
                required=True,
            )
        except BaseException:
            await effect.aclose()
            raise
        assert registration is not None
        registration.required_health = health

    def _register(
        self,
        plugin_id: str,
        generation_id: str,
        definition: BackgroundJobDefinition,
        owner_fiber: FiberHandle,
        activation_token: object,
    ) -> tuple[_Registration, object]:
        if self._frozen is not None:
            raise CompositionError(
                "BACKGROUND_JOBS_FROZEN",
                "插件 background job 声明已冻结，不能新增",
            )
        name_key = (plugin_id, definition.name)
        if name_key in self._names:
            raise CompositionError(
                "DUPLICATE_BACKGROUND_JOB",
                f"background job 名称重复: {definition.name}",
            )
        token = self._next_token
        self._next_token += 1
        descriptor = BackgroundJobDescriptor(
            owner=plugin_id,
            name=definition.name,
            triggers=definition.triggers,
            debounce_seconds=definition.debounce_seconds,
            coalesce=definition.coalesce,
            handler_export=definition.handler_export,
            retry_policy=definition.retry_policy,
            model_role=definition.model_role,
            programmatic_turns=definition.programmatic_turns,
        )
        registration = _Registration(
            token=token,
            plugin_id=plugin_id,
            generation_id=generation_id,
            definition=definition,
            descriptor=descriptor,
            owner_fiber=owner_fiber,
            activation_token=activation_token,
        )
        self._registrations[token] = registration
        self._names[name_key] = token

        def cleanup() -> None:
            self._registrations.pop(token, None)
            if self._names.get(name_key) == token:
                self._names.pop(name_key, None)

        return registration, cleanup

    def freeze(
        self,
        root_instance_token: object,
        generation_ids: Mapping[str, str] | None = None,
    ) -> BackgroundJobCatalog:
        if self._frozen is not None:
            if self._frozen.root_instance_token is not root_instance_token:
                raise RuntimeError("background job catalog 属于另一棵 Root")
            if generation_ids is not None and any(
                generation_ids.get(binding.plugin_id) != binding.generation_id
                for binding in self._frozen.values()
            ):
                raise RuntimeError("background job catalog generation identity 已冻结")
            return self._frozen
        bindings: dict[str, BackgroundJobBinding] = {}
        for registration in sorted(
            self._registrations.values(), key=lambda item: item.token
        ):
            health = registration.required_health
            if health is None:
                raise RuntimeError("background job 缺少 required Health")
            generation_id = (
                registration.generation_id
                if generation_ids is None
                else generation_ids.get(registration.plugin_id)
            )
            if generation_id is None:
                raise RuntimeError(
                    "background job owner 不属于 generations: "
                    f"{registration.plugin_id}"
                )
            binding = BackgroundJobBinding(
                generation_id=generation_id,
                plugin_id=registration.plugin_id,
                name=registration.definition.name,
                descriptor=registration.descriptor,
                definition=registration.definition,
                owner_fiber=registration.owner_fiber,
                activation_token=registration.activation_token,
                required_health=health,
            )
            bindings[_binding_key(binding)] = binding
        self._frozen = BackgroundJobCatalog(
            bindings,
            root_instance_token=root_instance_token,
        )
        return self._frozen


class PluginBackgroundJobs:
    """Expose only Root-token-bound job registration to plugins."""

    def __init__(self, root_instance_token: object) -> None:
        self._root_instance_token = root_instance_token
        self._declarations = _BackgroundJobDeclarations()

    async def register(self, ctx: Context, definition: BackgroundJobDefinition) -> None:
        if (
            ctx._root_instance_token() is not self._root_instance_token
            or ctx.require(BACKGROUND_JOBS) is not self
        ):
            raise CompositionError(
                "BACKGROUND_JOBS_SERVICE_ROOT_MISMATCH",
                "插件 background job Service 不属于当前 Root",
            )
        await self._declarations.register(ctx, definition)


def _freeze_plugin_background_jobs(
    value: object,
    root_instance_token: object,
    generation_ids: Mapping[str, str] | None = None,
) -> BackgroundJobCatalog:
    """Freeze the exact Core-created background job registration facade."""

    if not isinstance(value, PluginBackgroundJobs):
        raise RuntimeError("RuntimeSnapshot background job Service 类型无效")
    if value._root_instance_token is not root_instance_token:
        raise RuntimeError("RuntimeSnapshot background job Service 不属于 exact Root")
    return value._declarations.freeze(root_instance_token, generation_ids)


def _normalize_definition(
    definition: BackgroundJobDefinition,
) -> BackgroundJobDefinition:
    if not isinstance(definition, BackgroundJobDefinition):
        raise TypeError("PluginBackgroundJobs.register 只接受 BackgroundJobDefinition")
    return BackgroundJobDefinition(
        name=definition.name,
        triggers=tuple(definition.triggers),
        handler_export=definition.handler_export,
        debounce_seconds=definition.debounce_seconds,
        coalesce=definition.coalesce,
        retry_policy=definition.retry_policy,
        model_role=definition.model_role,
        programmatic_turns=definition.programmatic_turns,
    )


def _binding_key(binding: BackgroundJobBinding) -> str:
    return f"{binding.plugin_id}:{binding.name}"


def _trigger_identity(trigger: BackgroundJobTrigger) -> tuple[str, object]:
    return ("interval", trigger.seconds)


def _descriptor_identity(descriptor: BackgroundJobDescriptor) -> dict[str, object]:
    return {
        "owner": descriptor.owner,
        "name": descriptor.name,
        "triggers": [
            {"kind": kind, "value": value}
            for kind, value in (_trigger_identity(item) for item in descriptor.triggers)
        ],
        "debounce_seconds": descriptor.debounce_seconds,
        "coalesce": descriptor.coalesce,
        "handler_export": descriptor.handler_export,
        "retry_policy": {
            "max_attempts": descriptor.retry_policy.max_attempts,
            "base_delay_seconds": descriptor.retry_policy.base_delay_seconds,
            "max_delay_seconds": descriptor.retry_policy.max_delay_seconds,
        },
        "model_role": descriptor.model_role,
        "programmatic_turns": descriptor.programmatic_turns,
    }


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "BACKGROUND_JOBS",
    "BackgroundJobBinding",
    "BackgroundJobCatalog",
    "BackgroundJobDefinition",
    "BackgroundJobDescriptor",
    "BackgroundJobRetryPolicy",
    "BackgroundJobTrigger",
    "IntervalTrigger",
    "PluginBackgroundJobs",
    "ProgrammaticTurnPort",
    "ProgrammaticTurnPreAdmissionError",
    "ProgrammaticTurnReceipt",
    "ProgrammaticTurnUncertainError",
    "RetryPolicy",
    "_freeze_plugin_background_jobs",
]
