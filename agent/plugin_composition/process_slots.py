from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from agent.plugin_composition.context import Context, FiberHandle, HealthHandle
from agent.plugin_composition.model import (
    CompositionError,
    FiberState,
    IncidentView,
    ServiceKey,
)

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_RESERVED_ENV = frozenset(
    {
        "AKA_PLUGIN_DATA_DIR",
        "AKASHIC_PLUGIN_DATA_DIR",
        "AKASHIC_WORKSPACE",
    }
)


@dataclass(frozen=True, slots=True)
class ManagedProcessDefinition:
    name: str
    command: tuple[str, ...]
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)
    port_env: str = "PORT"
    formal_port: int = 0
    readiness_path: str = "/health"
    startup_timeout_seconds: float = 15.0


@dataclass(frozen=True, slots=True)
class ManagedProcessDescriptor:
    owner: str
    name: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    port_env: str
    formal_port: int
    readiness_path: str
    startup_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ManagedProcessBinding:
    descriptor: ManagedProcessDescriptor
    definition: ManagedProcessDefinition
    health: HealthHandle
    owner_fiber: FiberHandle
    activation_token: object
    runtime_plugin_dir: Path = field(repr=False, compare=False)
    runtime_data_dir: Path = field(repr=False, compare=False)
    runtime_workspace: Path = field(repr=False, compare=False)
    incident_reporter: Callable[[str, str], IncidentView] = field(
        repr=False,
        compare=False,
    )

    def is_live(self) -> bool:
        """Return whether the declaration still belongs to its Fiber activation."""

        return (
            self.owner_fiber.state is FiberState.ACTIVE
            and self.owner_fiber.activation_token is self.activation_token
            and self.health.healthy
        )


class ManagedProcessRegistry(Mapping[str, ManagedProcessBinding]):
    """Expose one immutable Root-local managed-process catalog."""

    def __init__(
        self,
        bindings: Mapping[str, ManagedProcessBinding],
        *,
        root_instance_token: object,
    ) -> None:
        self._root_instance_token = root_instance_token
        self._bindings = MappingProxyType(
            {name: bindings[name] for name in sorted(bindings)}
        )
        self._descriptors = tuple(
            binding.descriptor for binding in self._bindings.values()
        )
        payload = [
            {
                "owner": item.owner,
                "name": item.name,
                "command": list(item.command),
                "cwd": item.cwd,
                "env": list(item.env),
                "port_env": item.port_env,
                "formal_port": item.formal_port,
                "readiness_path": item.readiness_path,
                "startup_timeout_seconds": item.startup_timeout_seconds,
            }
            for item in self._descriptors
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
    def descriptors(self) -> tuple[ManagedProcessDescriptor, ...]:
        return self._descriptors

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def root_instance_token(self) -> object:
        return self._root_instance_token

    def __getitem__(self, name: str) -> ManagedProcessBinding:
        return self._bindings[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


MANAGED_PROCESSES = ServiceKey["PluginManagedProcesses"](
    "core.managed_processes"
)


@dataclass(slots=True)
class _ProcessRegistration:
    token: int
    owner: str
    definition: ManagedProcessDefinition
    descriptor: ManagedProcessDescriptor
    owner_fiber: FiberHandle
    activation_token: object
    runtime_plugin_dir: Path
    runtime_data_dir: Path
    runtime_workspace: Path
    incident_reporter: Callable[[str, str], IncidentView]
    health: HealthHandle | None = None


class _ManagedProcessDeclarations:
    """Own one Root-local mutable declaration set for Core."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _ProcessRegistration] = {}
        self._names: dict[str, int] = {}
        self._frozen: ManagedProcessRegistry | None = None

    async def register(
        self,
        ctx: Context,
        definition: ManagedProcessDefinition,
    ) -> None:
        """Validate and register one process declaration as Fiber-owned effects."""

        # 1. Freeze source-relative inputs before any Root state changes.
        normalized = _normalize_definition(ctx.runtime.plugin_dir, definition)
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError(
                "INACTIVE_FIBER",
                f"{ctx.runtime.plugin_id} 当前 Fiber 没有 active activation",
            )
        registration: _ProcessRegistration | None = None

        def setup() -> Callable[[], None]:
            nonlocal registration
            registration, cleanup = self._register(
                ctx.runtime.plugin_id,
                normalized,
                owner_fiber,
                activation_token,
                ctx.runtime.plugin_dir,
                ctx.runtime.data_dir,
                ctx.runtime.workspace,
                ctx.report_incident,
            )
            return cleanup

        # 2. Registration and required health either both settle or roll back.
        registration_effect = await ctx.effect(
            setup,
            label=f"managed-process:{normalized.name}",
        )
        try:
            health = await ctx.health(
                f"managed-process:{normalized.name}",
                required=True,
            )
        except BaseException:
            await registration_effect.aclose()
            raise
        assert registration is not None
        registration.health = health

    def freeze(self, root_instance_token: object) -> ManagedProcessRegistry:
        """Freeze declarations into an immutable snapshot registry."""

        if self._frozen is not None:
            if self._frozen.root_instance_token is not root_instance_token:
                raise RuntimeError("managed process registry 属于另一棵 Root")
            return self._frozen
        bindings: dict[str, ManagedProcessBinding] = {}
        for registration in sorted(
            self._registrations.values(), key=lambda item: item.token
        ):
            if registration.health is None:
                raise RuntimeError("managed process 声明缺少 required Health")
            bindings[registration.definition.name] = ManagedProcessBinding(
                descriptor=registration.descriptor,
                definition=registration.definition,
                health=registration.health,
                owner_fiber=registration.owner_fiber,
                activation_token=registration.activation_token,
                runtime_plugin_dir=registration.runtime_plugin_dir,
                runtime_data_dir=registration.runtime_data_dir,
                runtime_workspace=registration.runtime_workspace,
                incident_reporter=registration.incident_reporter,
            )
        self._frozen = ManagedProcessRegistry(
            bindings,
            root_instance_token=root_instance_token,
        )
        return self._frozen

    def _register(
        self,
        owner: str,
        definition: ManagedProcessDefinition,
        owner_fiber: FiberHandle,
        activation_token: object,
        runtime_plugin_dir: Path,
        runtime_data_dir: Path,
        runtime_workspace: Path,
        incident_reporter: Callable[[str, str], IncidentView],
    ) -> tuple[_ProcessRegistration, Callable[[], None]]:
        """Add one normalized declaration and return its exact inverse."""

        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_MANAGED_PROCESSES_FROZEN",
                "插件 managed process 声明已冻结",
            )
        if definition.name in self._names:
            raise CompositionError(
                "DUPLICATE_PLUGIN_MANAGED_PROCESS",
                f"插件 managed process 名称重复: {definition.name}",
            )
        token = self._next_token
        self._next_token += 1
        registration = _ProcessRegistration(
            token=token,
            owner=owner,
            definition=definition,
            descriptor=_descriptor(owner, definition),
            owner_fiber=owner_fiber,
            activation_token=activation_token,
            runtime_plugin_dir=runtime_plugin_dir,
            runtime_data_dir=runtime_data_dir,
            runtime_workspace=runtime_workspace,
            incident_reporter=incident_reporter,
        )
        self._registrations[token] = registration
        self._names[definition.name] = token

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)
            if self._names.get(definition.name) == token:
                _ = self._names.pop(definition.name)

        return registration, cleanup


class PluginManagedProcesses:
    """Expose only Fiber-owned process registration to plugins."""

    def __init__(self, root_instance_token: object) -> None:
        self._root_instance_token = root_instance_token
        self._declarations = _ManagedProcessDeclarations()

    async def register(
        self,
        ctx: Context,
        definition: ManagedProcessDefinition,
    ) -> None:
        """Register one process declaration through the Core-owned collector."""

        if (
            ctx._root_instance_token() is not self._root_instance_token
            or ctx.require(MANAGED_PROCESSES) is not self
        ):
            raise CompositionError(
                "MANAGED_PROCESS_SERVICE_ROOT_MISMATCH",
                "插件 managed process Service 不属于当前 Root",
            )
        await self._declarations.register(ctx, definition)


def _freeze_plugin_managed_processes(
    value: object,
    root_instance_token: object,
) -> ManagedProcessRegistry:
    """Freeze the exact Core-created process registration facade."""

    if not isinstance(value, PluginManagedProcesses):
        raise RuntimeError("RuntimeSnapshot managed process Service 类型无效")
    if value._root_instance_token is not root_instance_token:
        raise RuntimeError("RuntimeSnapshot managed process Service 不属于 exact Root")
    return value._declarations.freeze(root_instance_token)


def _normalize_definition(
    plugin_dir: Path,
    definition: ManagedProcessDefinition,
) -> ManagedProcessDefinition:
    """Validate and detach one plugin-owned process declaration."""

    if not isinstance(definition, ManagedProcessDefinition):
        raise TypeError("PluginManagedProcesses.register 只接受 ManagedProcessDefinition")
    if not isinstance(definition.name, str) or not _NAME.fullmatch(definition.name):
        raise ValueError(f"managed process name 无效: {definition.name}")
    command = _string_tuple(definition.command, "command", allow_empty=False)
    cwd = _relative_path(plugin_dir, definition.cwd, kind="cwd", directory=True)
    for item in command:
        if Path(item).is_absolute():
            raise ValueError("managed process command 不得声明绝对 artifact 路径")
        if item.startswith("-"):
            continue
        if "/" in item or "\\" in item or item.startswith(".") or item.endswith(".py"):
            _ = _relative_path(plugin_dir, item, kind="command", directory=False)
    env = _environment(definition.env)
    port_env = definition.port_env
    if (
        not isinstance(port_env, str)
        or not _ENV_NAME.fullmatch(port_env)
        or port_env in _RESERVED_ENV
        or port_env in env
    ):
        raise ValueError(f"managed process port_env 无效: {port_env}")
    formal_port = definition.formal_port
    if (
        not isinstance(formal_port, int)
        or isinstance(formal_port, bool)
        or not 1 <= formal_port <= 65535
    ):
        raise ValueError(f"managed process formal_port 无效: {formal_port}")
    readiness_path = _readiness_path(definition.readiness_path)
    timeout = definition.startup_timeout_seconds
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 300
    ):
        raise ValueError(f"managed process startup timeout 无效: {timeout}")
    return ManagedProcessDefinition(
        name=definition.name,
        command=command,
        cwd=cwd,
        env=MappingProxyType(env),
        port_env=port_env,
        formal_port=formal_port,
        readiness_path=readiness_path,
        startup_timeout_seconds=float(timeout),
    )


def _descriptor(
    owner: str,
    definition: ManagedProcessDefinition,
) -> ManagedProcessDescriptor:
    return ManagedProcessDescriptor(
        owner=owner,
        name=definition.name,
        command=definition.command,
        cwd=definition.cwd,
        env=tuple(sorted(definition.env.items())),
        port_env=definition.port_env,
        formal_port=definition.formal_port,
        readiness_path=definition.readiness_path,
        startup_timeout_seconds=definition.startup_timeout_seconds,
    )


def _environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("managed process env 必须是字符串 mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not _ENV_NAME.fullmatch(key)
            or key in _RESERVED_ENV
            or not isinstance(item, str)
        ):
            raise ValueError(f"managed process env 无效: {key}")
        result[key] = item
    return result


def _readiness_path(raw: str) -> str:
    if (
        not isinstance(raw, str)
        or not raw.startswith("/")
        or raw.startswith("//")
        or raw != raw.strip()
        or "\\" in raw
        or any(part in {".", ".."} for part in raw.split("/"))
    ):
        raise ValueError(f"managed process readiness_path 无效: {raw}")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"managed process readiness_path 无效: {raw}")
    return raw


def _string_tuple(
    value: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or (not value and not allow_empty):
        requirement = "非空 tuple" if not allow_empty else "tuple"
        raise ValueError(f"managed process {field_name} 必须是{requirement}")
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in value
    ):
        raise ValueError(f"managed process {field_name} 包含无效字符串")
    return tuple(value)


def _relative_path(
    plugin_dir: Path,
    raw: str,
    *,
    kind: str,
    directory: bool,
) -> str:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or Path(raw).is_absolute()
    ):
        raise ValueError(f"managed process {kind} 必须是 artifact 内相对路径")
    root = plugin_dir.resolve(strict=True)
    try:
        resolved = (root / raw).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"managed process {kind} 不存在: {raw}") from error
    valid_type = resolved.is_dir() if directory else resolved.is_file()
    if not resolved.is_relative_to(root) or not valid_type:
        raise ValueError(f"managed process {kind} 越过 immutable artifact: {raw}")
    return raw
