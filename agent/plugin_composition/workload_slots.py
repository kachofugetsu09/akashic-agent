from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from urllib.parse import urlsplit

from agent.plugin_composition.context import Context, FiberHandle, HealthHandle
from agent.plugin_composition.model import (
    CompositionError,
    IncidentView,
    ServiceKey,
)

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class WorkloadPort:
    name: str
    number: int
    loopback: int | None = None


@dataclass(frozen=True, slots=True)
class WorkloadData:
    name: str
    target: str
    writable: bool = True


@dataclass(frozen=True, slots=True)
class WorkloadHealth:
    port: str
    path: str = "/health"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class WorkloadLimits:
    """Limit workload resources; zero leaves that resource unlimited."""

    memory_mb: int
    cpu_count: float
    pids: int


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    image: str
    command: tuple[str, ...]
    ports: tuple[WorkloadPort, ...]
    data: tuple[WorkloadData, ...]
    health: WorkloadHealth
    limits: WorkloadLimits
    user_namespaces: bool = False


@dataclass(frozen=True, slots=True)
class WorkloadDescriptor:
    owner: str
    name: str
    image: str
    command: tuple[str, ...]
    ports: tuple[WorkloadPort, ...]
    data: tuple[WorkloadData, ...]
    health: WorkloadHealth
    limits: WorkloadLimits
    user_namespaces: bool


@dataclass(frozen=True, slots=True)
class WorkloadBinding:
    descriptor: WorkloadDescriptor
    health: HealthHandle
    owner_fiber: FiberHandle
    activation_token: object
    incident_reporter: Callable[[str, str], IncidentView]


class WorkloadRegistry(Mapping[str, WorkloadBinding]):
    """Expose one immutable Root-local workload catalog."""

    def __init__(
        self,
        bindings: Mapping[str, WorkloadBinding],
        *,
        root_instance_token: object,
    ) -> None:
        self._root_instance_token = root_instance_token
        self._bindings = MappingProxyType(
            {key: bindings[key] for key in sorted(bindings)}
        )
        self._descriptors = tuple(
            binding.descriptor for binding in self._bindings.values()
        )
        payload = [_descriptor_value(item) for item in self._descriptors]
        self._identity = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def descriptors(self) -> tuple[WorkloadDescriptor, ...]:
        return self._descriptors

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def root_instance_token(self) -> object:
        return self._root_instance_token

    def owned(self, owner: str, name: str) -> WorkloadBinding | None:
        return self._bindings.get(_binding_key(owner, name))

    def __getitem__(self, key: str) -> WorkloadBinding:
        return self._bindings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


WORKLOADS = ServiceKey["PluginWorkloads"]("core.workloads")


@dataclass(slots=True)
class _WorkloadRegistration:
    token: int
    descriptor: WorkloadDescriptor
    owner_fiber: FiberHandle
    activation_token: object
    incident_reporter: Callable[[str, str], IncidentView]
    health: HealthHandle | None = None


class _WorkloadDeclarations:
    """Own one Root-local mutable workload declaration set."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _WorkloadRegistration] = {}
        self._names: dict[tuple[str, str], int] = {}
        self._frozen: WorkloadRegistry | None = None

    async def register(self, ctx: Context, workload: Workload) -> None:
        """Validate and register one Fiber-owned workload declaration."""

        normalized = _normalize_workload(workload)
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError(
                "INACTIVE_FIBER",
                f"{ctx.runtime.plugin_id} 当前 Fiber 没有 active activation",
            )
        registration: _WorkloadRegistration | None = None

        def setup() -> Callable[[], None]:
            nonlocal registration
            registration, cleanup = self._register(
                ctx.runtime.plugin_id,
                normalized,
                owner_fiber,
                activation_token,
                ctx.report_incident,
            )
            return cleanup

        effect = await ctx.effect(setup, label=f"workload:{normalized.name}")
        try:
            health = await ctx.health(f"workload:{normalized.name}", required=True)
        except BaseException:
            await effect.aclose()
            raise
        assert registration is not None
        registration.health = health

    def freeze(self, root_instance_token: object) -> WorkloadRegistry:
        """Freeze declarations into an immutable snapshot registry."""

        if self._frozen is not None:
            if self._frozen.root_instance_token is not root_instance_token:
                raise RuntimeError("workload registry 属于另一棵 Root")
            return self._frozen
        bindings: dict[str, WorkloadBinding] = {}
        writable_data: set[tuple[str, str]] = set()
        for item in sorted(self._registrations.values(), key=lambda value: value.token):
            if item.health is None:
                raise RuntimeError("workload 声明缺少 required Health")
            for data in item.descriptor.data:
                claim = (item.descriptor.owner, data.name)
                if data.writable and claim in writable_data:
                    raise CompositionError(
                        "DUPLICATE_WORKLOAD_WRITER",
                        "插件 Workload data 有多个 writer: "
                        f"{item.descriptor.owner}:{data.name}",
                    )
                if data.writable:
                    writable_data.add(claim)
            bindings[
                _binding_key(item.descriptor.owner, item.descriptor.name)
            ] = WorkloadBinding(
                descriptor=item.descriptor,
                health=item.health,
                owner_fiber=item.owner_fiber,
                activation_token=item.activation_token,
                incident_reporter=item.incident_reporter,
            )
        self._frozen = WorkloadRegistry(
            bindings,
            root_instance_token=root_instance_token,
        )
        return self._frozen

    def _register(
        self,
        owner: str,
        workload: Workload,
        owner_fiber: FiberHandle,
        activation_token: object,
        incident_reporter: Callable[[str, str], IncidentView],
    ) -> tuple[_WorkloadRegistration, Callable[[], None]]:
        """Add one normalized declaration and return its exact inverse."""

        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_WORKLOADS_FROZEN", "插件 Workload 声明已冻结"
            )
        claim = (owner, workload.name)
        if claim in self._names:
            raise CompositionError(
                "DUPLICATE_PLUGIN_WORKLOAD",
                f"插件 Workload 名称重复: {owner}:{workload.name}",
            )
        token = self._next_token
        self._next_token += 1
        registration = _WorkloadRegistration(
            token=token,
            descriptor=_descriptor(owner, workload),
            owner_fiber=owner_fiber,
            activation_token=activation_token,
            incident_reporter=incident_reporter,
        )
        self._registrations[token] = registration
        self._names[claim] = token

        def cleanup() -> None:
            self._registrations.pop(token, None)
            if self._names.get(claim) == token:
                self._names.pop(claim)

        return registration, cleanup


class PluginWorkloads:
    """Expose only Fiber-owned workload registration to plugins."""

    def __init__(self, root_instance_token: object) -> None:
        self._root_instance_token = root_instance_token
        self._declarations = _WorkloadDeclarations()

    async def register(self, ctx: Context, workload: Workload) -> None:
        """Register one workload through the Core-owned collector."""

        if (
            ctx._root_instance_token() is not self._root_instance_token
            or ctx.require(WORKLOADS) is not self
        ):
            raise CompositionError(
                "WORKLOAD_SERVICE_ROOT_MISMATCH",
                "插件 Workload Service 不属于当前 Root",
            )
        await self._declarations.register(ctx, workload)


def _freeze_plugin_workloads(
    value: object,
    root_instance_token: object,
) -> WorkloadRegistry:
    """Freeze the exact Core-created workload registration facade."""

    if not isinstance(value, PluginWorkloads):
        raise RuntimeError("RuntimeSnapshot Workload Service 类型无效")
    if value._root_instance_token is not root_instance_token:
        raise RuntimeError("RuntimeSnapshot Workload Service 不属于 exact Root")
    return value._declarations.freeze(root_instance_token)


def _normalize_workload(value: Workload) -> Workload:
    """Validate and detach one plugin-owned workload declaration."""

    if not isinstance(value, Workload):
        raise TypeError("PluginWorkloads.register 只接受 Workload")
    name = _name(value.name, "Workload name")
    if not isinstance(value.image, str) or not _IMAGE.fullmatch(value.image):
        raise ValueError(f"Workload image 必须使用 sha256 digest: {value.image}")
    command = _strings(value.command, "command")
    ports = _ports(value.ports)
    data = _data(value.data)
    health = _health(value.health, ports)
    limits = _limits(value.limits)
    if not isinstance(value.user_namespaces, bool):
        raise TypeError("Workload user_namespaces 必须是 bool")
    return Workload(
        name=name,
        image=value.image,
        command=command,
        ports=ports,
        data=data,
        health=health,
        limits=limits,
        user_namespaces=value.user_namespaces,
    )


def _descriptor(owner: str, value: Workload) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        owner=owner,
        name=value.name,
        image=value.image,
        command=value.command,
        ports=value.ports,
        data=value.data,
        health=value.health,
        limits=value.limits,
        user_namespaces=value.user_namespaces,
    )


def _descriptor_value(value: WorkloadDescriptor) -> dict[str, object]:
    return {
        "owner": value.owner,
        "name": value.name,
        "image": value.image,
        "command": list(value.command),
        "ports": [
            {"name": item.name, "number": item.number, "loopback": item.loopback}
            for item in value.ports
        ],
        "data": [
            {"name": item.name, "target": item.target, "writable": item.writable}
            for item in value.data
        ],
        "health": {
            "port": value.health.port,
            "path": value.health.path,
            "timeout_seconds": value.health.timeout_seconds,
        },
        "limits": {
            "memory_mb": value.limits.memory_mb,
            "cpu_count": value.limits.cpu_count,
            "pids": value.limits.pids,
        },
        "user_namespaces": value.user_namespaces,
    }


def _ports(value: tuple[WorkloadPort, ...]) -> tuple[WorkloadPort, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("Workload ports 必须是非空 tuple")
    result: list[WorkloadPort] = []
    names: set[str] = set()
    numbers: set[int] = set()
    loopback_ports: set[int] = set()
    for item in value:
        if not isinstance(item, WorkloadPort):
            raise TypeError("Workload ports 只接受 WorkloadPort")
        name = _name(item.name, "Workload port name")
        loopback = item.loopback
        if (
            not isinstance(item.number, int)
            or isinstance(item.number, bool)
            or not 1 <= item.number <= 65535
            or name in names
            or item.number in numbers
            or (
                loopback is not None
                and (
                    not isinstance(loopback, int)
                    or isinstance(loopback, bool)
                    or not 1024 <= loopback <= 65535
                    or loopback in loopback_ports
                )
            )
        ):
            raise ValueError(f"Workload port 无效: {item!r}")
        names.add(name)
        numbers.add(item.number)
        if loopback is not None:
            loopback_ports.add(loopback)
        result.append(WorkloadPort(name, item.number, loopback))
    return tuple(result)


def _data(value: tuple[WorkloadData, ...]) -> tuple[WorkloadData, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Workload data 必须是 tuple")
    result: list[WorkloadData] = []
    names: set[str] = set()
    targets: set[str] = set()
    for item in value:
        if not isinstance(item, WorkloadData) or not isinstance(item.writable, bool):
            raise TypeError("Workload data 只接受 WorkloadData")
        name = _name(item.name, "Workload data name")
        target = _container_path(item.target)
        if name in names or target in targets:
            raise ValueError(f"Workload data 重复: {item!r}")
        names.add(name)
        targets.add(target)
        result.append(WorkloadData(name, target, item.writable))
    return tuple(result)


def _health(
    value: WorkloadHealth,
    ports: tuple[WorkloadPort, ...],
) -> WorkloadHealth:
    if not isinstance(value, WorkloadHealth):
        raise TypeError("Workload health 必须是 WorkloadHealth")
    if value.port not in {item.name for item in ports}:
        raise ValueError(f"Workload health port 不存在: {value.port}")
    path = _health_path(value.path)
    timeout = value.timeout_seconds
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 300
    ):
        raise ValueError(f"Workload health timeout 无效: {timeout}")
    return WorkloadHealth(value.port, path, float(timeout))


def _limits(value: WorkloadLimits) -> WorkloadLimits:
    if not isinstance(value, WorkloadLimits):
        raise TypeError("Workload limits 必须是 WorkloadLimits")
    if (
        not isinstance(value.memory_mb, int)
        or isinstance(value.memory_mb, bool)
        or not (value.memory_mb == 0 or 64 <= value.memory_mb <= 262_144)
        or not isinstance(value.cpu_count, (int, float))
        or isinstance(value.cpu_count, bool)
        or not math.isfinite(float(value.cpu_count))
        or not (float(value.cpu_count) == 0 or 0.1 <= float(value.cpu_count) <= 256)
        or not isinstance(value.pids, int)
        or isinstance(value.pids, bool)
        or not (value.pids == 0 or 16 <= value.pids <= 1_048_576)
    ):
        raise ValueError(f"Workload limits 无效: {value!r}")
    return WorkloadLimits(value.memory_mb, float(value.cpu_count), value.pids)


def _health_path(raw: str) -> str:
    if (
        not isinstance(raw, str)
        or not raw.startswith("/")
        or raw.startswith("//")
        or raw != raw.strip()
        or "\\" in raw
        or any(part in {".", ".."} for part in raw.split("/"))
    ):
        raise ValueError(f"Workload health path 无效: {raw}")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Workload health path 无效: {raw}")
    return raw


def _container_path(raw: str) -> str:
    if not isinstance(raw, str) or raw != raw.strip():
        raise ValueError(f"Workload data target 无效: {raw}")
    path = PurePosixPath(raw)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"Workload data target 无效: {raw}")
    return str(path)


def _strings(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"Workload {field_name} 必须是 tuple")
    if any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise ValueError(f"Workload {field_name} 包含无效字符串")
    return tuple(value)


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{label} 无效: {value}")
    return value


def _binding_key(owner: str, name: str) -> str:
    return f"{owner}/{name}"
