from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit
from agent.plugin_composition.context import FiberHandle, HealthHandle
from agent.plugin_composition.model import IncidentView
from agent.plugin_composition.workload_slots import Workload, WorkloadPort, WorkloadData, WorkloadHealth, WorkloadLimits

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


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


def _normalize_workload(value: Workload) -> Workload:
    """Validate and detach one plugin-owned workload declaration."""

    if not isinstance(value, Workload):
        raise TypeError("Workloads.register 只接受 Workload")
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
