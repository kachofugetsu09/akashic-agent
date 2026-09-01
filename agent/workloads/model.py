from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

WorkloadMode = Literal["candidate", "formal"]


@dataclass(frozen=True, slots=True)
class WorkloadEndpoint:
    name: str
    url: str


@dataclass(frozen=True, slots=True)
class WorkloadLease:
    workspace_id: str
    plugin_id: str
    workload: str
    mode: WorkloadMode
    transaction_id: str
    generation_id: str
    container_id: str
    spec_digest: str


@dataclass(frozen=True, slots=True)
class WorkloadStartRequest:
    workspace_id: str
    plugin_id: str
    workload: str
    mode: WorkloadMode
    transaction_id: str
    generation_id: str
    spec_digest: str
    image: str
    command: tuple[str, ...]
    ports: tuple[tuple[str, int], ...]
    data: tuple[tuple[str, str, bool], ...]
    health: tuple[str, str, float]
    limits: tuple[int, float, int]
    loopback_ports: tuple[tuple[str, int], ...] = ()
    user_namespaces: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkloadStartReceipt:
    lease: WorkloadLease
    endpoints: tuple[WorkloadEndpoint, ...]
    adopted_from_generation: str | None


@dataclass(frozen=True, slots=True)
class WorkloadStopReceipt:
    lease: WorkloadLease
    container_absent: bool
    mounts_released: bool


def workload_spec_digest(
    *,
    plugin_id: str,
    workload: str,
    image: str,
    command: tuple[str, ...],
    ports: tuple[tuple[str, int], ...],
    data: tuple[tuple[str, str, bool], ...],
    health: tuple[str, str, float],
    limits: tuple[int, float, int],
    loopback_ports: tuple[tuple[str, int], ...] = (),
    user_namespaces: bool = False,
) -> str:
    """Hash the complete immutable Workload spec with one fixed encoding."""

    value = {
        "owner": plugin_id,
        "name": workload,
        "image": image,
        "command": list(command),
        "ports": [list(item) for item in ports],
        "data": [list(item) for item in data],
        "health": list(health),
        "limits": list(limits),
        "loopback_ports": [list(item) for item in loopback_ports],
        "user_namespaces": user_namespaces,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
