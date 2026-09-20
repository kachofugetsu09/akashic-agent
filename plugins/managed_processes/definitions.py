from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit
from agent.plugin_composition.process_slots import ManagedProcessDefinition

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_RESERVED_ENV = frozenset(
    {
        "AKA_PLUGIN_DATA_DIR",
        "AKASHIC_PLUGIN_DATA_DIR",
        "AKASHIC_WORKSPACE",
        "AKASHIC_BOOT_ID",
        "AKASHIC_SUPERVISED",
    }
)


def _normalize_definition(
    plugin_dir: Path,
    definition: ManagedProcessDefinition,
) -> ManagedProcessDefinition:
    """Validate and detach one plugin-owned process declaration."""

    if not isinstance(definition, ManagedProcessDefinition):
        raise TypeError("ManagedProcesses.register 只接受 ManagedProcessDefinition")
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
    candidate_env = _environment(definition.candidate_env)
    port_env = definition.port_env
    if (
        not isinstance(port_env, str)
        or not _ENV_NAME.fullmatch(port_env)
        or port_env in _RESERVED_ENV
        or port_env in env
        or port_env in candidate_env
    ):
        raise ValueError(f"managed process port_env 无效: {port_env}")
    formal_port = definition.formal_port
    if (
        not isinstance(formal_port, int)
        or isinstance(formal_port, bool)
        or not 0 <= formal_port <= 65535
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
        candidate_env=MappingProxyType(candidate_env),
        port_env=port_env,
        formal_port=formal_port,
        readiness_path=readiness_path,
        startup_timeout_seconds=float(timeout),
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
