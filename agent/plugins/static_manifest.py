"""Static identity and runtime policy for external v3 plugin artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from urllib.parse import urlsplit

STATIC_MANIFEST_FILENAME = "akashic.plugin.toml"

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_CONFIG_KEY = re.compile(r"^[a-z][A-Za-z0-9_-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_RESERVED_ENV = frozenset(
    {
        "AKA_PLUGIN_DATA_DIR",
        "AKASHIC_PLUGIN_DATA_DIR",
        "AKASHIC_WORKSPACE",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "api_version",
        "entrypoint",
        "candidate_data_mode",
        "python",
        "validation",
        "mcp",
        "mcp_servers",
        "process",
        "processes",
        "managed_processes",
        "workload",
        "workloads",
        "channel_credentials",
    }
)
_PYTHON_COMMAND = re.compile(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?")


@dataclass(frozen=True, slots=True)
class StaticPythonRuntime:
    """One source-relative requirements file that must be staged before use."""

    requirements: str
    runtime_root: str


@dataclass(frozen=True, slots=True)
class StaticMcpDeclaration:
    """The import-free MCP declaration projection from an artifact manifest."""

    name: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    required_tools: tuple[str, ...]
    candidate_read_only_tools: tuple[str, ...]
    endpoint_env: tuple[tuple[str, str], ...]
    workload_env: tuple[tuple[str, str, str], ...]
    candidate_env: tuple[tuple[str, str], ...]
    python_runtime: str | None


@dataclass(frozen=True, slots=True)
class StaticManagedProcessDeclaration:
    """The import-free managed-process declaration projection."""

    name: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    port_env: str
    formal_port: int
    readiness_path: str
    startup_timeout_seconds: float
    python_runtime: str | None


@dataclass(frozen=True, slots=True)
class StaticWorkloadDeclaration:
    """The import-free Workload declaration projection."""

    name: str
    image: str
    command: tuple[str, ...]
    ports: tuple[tuple[str, int], ...]
    loopback_ports: tuple[tuple[str, int], ...]
    data: tuple[tuple[str, str, bool], ...]
    health: tuple[str, str, float]
    limits: tuple[int, float, int]
    user_namespaces: bool


@dataclass(frozen=True, slots=True)
class StaticPluginManifest:
    """Validated immutable identity, runtime and validation policy."""

    schema_version: int
    name: str
    version: str
    api_version: int
    entrypoint: str
    candidate_data_mode: Literal["isolated_copy", "shared_read"]
    python: tuple[StaticPythonRuntime, ...]
    exclude_data_paths: tuple[str, ...]
    mcp_servers: tuple[StaticMcpDeclaration, ...]
    managed_processes: tuple[StaticManagedProcessDeclaration, ...]
    workloads: tuple[StaticWorkloadDeclaration, ...]
    channel_credentials: tuple[tuple[str, tuple[str, ...]], ...]
    identity_digest: str

    @property
    def requirements(self) -> tuple[str, ...]:
        """Return all declared requirements paths in manifest order."""

        return tuple(runtime.requirements for runtime in self.python)

    @property
    def runtime(self) -> tuple[StaticPythonRuntime, ...]:
        """Return the immutable Python runtime declarations."""

        return self.python


def load_static_plugin_manifest(plugin_root: Path) -> StaticPluginManifest:
    """Parse and validate one artifact manifest without importing plugin code."""

    # 1. Resolve the artifact root without accepting a symlink as its owner.
    root = plugin_root.resolve(strict=True)
    if plugin_root.is_symlink() or not root.is_dir():
        raise ValueError(f"插件 artifact 根必须是普通目录: {plugin_root}")
    path = root / STATIC_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"v3 插件缺少静态 manifest: {path}")

    # 2. Parse only data; no module, callable or process is touched here.
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"插件静态 manifest 无法解析: {path}") from error
    return _validate_manifest(root, raw)


def validate_module_exports(
    manifest: StaticPluginManifest,
    module: object,
    *,
    plugin_root: Path | None = None,
) -> None:
    """Verify imported module identity matches its already validated manifest."""

    for field_name, expected in (
        ("api_version", manifest.api_version),
        ("name", manifest.name),
        ("version", manifest.version),
    ):
        actual = getattr(module, field_name, None)
        if actual != expected:
            raise ValueError(
                f"v3 插件 module.{field_name} 与静态 manifest 不一致: "
                f"expected={expected!r}, actual={actual!r}"
            )
    entrypoint = getattr(module, "__file__", None)
    if not isinstance(entrypoint, str):
        raise ValueError("v3 插件 module 缺少 __file__")
    imported_path = Path(entrypoint).resolve(strict=False)
    if plugin_root is not None:
        expected_path = (plugin_root / manifest.entrypoint).resolve(strict=False)
        if imported_path != expected_path:
            raise ValueError(
                "v3 插件 module entrypoint 与静态 manifest 不一致: "
                f"expected={expected_path}, actual={imported_path}"
            )
    elif imported_path.name != Path(manifest.entrypoint).name:
        raise ValueError("v3 插件 module entrypoint 无法核对")


def staged_python_interpreter(
    plugin_root: Path,
    runtime: StaticPythonRuntime,
) -> Path:
    """Return the executable staged for one manifest Python runtime."""

    root = plugin_root.resolve(strict=True)
    runtime_root = (root / runtime.runtime_root).resolve(strict=True)
    if not runtime_root.is_relative_to(root):
        raise ValueError("插件 Python runtime 越过 artifact")
    interpreter = _venv_python(runtime_root / ".venv")
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RuntimeError(
            "插件 Python runtime 尚未完成 staging: "
            f"requirements={runtime.requirements} interpreter={interpreter}"
        )
    return interpreter


def materialize_static_command(
    plugin_root: Path,
    manifest: StaticPluginManifest,
    declaration: StaticMcpDeclaration | StaticManagedProcessDeclaration,
) -> tuple[str, ...]:
    """Bind a static Python command to its staged artifact interpreter."""

    runtime_root = declaration.python_runtime
    if runtime_root is None:
        head = declaration.command[0]
        if _looks_like_artifact_path(head):
            executable = plugin_root.joinpath(
                *PurePosixPath(head).parts
            ).resolve(strict=True)
            if not executable.is_relative_to(plugin_root.resolve(strict=True)):
                raise RuntimeError("静态 command executable 越过 artifact")
            return (str(executable), *declaration.command[1:])
        return declaration.command
    runtime = next(
        (item for item in manifest.python if item.runtime_root == runtime_root),
        None,
    )
    if runtime is None:
        raise RuntimeError(f"静态 command 引用了未知 Python runtime: {runtime_root}")
    interpreter = staged_python_interpreter(plugin_root, runtime)
    return (str(interpreter), *declaration.command[1:])


def _validate_manifest(root: Path, raw: Mapping[str, object]) -> StaticPluginManifest:
    """Validate manifest identity, declarations and artifact-relative paths."""

    # 1. Reject fields for which Core has no static contract.
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"插件静态 manifest 包含未知字段: {unknown}")
    schema_version = _integer(raw, "schema_version")
    if schema_version != 1:
        raise ValueError("插件静态 manifest schema_version 必须为 1")
    name = _name(raw.get("name"), "name")
    version = _version(raw.get("version"), "version")
    api_version = _integer(raw, "api_version")
    if api_version != 3:
        raise ValueError("静态 artifact manifest 只接受 api_version = 3")
    entrypoint = _relative_artifact_path(
        root,
        raw.get("entrypoint"),
        label="entrypoint",
        must_exist=True,
        require_file=True,
    )
    if not entrypoint.endswith(".py"):
        raise ValueError("插件静态 manifest entrypoint 必须指向 Python 文件")
    candidate_data_mode = _candidate_data_mode(raw.get("candidate_data_mode"))

    # 2. Requirements are complete before the artifact is published.
    python = _python_runtimes(root, raw.get("python", []))
    exclude_data_paths = _validation_paths(root, raw.get("validation", {}))

    # 3. Optional declarations are checked statically and kept immutable.
    mcp_servers = _mcp_declarations(root, raw, python)
    managed_processes = _process_declarations(root, raw, python)
    workloads = _workload_declarations(raw)
    channel_credentials = _channel_credentials(raw.get("channel_credentials", {}))
    _validate_endpoint_process_refs(mcp_servers, managed_processes)
    _validate_endpoint_workload_refs(mcp_servers, workloads)
    identity: dict[str, object] = {
        "schema_version": schema_version,
        "name": name,
        "version": version,
        "api_version": api_version,
        "entrypoint": entrypoint,
        "python": [
            {
                "requirements": item.requirements,
                "runtime_root": item.runtime_root,
            }
            for item in python
        ],
        "exclude_data_paths": list(exclude_data_paths),
        "mcp_servers": [_mcp_identity(item) for item in mcp_servers],
        "managed_processes": [_process_identity(item) for item in managed_processes],
        "workloads": [_workload_identity(item) for item in workloads],
        "channel_credentials": [
            {"channel": channel, "paths": list(paths)}
            for channel, paths in channel_credentials
        ],
    }
    if "candidate_data_mode" in raw:
        identity["candidate_data_mode"] = candidate_data_mode
    identity_digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return StaticPluginManifest(
        schema_version=schema_version,
        name=name,
        version=version,
        api_version=api_version,
        entrypoint=entrypoint,
        candidate_data_mode=candidate_data_mode,
        python=python,
        exclude_data_paths=exclude_data_paths,
        mcp_servers=mcp_servers,
        managed_processes=managed_processes,
        workloads=workloads,
        channel_credentials=channel_credentials,
        identity_digest=identity_digest,
    )


def _candidate_data_mode(
    raw: object,
) -> Literal["isolated_copy", "shared_read"]:
    if raw is None:
        return "isolated_copy"
    if not isinstance(raw, str) or raw not in {"isolated_copy", "shared_read"}:
        raise ValueError(
            "插件静态 manifest candidate_data_mode 必须为 "
            "isolated_copy 或 shared_read"
        )
    return cast(Literal["isolated_copy", "shared_read"], raw)


def _channel_credentials(
    raw: object,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate import-free channel credential paths from the artifact manifest."""

    # 1. Each channel owns one sorted set of dotted config paths.
    table = _table(raw, "channel_credentials")
    result: list[tuple[str, tuple[str, ...]]] = []
    for channel, paths in sorted(table.items()):
        name = _name(channel, f"channel_credentials.{channel}")
        values = _string_list(paths, f"channel_credentials.{name}")
        normalized: list[str] = []
        for value in values:
            parts = value.split(".")
            if any(_CONFIG_KEY.fullmatch(part) is None for part in parts):
                raise ValueError(
                    f"channel_credentials.{name} 包含无效 config path: {value}"
                )
            normalized.append(".".join(parts))

        # 2. Prefix overlap would make redaction order observable.
        path_set = set(normalized)
        for value in normalized:
            parts = value.split(".")
            if any(
                ".".join(parts[:index]) in path_set for index in range(1, len(parts))
            ):
                raise ValueError(f"channel_credentials.{name} 路径重叠: {value}")
        result.append((name, tuple(sorted(normalized))))

    # 3. Two channels may reuse one exact credential, but not overlapping paths.
    all_paths = {path for _channel, paths in result for path in paths}
    for value in all_paths:
        parts = value.split(".")
        if any(".".join(parts[:index]) in all_paths for index in range(1, len(parts))):
            raise ValueError(f"channel_credentials 跨 channel 路径重叠: {value}")
    return tuple(result)


def _python_runtimes(
    root: Path,
    raw: object,
) -> tuple[StaticPythonRuntime, ...]:
    if not isinstance(raw, list):
        raise ValueError("插件静态 manifest python 必须是表数组")
    result: list[StaticPythonRuntime] = []
    seen: set[str] = set()
    runtime_roots: set[str] = set()
    for index, item in enumerate(raw):
        mapping = _table(item, f"python[{index}]")
        _exact_keys(mapping, {"requirements"}, f"python[{index}]")
        requirements = _relative_artifact_path(
            root,
            mapping.get("requirements"),
            label=f"python[{index}].requirements",
            must_exist=True,
            require_file=True,
        )
        if requirements in seen:
            raise ValueError(f"插件 requirements 重复: {requirements}")
        seen.add(requirements)
        runtime_root = str(PurePosixPath(requirements).parent)
        if runtime_root in runtime_roots:
            raise ValueError(f"插件 Python runtime root 重复: {runtime_root}")
        runtime_roots.add(runtime_root)
        result.append(
            StaticPythonRuntime(
                requirements=requirements,
                runtime_root=runtime_root,
            )
        )
    return tuple(result)


def _validation_paths(root: Path, raw: object) -> tuple[str, ...]:
    if raw == {}:
        return ()
    table = _table(raw, "validation")
    _exact_keys(table, {"exclude_data_paths"}, "validation")
    paths = table.get("exclude_data_paths", [])
    if not isinstance(paths, list):
        raise ValueError("validation.exclude_data_paths 必须是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(paths):
        normalized = _relative_policy_path(
            root,
            item,
            label=f"validation.exclude_data_paths[{index}]",
        )
        if normalized in seen:
            raise ValueError(f"validation.exclude_data_paths 重复: {normalized}")
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _mcp_declarations(
    root: Path,
    raw: Mapping[str, object],
    python: tuple[StaticPythonRuntime, ...],
) -> tuple[StaticMcpDeclaration, ...]:
    items = _alias_array(raw, ("mcp", "mcp_servers"), "MCP")
    result: list[StaticMcpDeclaration] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        table = _table(item, f"mcp[{index}]")
        allowed = {
            "name",
            "command",
            "cwd",
            "env",
            "required_tools",
            "candidate_read_only_tools",
            "endpoint_env",
            "workload_env",
            "candidate_env",
        }
        _exact_keys(table, allowed, f"mcp[{index}]")
        name = _name(table.get("name"), f"mcp[{index}].name")
        if name in seen:
            raise ValueError(f"MCP server 名称重复: {name}")
        seen.add(name)
        command = _command(root, table.get("command"), f"mcp[{index}].command")
        cwd = _relative_artifact_path(
            root,
            table.get("cwd", "."),
            label=f"mcp[{index}].cwd",
            must_exist=True,
            require_file=False,
        )
        env = _environment(table.get("env", {}), f"mcp[{index}].env")
        candidate_env = _environment(
            table.get("candidate_env", {}),
            f"mcp[{index}].candidate_env",
        )
        required_tools = _string_list(
            table.get("required_tools", []), f"mcp[{index}].required_tools"
        )
        candidate_tools = _string_list(
            table.get("candidate_read_only_tools", []),
            f"mcp[{index}].candidate_read_only_tools",
        )
        endpoint_env = _endpoint_env(
            table.get("endpoint_env", []), f"mcp[{index}].endpoint_env"
        )
        workload_env = _workload_env(
            table.get("workload_env", []), f"mcp[{index}].workload_env"
        )
        occupied = set(env) | set(candidate_env)
        endpoint_names = [item[0] for item in endpoint_env]
        endpoint_names.extend(item[0] for item in workload_env)
        if occupied.intersection(endpoint_names) or len(endpoint_names) != len(
            set(endpoint_names)
        ):
            raise ValueError(f"MCP endpoint env 与声明 env 冲突: {name}")
        python_runtime = _python_runtime_binding(
            root,
            command,
            cwd,
            python,
            label=f"mcp[{index}].command",
        )
        result.append(
            StaticMcpDeclaration(
                name=name,
                command=command,
                cwd=cwd,
                env=env,
                required_tools=required_tools,
                candidate_read_only_tools=candidate_tools,
                endpoint_env=endpoint_env,
                workload_env=workload_env,
                candidate_env=candidate_env,
                python_runtime=python_runtime,
            )
        )
    return tuple(result)


def _process_declarations(
    root: Path,
    raw: Mapping[str, object],
    python: tuple[StaticPythonRuntime, ...],
) -> tuple[StaticManagedProcessDeclaration, ...]:
    items = _alias_array(
        raw,
        ("process", "processes", "managed_processes"),
        "managed process",
    )
    result: list[StaticManagedProcessDeclaration] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        table = _table(item, f"process[{index}]")
        allowed = {
            "name",
            "command",
            "cwd",
            "env",
            "port_env",
            "formal_port",
            "readiness_path",
            "startup_timeout_seconds",
        }
        _exact_keys(table, allowed, f"process[{index}]")
        name = _name(table.get("name"), f"process[{index}].name")
        if name in seen:
            raise ValueError(f"managed process 名称重复: {name}")
        seen.add(name)
        command = _command(
            root,
            table.get("command"),
            f"process[{index}].command",
        )
        cwd = _relative_artifact_path(
            root,
            table.get("cwd", "."),
            label=f"process[{index}].cwd",
            must_exist=True,
            require_file=False,
        )
        env = _environment(table.get("env", {}), f"process[{index}].env")
        port_env = table.get("port_env")
        if (
            not isinstance(port_env, str)
            or not _ENV_NAME.fullmatch(port_env)
            or port_env in _RESERVED_ENV
        ):
            raise ValueError(f"process[{index}].port_env 无效")
        formal_port = table.get("formal_port")
        if (
            isinstance(formal_port, bool)
            or not isinstance(formal_port, int)
            or not 1 <= formal_port <= 65535
        ):
            raise ValueError(f"process[{index}].formal_port 无效")
        if port_env in dict(env):
            raise ValueError(f"process[{index}].env 不得覆盖 port_env: {port_env}")
        readiness_path = table.get("readiness_path", "/health")
        if (
            not isinstance(readiness_path, str)
            or not readiness_path.startswith("/")
            or readiness_path.startswith("//")
            or readiness_path != readiness_path.strip()
            or "\\" in readiness_path
            or any(part in {".", ".."} for part in readiness_path.split("/"))
        ):
            raise ValueError(f"process[{index}].readiness_path 无效")
        parsed_readiness = urlsplit(readiness_path)
        if (
            parsed_readiness.scheme
            or parsed_readiness.netloc
            or parsed_readiness.query
            or parsed_readiness.fragment
        ):
            raise ValueError(f"process[{index}].readiness_path 无效")
        timeout = table.get("startup_timeout_seconds", 15.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 0 < float(timeout) <= 300
        ):
            raise ValueError(f"process[{index}].startup_timeout_seconds 无效")
        python_runtime = _python_runtime_binding(
            root,
            command,
            cwd,
            python,
            label=f"process[{index}].command",
        )
        result.append(
            StaticManagedProcessDeclaration(
                name=name,
                command=command,
                cwd=cwd,
                env=env,
                port_env=port_env,
                formal_port=formal_port,
                readiness_path=readiness_path,
                startup_timeout_seconds=float(timeout),
                python_runtime=python_runtime,
            )
        )
    return tuple(result)


def _validate_endpoint_process_refs(
    servers: tuple[StaticMcpDeclaration, ...],
    processes: tuple[StaticManagedProcessDeclaration, ...],
) -> None:
    names = {item.name for item in processes}
    for server in servers:
        for _, process in server.endpoint_env:
            if process not in names:
                raise ValueError(
                    f"MCP endpoint_env 引用了未声明的 managed process: {process}"
                )


def _workload_declarations(
    raw: Mapping[str, object],
) -> tuple[StaticWorkloadDeclaration, ...]:
    """Validate fixed Workload data without importing plugin code."""

    items = _alias_array(raw, ("workload", "workloads"), "Workload")
    result: list[StaticWorkloadDeclaration] = []
    seen: set[str] = set()
    image_pattern = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
    for index, item in enumerate(items):
        table = _table(item, f"workload[{index}]")
        _exact_keys(
            table,
            {
                "name",
                "image",
                "command",
                "ports",
                "data",
                "health",
                "limits",
                "user_namespaces",
            },
            f"workload[{index}]",
        )
        name = _name(table.get("name"), f"workload[{index}].name")
        if name in seen:
            raise ValueError(f"Workload 名称重复: {name}")
        seen.add(name)
        image = table.get("image")
        if not isinstance(image, str) or image_pattern.fullmatch(image) is None:
            raise ValueError(f"workload[{index}].image 必须使用 sha256 digest")
        command = _string_list(table.get("command", []), f"workload[{index}].command")
        if not command:
            raise ValueError(f"workload[{index}].command 不能为空")
        ports, loopback_ports = _workload_ports(
            table.get("ports"), f"workload[{index}].ports"
        )
        data = _workload_data(table.get("data", []), f"workload[{index}].data")
        health = _workload_health(
            table.get("health"), ports, f"workload[{index}].health"
        )
        limits = _workload_limits(table.get("limits"), f"workload[{index}].limits")
        user_namespaces = table.get("user_namespaces", False)
        if not isinstance(user_namespaces, bool):
            raise ValueError(f"workload[{index}].user_namespaces 必须是 bool")
        result.append(
            StaticWorkloadDeclaration(
                name,
                image,
                command,
                ports,
                loopback_ports,
                data,
                health,
                limits,
                user_namespaces,
            )
        )
    return tuple(result)


def _validate_endpoint_workload_refs(
    servers: tuple[StaticMcpDeclaration, ...],
    workloads: tuple[StaticWorkloadDeclaration, ...],
) -> None:
    ports = {item.name: {name for name, _ in item.ports} for item in workloads}
    for server in servers:
        for _, workload, port in server.workload_env:
            if workload not in ports or port not in ports[workload]:
                raise ValueError(
                    "MCP workload_env 引用了未声明的 Workload 端口: "
                    f"{workload}:{port}"
                )


def _alias_array(
    raw: Mapping[str, object], aliases: tuple[str, ...], label: str
) -> list[object]:
    present = [name for name in aliases if name in raw]
    if len(present) > 1:
        raise ValueError(f"{label} 声明不能同时使用: {present}")
    if not present:
        return []
    value = raw[present[0]]
    if not isinstance(value, list):
        raise ValueError(f"{label} 声明必须是表数组")
    return cast(list[object], value)


def _command(root: Path, raw: object, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} 必须是非空字符串数组")
    values = _string_list(raw, label)
    for index, value in enumerate(values):
        if _is_absolute_path(value):
            raise ValueError(f"{label}[{index}] 不得是 artifact 外绝对路径")
        if _looks_like_artifact_path(value):
            _ = _relative_artifact_path(
                root,
                value,
                label=f"{label}[{index}] path",
                must_exist=True,
                require_file=True,
            )
    return values


def _python_runtime_binding(
    root: Path,
    command: tuple[str, ...],
    cwd: str,
    runtimes: tuple[StaticPythonRuntime, ...],
    *,
    label: str,
) -> str | None:
    """Resolve a Python command to exactly one staged runtime root."""

    if _PYTHON_COMMAND.fullmatch(PurePosixPath(command[0]).name.lower()) is None:
        return None
    target = root.joinpath(*PurePosixPath(cwd).parts).resolve(strict=True)
    for item in command[1:]:
        if item.startswith("-"):
            continue
        if _looks_like_artifact_path(item):
            target = root.joinpath(*PurePosixPath(item).parts).resolve(strict=True)
        break
    matches = tuple(
        runtime
        for runtime in runtimes
        if target.is_relative_to(
            root.joinpath(*PurePosixPath(runtime.runtime_root).parts).resolve(
                strict=True
            )
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"{label} 必须唯一绑定已声明 Python runtime: "
            f"matches={[item.runtime_root for item in matches]}"
        )
    return matches[0].runtime_root


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _endpoint_env(raw: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} 必须是表数组")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        table = _table(item, f"{label}[{index}]")
        _exact_keys(table, {"env", "process"}, f"{label}[{index}]")
        env = table.get("env")
        process = table.get("process")
        if (
            not isinstance(env, str)
            or not _ENV_NAME.fullmatch(env)
            or env in _RESERVED_ENV
            or not isinstance(process, str)
            or not _NAME.fullmatch(process)
        ):
            raise ValueError(f"{label}[{index}] 无效")
        if env in seen:
            raise ValueError(f"{label} 环境变量重复: {env}")
        seen.add(env)
        result.append((env, process))
    return tuple(result)


def _workload_env(
    raw: object,
    label: str,
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} 必须是表数组")
    result: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        table = _table(item, f"{label}[{index}]")
        _exact_keys(table, {"env", "workload", "port"}, f"{label}[{index}]")
        env = table.get("env")
        workload = table.get("workload")
        port = table.get("port")
        if (
            not isinstance(env, str)
            or not _ENV_NAME.fullmatch(env)
            or env in _RESERVED_ENV
            or not isinstance(workload, str)
            or not _NAME.fullmatch(workload)
            or not isinstance(port, str)
            or not _NAME.fullmatch(port)
        ):
            raise ValueError(f"{label}[{index}] 无效")
        if env in seen:
            raise ValueError(f"{label} 环境变量重复: {env}")
        seen.add(env)
        result.append((env, workload, port))
    return tuple(result)


def _workload_ports(
    raw: object,
    label: str,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} 必须是非空表数组")
    result: list[tuple[str, int]] = []
    names: set[str] = set()
    numbers: set[int] = set()
    loopback_numbers: set[int] = set()
    loopback_ports: list[tuple[str, int]] = []
    for index, item in enumerate(raw):
        table = _table(item, f"{label}[{index}]")
        _exact_keys(table, {"name", "number", "loopback"}, f"{label}[{index}]")
        name = _name(table.get("name"), f"{label}[{index}].name")
        number = table.get("number")
        loopback = table.get("loopback")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= 65535
            or name in names
            or number in numbers
            or (
                loopback is not None
                and (
                    isinstance(loopback, bool)
                    or not isinstance(loopback, int)
                    or not 1024 <= loopback <= 65535
                    or loopback in loopback_numbers
                )
            )
        ):
            raise ValueError(f"{label}[{index}] 无效")
        names.add(name)
        numbers.add(number)
        result.append((name, number))
        if loopback is not None:
            loopback_numbers.add(loopback)
            loopback_ports.append((name, loopback))
    return tuple(result), tuple(loopback_ports)


def _workload_data(
    raw: object,
    label: str,
) -> tuple[tuple[str, str, bool], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} 必须是表数组")
    result: list[tuple[str, str, bool]] = []
    names: set[str] = set()
    targets: set[str] = set()
    for index, item in enumerate(raw):
        table = _table(item, f"{label}[{index}]")
        _exact_keys(table, {"name", "target", "writable"}, f"{label}[{index}]")
        name = _name(table.get("name"), f"{label}[{index}].name")
        target = table.get("target")
        writable = table.get("writable", True)
        if not isinstance(target, str) or target != target.strip():
            raise ValueError(f"{label}[{index}].target 无效")
        path = PurePosixPath(target)
        if (
            not path.is_absolute()
            or path == PurePosixPath("/")
            or ".." in path.parts
            or not isinstance(writable, bool)
            or name in names
            or str(path) in targets
        ):
            raise ValueError(f"{label}[{index}] 无效")
        names.add(name)
        targets.add(str(path))
        result.append((name, str(path), writable))
    return tuple(result)


def _workload_health(
    raw: object,
    ports: tuple[tuple[str, int], ...],
    label: str,
) -> tuple[str, str, float]:
    table = _table(raw, label)
    _exact_keys(table, {"port", "path", "timeout_seconds"}, label)
    port = table.get("port")
    path = table.get("path", "/health")
    timeout = table.get("timeout_seconds", 60.0)
    if not isinstance(port, str) or port not in {name for name, _ in ports}:
        raise ValueError(f"{label}.port 无效")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or path != path.strip()
        or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"{label}.path 无效")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"{label}.path 无效")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 300
    ):
        raise ValueError(f"{label}.timeout_seconds 无效")
    return port, path, float(timeout)


def _workload_limits(raw: object, label: str) -> tuple[int, float, int]:
    table = _table(raw, label)
    _exact_keys(table, {"memory_mb", "cpu_count", "pids"}, label)
    memory = table.get("memory_mb")
    cpu = table.get("cpu_count")
    pids = table.get("pids")
    if (
        isinstance(memory, bool)
        or not isinstance(memory, int)
        or not (memory == 0 or 64 <= memory <= 262_144)
        or isinstance(cpu, bool)
        or not isinstance(cpu, (int, float))
        or not math.isfinite(float(cpu))
        or not (float(cpu) == 0 or 0.1 <= float(cpu) <= 256)
        or isinstance(pids, bool)
        or not isinstance(pids, int)
        or not (pids == 0 or 16 <= pids <= 1_048_576)
    ):
        raise ValueError(f"{label} 无效")
    return memory, float(cpu), pids


def _environment(raw: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} 必须是字符串映射")
    values: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw).items():
        if (
            not isinstance(key, str)
            or not _ENV_NAME.fullmatch(key)
            or not isinstance(value, str)
        ):
            raise ValueError(f"{label} 包含无效环境变量")
        if key in _RESERVED_ENV:
            raise ValueError(f"{label} 不得覆盖 Core 保留环境变量: {key}")
        values[key] = value
    return tuple(sorted(values.items()))


def _string_list(raw: object, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item and item == item.strip() for item in raw
    ):
        raise ValueError(f"{label} 必须是非空字符串数组")
    values = tuple(cast(str, item) for item in raw)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} 不得重复")
    return values


def _relative_artifact_path(
    root: Path,
    raw: object,
    *,
    label: str,
    must_exist: bool,
    require_file: bool,
) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ValueError(f"{label} 必须是非空相对路径")
    path = PurePosixPath(raw.replace("\\", "/"))
    if _is_absolute_path(raw) or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} 必须是 artifact 内的相对路径")
    resolved = root.joinpath(*path.parts)
    _reject_symlink_ancestors(root, resolved, label)
    if must_exist:
        if not resolved.exists():
            raise ValueError(f"{label} 不存在: {raw}")
        if require_file and not resolved.is_file():
            raise ValueError(f"{label} 必须是文件: {raw}")
        if not require_file and not resolved.is_dir():
            raise ValueError(f"{label} 必须是目录: {raw}")
    return "/".join(path.parts) or "."


def _relative_policy_path(root: Path, raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ValueError(f"{label} 必须是非空相对路径")
    path = PurePosixPath(raw.replace("\\", "/"))
    if (
        not path.parts
        or _is_absolute_path(raw)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} 必须是 artifact/data 内的相对路径")
    resolved = root.joinpath(*path.parts)
    _reject_symlink_ancestors(root, resolved, label)
    return "/".join(path.parts)


def _reject_symlink_ancestors(root: Path, path: Path, label: str) -> None:
    current = root
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} 越界: {path}") from error
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} 不能经过符号链接: {current}")


def _looks_like_artifact_path(value: str) -> bool:
    return (
        "/" in value or "\\" in value or value.startswith(".") or value.endswith(".py")
    )


def _is_absolute_path(value: str) -> bool:
    """Reject POSIX and Windows absolute paths before PurePosix normalization."""

    return Path(value).is_absolute() or bool(re.match(r"^[A-Za-z]:[/\\]", value))


def _table(raw: object, label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} 必须是表")
    return cast(dict[str, object], raw)


def _exact_keys(raw: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} 包含未知字段: {unknown}")


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"插件静态 manifest {key} 必须是整数")
    return value


def _name(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not _NAME.fullmatch(raw):
        raise ValueError(f"插件静态 manifest {label} 无效")
    return raw


def _version(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not _VERSION.fullmatch(raw):
        raise ValueError(f"插件静态 manifest {label} 无效")
    return raw


def _mcp_identity(item: StaticMcpDeclaration) -> dict[str, object]:
    return {
        "name": item.name,
        "command": list(item.command),
        "cwd": item.cwd,
        "env": list(item.env),
        "required_tools": list(item.required_tools),
        "candidate_read_only_tools": list(item.candidate_read_only_tools),
        "endpoint_env": [list(value) for value in item.endpoint_env],
        "workload_env": [list(value) for value in item.workload_env],
        "candidate_env": list(item.candidate_env),
        "python_runtime": item.python_runtime,
    }


def _process_identity(item: StaticManagedProcessDeclaration) -> dict[str, object]:
    return {
        "name": item.name,
        "command": list(item.command),
        "cwd": item.cwd,
        "env": list(item.env),
        "port_env": item.port_env,
        "formal_port": item.formal_port,
        "readiness_path": item.readiness_path,
        "startup_timeout_seconds": item.startup_timeout_seconds,
        "python_runtime": item.python_runtime,
    }


def _workload_identity(item: StaticWorkloadDeclaration) -> dict[str, object]:
    return {
        "name": item.name,
        "image": item.image,
        "command": list(item.command),
        "ports": [list(value) for value in item.ports],
        "loopback_ports": [list(value) for value in item.loopback_ports],
        "data": [list(value) for value in item.data],
        "health": list(item.health),
        "limits": list(item.limits),
        "user_namespaces": item.user_namespaces,
    }
