"""Static identity and runtime policy for external v3 plugin artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

STATIC_MANIFEST_FILENAME = "akashic.plugin.toml"

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_CONFIG_KEY = re.compile(r"^[a-z][A-Za-z0-9_-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "api_version",
        "entrypoint",
        "python",
        "validation",
        "channel_credentials",
        "credential_paths",
    }
)
_PYTHON_COMMAND = re.compile(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?")


@dataclass(frozen=True, slots=True)
class StaticPythonRuntime:
    """One source-relative requirements file that must be staged before use."""

    requirements: str
    runtime_root: str


@dataclass(frozen=True, slots=True)
class StaticPluginManifest:
    """Validated immutable identity, runtime and validation policy."""

    schema_version: int
    name: str
    version: str
    api_version: int
    entrypoint: str
    python: tuple[StaticPythonRuntime, ...]
    exclude_data_paths: tuple[str, ...]
    channel_credentials: tuple[tuple[str, tuple[str, ...]], ...]
    identity_digest: str
    credential_paths: tuple[str, ...] = ()

    @property
    def all_credential_paths(self) -> tuple[str, ...]:
        """完整脱敏范围；渠道仍保留各自更窄的凭据授权。"""
        return tuple(sorted(set(self.credential_paths) | {
            path for _channel, paths in self.channel_credentials for path in paths
        }))

    @property
    def requirements(self) -> tuple[str, ...]:
        """Return all declared requirements paths in manifest order."""

        return tuple(runtime.requirements for runtime in self.python)


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


def materialize_command(
    plugin_root: Path,
    python: tuple[StaticPythonRuntime, ...],
    command: tuple[str, ...],
    cwd: str = ".",
    *,
    environment_root: Path | None = None,
) -> tuple[str, ...]:
    """把实际注册命令绑定到固定制品及其 Python 环境。"""

    runtime_root = command_python_runtime(plugin_root, command, cwd, python)
    if runtime_root is None:
        head = command[0]
        if _looks_like_artifact_path(head):
            executable = plugin_root.joinpath(
                *PurePosixPath(head).parts
            ).resolve(strict=True)
            if not executable.is_relative_to(plugin_root.resolve(strict=True)):
                raise RuntimeError("command executable 越过 artifact")
            return (str(executable), *command[1:])
        return command
    runtime = next(item for item in python if item.runtime_root == runtime_root)
    if environment_root is None:
        raise RuntimeError("Python command 缺少显式运行环境")
    interpreter = staged_python_interpreter(environment_root, runtime)
    return (str(interpreter), "-E", "-s", "-B", *command[1:])


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
    # 2. Requirements are complete before the artifact is published.
    python = _python_runtimes(root, raw.get("python", []))
    exclude_data_paths = _validation_paths(root, raw.get("validation", {}))

    # 3. Optional declarations are checked statically and kept immutable.
    channel_credentials = _channel_credentials(raw.get("channel_credentials", {}))
    credential_paths = _credential_paths(raw.get("credential_paths", []), "credential_paths")
    _check_credential_overlap(set(credential_paths) | {
        path for _channel, paths in channel_credentials for path in paths
    }, "credential_paths/channel_credentials")
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
        "channel_credentials": [
            {"channel": channel, "paths": list(paths)}
            for channel, paths in channel_credentials
        ],
    }
    # 原渠道 manifest 的身份保持不变；通用声明参与自身不可变身份。
    if credential_paths:
        identity["credential_paths"] = list(credential_paths)
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
        python=python,
        exclude_data_paths=exclude_data_paths,
        channel_credentials=channel_credentials,
        identity_digest=identity_digest,
        credential_paths=credential_paths,
    )


def _channel_credentials(
    raw: object,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate import-free channel credential paths from the artifact manifest."""

    # 1. Each channel owns one sorted set of dotted config paths.
    table = _table(raw, "channel_credentials")
    result: list[tuple[str, tuple[str, ...]]] = []
    for channel, paths in sorted(table.items()):
        name = _name(channel, f"channel_credentials.{channel}")
        result.append((name, _credential_paths(paths, f"channel_credentials.{name}")))

    # 3. Two channels may reuse one exact credential, but not overlapping paths.
    all_paths = {path for _channel, paths in result for path in paths}
    _check_credential_overlap(all_paths, "channel_credentials 跨 channel")
    return tuple(result)


def _credential_paths(raw: object, label: str) -> tuple[str, ...]:
    """在静态文件边界校验凭据路径，防止脱敏依赖处理顺序。"""
    values = _string_list(raw, label)
    for value in values:
        if any(_CONFIG_KEY.fullmatch(part) is None for part in value.split(".")):
            raise ValueError(f"{label} 包含无效 config path: {value}")
    _check_credential_overlap(set(values), label)
    return tuple(sorted(values))


def _check_credential_overlap(paths: set[str], label: str) -> None:
    for value in paths:
        parts = value.split(".")
        if any(".".join(parts[:index]) in paths for index in range(1, len(parts))):
            raise ValueError(f"{label} 路径重叠: {value}")


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


def command_python_runtime(
    root: Path,
    command: tuple[str, ...],
    cwd: str,
    runtimes: tuple[StaticPythonRuntime, ...],
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
            "command 必须唯一绑定已声明 Python runtime: "
            f"matches={[item.runtime_root for item in matches]}"
        )
    return matches[0].runtime_root


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


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
