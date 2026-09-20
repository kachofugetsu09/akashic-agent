"""从固定插件制品读取代码身份和 Python 安装输入。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PYTHON_COMMAND = re.compile(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?")


@dataclass(frozen=True, slots=True)
class StaticPythonRuntime:
    """One source-relative requirements file that must be staged before use."""

    requirements: str
    runtime_root: str


@dataclass(frozen=True, slots=True)
class StaticPluginManifest:
    """plugin.py 身份和制品内 requirements 安装输入。"""

    name: str
    version: str
    api_version: int
    python: tuple[StaticPythonRuntime, ...]
    identity_digest: str

    @property
    def requirements(self) -> tuple[str, ...]:
        """返回按路径排序的制品 requirements 文件。"""

        return tuple(runtime.requirements for runtime in self.python)


def load_static_plugin_manifest(plugin_root: Path) -> StaticPluginManifest:
    """不导入插件，只读取 plugin.py 身份和实际 requirements 文件。"""

    # 1. Resolve the artifact root without accepting a symlink as its owner.
    root = plugin_root.resolve(strict=True)
    if plugin_root.is_symlink() or not root.is_dir():
        raise ValueError(f"插件 artifact 根必须是普通目录: {plugin_root}")
    # 2. 身份与 requirements 都来自固定代码制品，不读取数据复制策略。
    name, version, api_version = load_plugin_identity(root)
    python = _python_runtimes(root)
    identity: dict[str, object] = {
        "name": name,
        "version": version,
        "api_version": api_version,
        "python": [
            {
                "requirements": item.requirements,
                "runtime_root": item.runtime_root,
            }
            for item in python
        ],
    }
    identity_digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return StaticPluginManifest(
        name=name,
        version=version,
        api_version=api_version,
        python=python,
        identity_digest=identity_digest,
    )


def load_plugin_identity(plugin_root: Path) -> tuple[str, str, int]:
    """只读取三个顶层字面量身份，不执行模块或解释其他声明。"""
    # 1. plugin.py 是唯一身份来源，链接和缺失入口不能参与安装。
    path = plugin_root / "plugin.py"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"插件 plugin.py 必须是普通文件: {path}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ValueError(f"插件身份源码无法解析: {path}") from error

    # 2. 只接受单次、直接赋值；表达式、导入和条件分支都不提供身份。
    fields = {"name", "version", "api_version"}
    values: dict[str, object] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name) and target.id in fields]
        if not names:
            continue
        if len(targets) != 1 or len(names) != 1 or not isinstance(value, ast.Constant):
            raise ValueError(f"插件身份必须直接赋字面量: {path}:{statement.lineno}")
        name = names[0]
        if name in values:
            raise ValueError(f"插件身份重复赋值: {name}")
        values[name] = value.value
    missing = sorted(fields - values.keys())
    if missing:
        raise ValueError(f"plugin.py 缺少顶层字面量身份: {missing}")
    api_version = _integer(values, "api_version")
    if api_version != 3:
        raise ValueError("plugin.py 只接受 api_version = 3")
    return _name(values["name"], "name"), _version(values["version"], "version"), api_version


def staged_python_interpreter(
    plugin_root: Path,
    runtime: StaticPythonRuntime,
) -> Path:
    """Return the executable staged for one manifest Python runtime."""

    root = plugin_root.resolve(strict=True)
    _reject_symlink_ancestors(root, root / runtime.runtime_root, "Python runtime")
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


def _python_runtimes(root: Path) -> tuple[StaticPythonRuntime, ...]:
    """从固定制品发现 requirements.txt；不读取运行数据或准备环境。"""
    excluded = {
        ".git", ".venv", "venv", "node_modules", "cache", ".cache",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    }
    result: list[StaticPythonRuntime] = []

    def visit(directory: Path) -> None:
        # 1. 不进入依赖和缓存；其余目录链接可能隐藏 runtime，直接拒绝。
        for path in sorted(directory.iterdir()):
            if path.name in excluded:
                continue
            if path.is_symlink():
                if path.name == "requirements.txt" or path.is_dir() or not path.exists():
                    raise ValueError(f"插件 Python runtime 不能经过符号链接: {path}")
                continue
            if path.is_dir():
                if path.name == "requirements.txt":
                    raise ValueError(f"插件 requirements.txt 必须是文件: {path}")
                visit(path)
            elif path.name == "requirements.txt":
                # 2. 精确文件名是 runtime 标记，其他 requirements 文件不独立安装。
                requirements = _relative_artifact_path(
                    root, path.relative_to(root).as_posix(),
                    label="requirements", must_exist=True, require_file=True,
                )
                result.append(StaticPythonRuntime(
                    requirements=requirements,
                    runtime_root=str(PurePosixPath(requirements).parent),
                ))

    visit(root)
    return tuple(sorted(result, key=lambda item: item.requirements))


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
    if not matches:
        raise ValueError(
            "command 必须绑定制品 Python runtime: "
            f"matches={[item.runtime_root for item in matches]}"
        )
    # 嵌套 runtime 拥有自己的命令，根 runtime 只承接其余路径。
    return max(
        matches, key=lambda item: len(PurePosixPath(item.runtime_root).parts)
    ).runtime_root


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


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
