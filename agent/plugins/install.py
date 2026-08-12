from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.artifacts import (
    ArtifactPointer,
    ArtifactPointers,
    pointer_state_path,
    read_pointers,
    relative_artifact_pointer,
    write_pointers,
)
from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    load_package_manifest,
    load_plugin_manifest,
    remove_plugin_manifest_entry,
    set_plugin_enabled,
    upsert_plugin_manifest,
    plugins_root,
    workspace_plugin_data_dir,
)
from agent.plugins.registry import plugin_registry
from agent.plugins.specs import McpServerSpec


@dataclass(frozen=True)
class PluginInstallResult:
    plugin_name: str
    plugin_version: str
    marketplace: str
    installed_path: Path
    data_path: Path
    source_revision: str
    staged_candidate: bool


@dataclass
class _CacheActivation:
    result: PluginInstallResult
    plugin_base: Path
    previous_pointers: ArtifactPointers | None
    created_artifact: bool

    def rollback(self) -> None:
        """撤销已发布 cache，并恢复发布前的可运行版本。"""

        # 1. 恢复发布前完整 pointer state；缺失状态也精确恢复。
        _restore_pointers(self.plugin_base, self.previous_pointers)

        # 2. 只删除本事务新建、且尚未被任何旧 pointer 引用的 artifact。
        target_root = self.result.installed_path
        if self.created_artifact and (target_root.exists() or target_root.is_symlink()):
            _remove_path(target_root)

    def finalize(self) -> None:
        """Immutable artifacts require no destructive post-commit cleanup."""


def aka_plugins_root() -> Path:
    return plugins_root()


def installed_cache_root() -> Path:
    return aka_plugins_root() / "cache"


def set_installed_plugin_enabled(
    plugin_id: str,
    *,
    enabled: bool,
    plugins_home: Path | None = None,
) -> Path:
    home = plugins_home or aka_plugins_root()
    _ = _split_installed_plugin_id(plugin_id)
    return set_plugin_enabled(
        plugin_id,
        enabled=enabled,
        plugins_home=home,
    )


def uninstall_plugin(
    plugin_id: str,
    *,
    workspace: Path,
    plugins_home: Path | None = None,
    wait_until_disabled: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    home = plugins_home or aka_plugins_root()
    _ = set_plugin_enabled(plugin_id, enabled=False, plugins_home=home)
    if wait_until_disabled is not None:
        wait_until_disabled(plugin_id)
    return finalize_uninstall_plugin(
        plugin_id,
        workspace=workspace,
        plugins_home=home,
    )


def finalize_uninstall_plugin(
    plugin_id: str,
    *,
    workspace: Path,
    plugins_home: Path | None = None,
) -> tuple[Path, Path]:
    """删除已禁用插件的代码和清单，并保留 workspace plugin-data。"""

    home = plugins_home or aka_plugins_root()
    plugin_name, marketplace = _split_installed_plugin_id(plugin_id)
    cache_path = home / "cache" / marketplace / plugin_name
    data_path = workspace_plugin_data_dir(workspace, plugin_name, marketplace)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    _ = remove_plugin_manifest_entry(plugin_id, plugins_home=home)
    return cache_path, data_path


def _split_installed_plugin_id(plugin_id: str) -> tuple[str, str]:
    plugin_name, separator, marketplace = plugin_id.rpartition("@")
    if (
        not separator
        or not plugin_name
        or not marketplace
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", plugin_name) is None
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", marketplace) is None
    ):
        raise ValueError(f"无效的已安装插件 ID: {plugin_id}")
    return plugin_name, marketplace


def install_git_plugin(
    *,
    workspace: Path,
    source: str,
    marketplace: str,
    ref_name: str = "",
    sparse_paths: list[str] | None = None,
    plugins_home: Path | None = None,
    stage_candidate: bool = False,
) -> PluginInstallResult:
    home = (plugins_home or aka_plugins_root()).resolve(strict=False)
    _ = _validate_path_segment(marketplace, "marketplace")
    if not isinstance(source, str) or not source or source != source.strip():
        raise ValueError("插件 source 必须是非空且不含首尾空白的字符串")
    if not isinstance(ref_name, str):
        raise ValueError("插件 ref 必须是字符串")
    if ref_name != ref_name.strip():
        raise ValueError("插件 ref 不能包含首尾空白")
    if ref_name.startswith("-"):
        raise ValueError("插件 ref 不能以命令选项开头")
    sparse = sparse_paths or []
    if not all(
        isinstance(path, str) and path and path == path.strip() for path in sparse
    ):
        raise ValueError("插件 sparse path 必须是非空字符串")
    marketplace_root = home / "marketplaces" / marketplace
    cache_root = home / "cache" / marketplace
    _ensure_directory_tree(home, marketplace_root)
    _ensure_directory_tree(home, cache_root)

    # 1. 在任何 cache 改动前校验 manifest，避免坏配置把安装事务推到半路
    _ = load_plugin_manifest(home)
    _ = load_package_manifest(home)

    with tempfile.TemporaryDirectory(
        dir=marketplace_root, prefix="clone-"
    ) as clone_dir:
        clone_root = Path(clone_dir)
        _clone_git_source(
            source=source,
            destination=clone_root,
            ref_name=ref_name,
            sparse_paths=sparse,
        )
        _validate_source_tree(clone_root)
        source_revision = _run_git(["rev-parse", "HEAD"], cwd=clone_root)
        if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
            raise RuntimeError(f"插件 Git HEAD 无效: {source_revision}")
        plugin_class = _load_plugin_class(clone_root)
        plugin_name = _validate_path_segment(
            getattr(plugin_class, "name", None),
            "插件 name",
        )
        plugin_version = _validate_path_segment(
            getattr(plugin_class, "version", None),
            "插件 version",
        )
        mcp_servers = _load_mcp_specs(plugin_class)
        activation = _activate_plugin_version(
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            mcp_servers=mcp_servers,
            marketplace=marketplace,
            clone_root=clone_root,
            cache_root=cache_root,
            data_root=workspace.resolve(strict=False) / "plugin-data",
            workspace=workspace,
            source_revision=source_revision,
            stage_candidate=stage_candidate,
        )
        plugin_id = f"{plugin_name}@{marketplace}"
        try:
            # 2. manifest 原子写入成功后，cache 才算完成安装
            _ = upsert_plugin_manifest(
                plugin_id,
                enabled=True,
                plugins_home=home,
            )
        except BaseException:
            activation.rollback()
            raise
        activation.finalize()
    return activation.result


def _clone_git_source(
    *,
    source: str,
    destination: Path,
    ref_name: str,
    sparse_paths: list[str],
) -> None:
    if sparse_paths:
        _ = _run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--",
                source,
                str(destination),
            ]
        )
        _ = _run_git(
            ["sparse-checkout", "set", "--", *sparse_paths],
            cwd=destination,
        )
        checkout_ref = _resolve_git_ref(ref_name or "HEAD", destination)
        _ = _run_git(
            ["checkout", "--detach", checkout_ref],
            cwd=destination,
        )
        return
    _ = _run_git(["clone", "--", source, str(destination)])
    if ref_name:
        checkout_ref = _resolve_git_ref(ref_name, destination)
        _ = _run_git(["checkout", "--detach", checkout_ref], cwd=destination)


def _resolve_git_ref(ref_name: str, repository: Path) -> str:
    """将用户 ref 解析为 commit SHA，避免 checkout 参数歧义。"""

    candidates = [ref_name]
    if ref_name.startswith("refs/heads/"):
        candidates.append(f"refs/remotes/origin/{ref_name.removeprefix('refs/heads/')}")
    elif not ref_name.startswith("refs/remotes/origin/"):
        candidates.append(f"refs/remotes/origin/{ref_name}")
    last_error: RuntimeError | None = None
    for candidate in candidates:
        try:
            resolved = _run_git(
                [
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    f"{candidate}^{{commit}}",
                ],
                cwd=repository,
            )
        except RuntimeError as error:
            last_error = error
            continue
        if resolved:
            return resolved
    raise RuntimeError(f"git ref 无法解析: {ref_name}") from last_error


def _activate_plugin_version(
    *,
    plugin_name: str,
    plugin_version: str,
    mcp_servers: list[McpServerSpec],
    marketplace: str,
    clone_root: Path,
    cache_root: Path,
    data_root: Path,
    workspace: Path,
    source_revision: str,
    stage_candidate: bool,
) -> _CacheActivation:
    """Prepare one immutable artifact and publish it as latest."""

    # 1. 创建受保护的数据目录和 cache 父目录
    data_path = data_root / f"{plugin_name}-{marketplace}"
    ensure_workspace_plugin_data_dir(data_path, workspace)
    plugin_base = cache_root / plugin_name
    _ensure_directory(plugin_base)
    visible_versions = _cache_version_dirs(plugin_base)
    if len(visible_versions) > 1:
        paths = ", ".join(str(path) for path in visible_versions)
        raise ValueError(f"插件 cache 可见版本冲突: {paths}")
    previous_pointers = read_pointers(plugin_base)
    if (
        previous_pointers is not None
        and previous_pointers.stable != previous_pointers.latest
    ):
        raise RuntimeError(
            f"插件已有 latest 等待 promote/discard: {plugin_name}@{marketplace}"
        )
    stable = previous_pointers.stable if previous_pointers is not None else None
    if stable is None:
        stable = ArtifactPointer(visible_versions[0].name if visible_versions else None)
    stage_latest = stage_candidate

    artifacts_root = plugin_base / ".artifacts"
    _ensure_directory(artifacts_root)
    artifact_id = f"{plugin_version}-{source_revision[:16]}"
    target_root = artifacts_root / artifact_id
    if target_root.is_symlink():
        raise ValueError(f"插件 artifact 目标不能是符号链接: {target_root}")
    if target_root.exists() and not target_root.is_dir():
        raise ValueError(f"插件 artifact 目标不是目录: {target_root}")

    staging_root = Path(
        tempfile.mkdtemp(dir=cache_root, prefix=f".{plugin_name}-install-")
    )
    created_artifact = False
    try:
        # 2. 在不可发现的 staging 目录复制代码并准备依赖，旧版本保持可见
        _ = shutil.copytree(clone_root, staging_root, dirs_exist_ok=True)
        _prepare_plugin_mcp_runtimes(staging_root, mcp_servers)

        # 3. Artifact 只创建一次；一次原子写发布完整 stable/latest pair。
        if target_root.exists():
            _validate_source_tree(target_root)
            existing_revision = _run_git(["rev-parse", "HEAD"], cwd=target_root)
            if existing_revision != source_revision:
                raise RuntimeError(f"插件 artifact 身份冲突: {target_root}")
            _remove_path(staging_root)
        else:
            os.replace(staging_root, target_root)
            created_artifact = True
        latest = relative_artifact_pointer(plugin_base, target_root)
        candidate_staged = stage_latest and stable != latest
        _ = write_pointers(
            plugin_base,
            stable=stable if stage_latest else latest,
            latest=latest,
        )
    except BaseException:
        _restore_pointers(plugin_base, previous_pointers)
        if created_artifact and (target_root.exists() or target_root.is_symlink()):
            _remove_path(target_root)
        if staging_root.exists() or staging_root.is_symlink():
            _remove_path(staging_root)
        raise

    result = PluginInstallResult(
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        marketplace=marketplace,
        installed_path=target_root,
        data_path=data_path,
        source_revision=source_revision,
        staged_candidate=candidate_staged,
    )
    return _CacheActivation(
        result=result,
        plugin_base=plugin_base,
        previous_pointers=previous_pointers,
        created_artifact=created_artifact,
    )


def _restore_pointers(
    plugin_base: Path,
    pointers: ArtifactPointers | None,
) -> None:
    path = pointer_state_path(plugin_base)
    if pointers is None:
        if path.exists() or path.is_symlink():
            path.unlink()
        return
    _ = write_pointers(
        plugin_base,
        stable=pointers.stable,
        latest=pointers.latest,
    )


def _cache_version_dirs(plugin_base: Path) -> list[Path]:
    """列出可被 watcher 发现的旧版本目录。"""

    result: list[Path] = []
    for child in sorted(plugin_base.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_symlink():
            raise ValueError(f"插件 cache 版本不能是符号链接: {child}")
        if not child.is_dir():
            raise ValueError(f"插件 cache 版本不是目录: {child}")
        result.append(child)
    return result


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"插件路径不能是符号链接: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"插件路径不是目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _ensure_directory_tree(root: Path, path: Path) -> None:
    """在指定 root 内创建目录，并拒绝中间符号链接。"""

    root = root.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"插件路径越界: {path}") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"插件路径不能穿过符号链接: {current}")
    _ensure_directory(path)


def _validate_path_segment(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise ValueError(f"{label} 必须是安全的单一路径段")
    return value


def _validate_source_tree(root: Path) -> None:
    """校验 Git source 符号链接只指向 clone root 内的真实对象。"""

    root = root.resolve(strict=True)
    # 1. lstat 所有文件和目录项，但不让 os.walk 跟随 source 链接
    for current, directories, filenames in os.walk(root, followlinks=False):
        for name in [*directories, *filenames]:
            path = Path(current) / name
            if not path.is_symlink():
                continue
            try:
                resolved = path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as error:
                raise ValueError(f"插件 source 符号链接无效: {path}") from error
            if not resolved.is_relative_to(root):
                raise ValueError(f"插件 source 符号链接越界: {path} -> {resolved}")
            if resolved == root or path.parent.resolve(strict=True).is_relative_to(
                resolved
            ):
                raise ValueError(f"插件 source 符号链接形成循环: {path} -> {resolved}")


def _prepare_plugin_mcp_runtimes(
    plugin_root: Path,
    servers: list[McpServerSpec],
) -> None:
    for server in servers:
        _prepare_single_mcp_server(plugin_root=plugin_root, server=server)


def _prepare_single_mcp_server(
    *,
    plugin_root: Path,
    server: McpServerSpec,
) -> None:
    command_items = list(server.command)
    if not _is_python_command(command_items[0]):
        return
    runtime_root = _resolve_mcp_runtime_root(plugin_root, server.cwd, command_items)
    if runtime_root is None:
        return
    requirements = runtime_root / "requirements.txt"
    if not requirements.exists() or requirements.is_symlink():
        return
    _ = _ensure_python_runtime(runtime_root, requirements, server.name)


def _resolve_mcp_runtime_root(
    plugin_root: Path,
    cwd_raw: str,
    command_items: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if len(command_items) >= 2:
        script_path = Path(command_items[1])
        if _looks_like_plugin_path(command_items[1]):
            script_candidate = (
                script_path if script_path.is_absolute() else plugin_root / script_path
            )
            resolved_script = script_candidate.resolve(strict=False)
            _require_plugin_path(plugin_root, resolved_script, "MCP command")
            candidates.append(script_candidate.parent)
    if cwd_raw:
        cwd_path = Path(cwd_raw)
        cwd_candidate = cwd_path if cwd_path.is_absolute() else plugin_root / cwd_path
        _require_plugin_path(
            plugin_root,
            cwd_candidate.resolve(strict=False),
            "MCP cwd",
        )
        candidates.append(cwd_candidate)
    candidates.append(plugin_root)
    for candidate in candidates:
        if (candidate / "requirements.txt").exists():
            return candidate
    return None


def _looks_like_plugin_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or "/" in value
        or "\\" in value
        or value.startswith(".")
    )


def _require_plugin_path(plugin_root: Path, path: Path, label: str) -> None:
    plugin_root = plugin_root.resolve(strict=False)
    path = path.resolve(strict=False)
    try:
        _ = path.relative_to(plugin_root)
    except ValueError as error:
        raise ValueError(f"插件 {label} 越界: {path}") from error


def _load_plugin_class(plugin_root: Path) -> type:
    plugin_path = plugin_root / "plugin.py"
    if not plugin_path.exists():
        raise ValueError("插件缺少 plugin.py")
    module_name = f"akasic_plugin_install_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_path,
        submodule_search_locations=[str(plugin_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载插件文件: {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        plugin_class = plugin_registry.get_class(module_name)
        if plugin_class is None:
            raise ValueError("plugin.py 未声明 Plugin 子类")
        return plugin_class
    finally:
        plugin_registry.remove_module_tree(module_name)
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(
                f"{module_name}."
            ):
                _ = sys.modules.pop(imported_name, None)


def _load_mcp_specs(plugin_class: type) -> list[McpServerSpec]:
    provider = getattr(plugin_class, "mcp_servers", None)
    if not callable(provider):
        raise ValueError("插件缺少 mcp_servers() 声明")
    raw = cast(Callable[[], object], provider)()
    if not isinstance(raw, list):
        raise ValueError("mcp_servers() 必须返回 list")
    raw_items = cast(list[object], raw)
    result: list[McpServerSpec] = []
    names: set[str] = set()
    for item in raw_items:
        if (
            not isinstance(item, McpServerSpec)
            or not isinstance(item.name, str)
            or not item.name
            or not item.command
            or not isinstance(item.command, tuple)
            or not isinstance(item.cwd, str)
            or not isinstance(item.env, dict)
            or not isinstance(item.candidate_read_only_tools, tuple)
            or not all(isinstance(value, str) and value for value in item.command)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in item.env.items()
            )
            or not all(
                isinstance(value, str) and value
                for value in item.candidate_read_only_tools
            )
            or len(set(item.candidate_read_only_tools))
            != len(item.candidate_read_only_tools)
        ):
            raise ValueError(f"MCP server 声明无效: {item!r}")
        if item.name in names:
            raise ValueError(f"MCP server 名称重复: {item.name}")
        names.add(item.name)
        result.append(item)
    return result


def _ensure_python_runtime(
    runtime_root: Path,
    requirements: Path,
    server_name: str,
) -> Path:
    venv_dir = runtime_root / ".venv"
    venv_python = _venv_python_path(venv_dir)
    if not venv_python.exists():
        _run_command(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=runtime_root,
            label=f"{server_name} venv",
        )
    _run_command(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
        cwd=runtime_root,
        label=f"{server_name} pip install",
    )
    return venv_python


def _venv_python_path(venv_dir: Path) -> Path:
    return (
        venv_dir / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "python"
    )


def _is_python_command(value: str) -> bool:
    name = Path(value).name.lower()
    return name in {"python", "python3", "python.exe"}


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError(
        "git 命令失败: "
        + " ".join(args)
        + f"\nstdout:\n{result.stdout.strip()}\nstderr:\n{result.stderr.strip()}"
    )


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    label: str,
) -> None:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"{label} 失败: {' '.join(args)}"
        + f"\nstdout:\n{result.stdout.strip()}\nstderr:\n{result.stderr.strip()}"
    )
