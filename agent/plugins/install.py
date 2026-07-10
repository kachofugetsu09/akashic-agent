from __future__ import annotations

import os
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from agent.plugins.manifest import upsert_plugin_manifest
from agent.plugins.registry import plugin_registry
from agent.plugins.specs import McpServerSpec


@dataclass(frozen=True)
class PluginInstallResult:
    plugin_name: str
    plugin_version: str
    marketplace: str
    installed_path: Path
    data_path: Path


def aka_plugins_root() -> Path:
    return Path.home() / ".akashic-plugin"


def installed_cache_root() -> Path:
    return aka_plugins_root() / "cache"


def plugin_data_root(
    plugin_name: str,
    marketplace: str,
) -> Path:
    return aka_plugins_root() / "data" / f"{plugin_name}-{marketplace}"


def install_git_plugin(
    *,
    source: str,
    marketplace: str,
    ref_name: str = "",
    sparse_paths: list[str] | None = None,
    plugins_home: Path | None = None,
) -> PluginInstallResult:
    home = plugins_home or aka_plugins_root()
    marketplace_root = home / "marketplaces" / marketplace
    cache_root = home / "cache" / marketplace
    data_root = home / "data"
    marketplace_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=marketplace_root, prefix="clone-") as clone_dir:
        clone_root = Path(clone_dir)
        _clone_git_source(
            source=source,
            destination=clone_root,
            ref_name=ref_name,
            sparse_paths=sparse_paths or [],
        )
        plugin_class = _load_plugin_class(clone_root)
        plugin_name = str(getattr(plugin_class, "name", "") or "").strip()
        plugin_version = str(getattr(plugin_class, "version", "") or "").strip()
        if not plugin_name or not plugin_version:
            raise ValueError("插件必须在 plugin.py 声明 name 和 version")
        mcp_servers = _load_mcp_specs(plugin_class)
        install_result = _activate_plugin_version(
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            mcp_servers=mcp_servers,
            marketplace=marketplace,
            clone_root=clone_root,
            cache_root=cache_root,
            data_root=data_root,
        )
        plugin_id = f"{plugin_name}@{marketplace}"
        _ = upsert_plugin_manifest(
            plugin_id,
            enabled=True,
            plugins_home=home,
        )
    return install_result


def _clone_git_source(
    *,
    source: str,
    destination: Path,
    ref_name: str,
    sparse_paths: list[str],
) -> None:
    if sparse_paths:
        _run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source,
                str(destination),
            ]
        )
        _run_git(
            ["sparse-checkout", "set", *sparse_paths],
            cwd=destination,
        )
        _run_git(
            ["checkout", ref_name or "HEAD"],
            cwd=destination,
        )
        return
    _run_git(["clone", source, str(destination)])
    if ref_name:
        _run_git(["checkout", ref_name], cwd=destination)


def _activate_plugin_version(
    *,
    plugin_name: str,
    plugin_version: str,
    mcp_servers: list[McpServerSpec],
    marketplace: str,
    clone_root: Path,
    cache_root: Path,
    data_root: Path,
) -> PluginInstallResult:
    data_path = data_root / f"{plugin_name}-{marketplace}"
    data_path.mkdir(parents=True, exist_ok=True)
    plugin_base = cache_root / plugin_name
    target_root = plugin_base / plugin_version
    plugin_base.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        shutil.rmtree(target_root)
    _ = shutil.copytree(clone_root, target_root)
    _prepare_plugin_mcp_runtimes(target_root, mcp_servers)
    _remove_old_versions(plugin_base, plugin_version)
    return PluginInstallResult(
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        marketplace=marketplace,
        installed_path=target_root,
        data_path=data_path,
    )


def _remove_old_versions(
    plugin_base: Path,
    active_version: str,
) -> None:
    for child in plugin_base.iterdir():
        if not child.is_dir() or child.name == active_version:
            continue
        shutil.rmtree(child)


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
    if not requirements.exists():
        return
    _ensure_python_runtime(runtime_root, requirements, server.name)


def _resolve_mcp_runtime_root(
    plugin_root: Path,
    cwd_raw: str,
    command_items: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if len(command_items) >= 2:
        script_path = Path(command_items[1])
        if not script_path.is_absolute():
            candidates.append((plugin_root / script_path).resolve(strict=False).parent)
    if cwd_raw:
        cwd_path = Path(cwd_raw)
        resolved_cwd = (
            cwd_path
            if cwd_path.is_absolute()
            else (plugin_root / cwd_path).resolve(strict=False)
        )
        candidates.append(resolved_cwd)
    candidates.append(plugin_root)
    for candidate in candidates:
        if (candidate / "requirements.txt").exists():
            return candidate
    return None


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
        plugin_class = plugin_registry._classes.get(module_name)
        if plugin_class is None:
            raise ValueError("plugin.py 未声明 Plugin 子类")
        return plugin_class
    finally:
        plugin_registry.remove_plugin(module_name)
        sys.modules.pop(module_name, None)


def _load_mcp_specs(plugin_class: type) -> list[McpServerSpec]:
    raw = plugin_class.mcp_servers()
    if not isinstance(raw, list):
        raise ValueError("mcp_servers() 必须返回 list")
    result: list[McpServerSpec] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, McpServerSpec) or not item.name or not item.command:
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
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _is_python_command(value: str) -> bool:
    name = Path(value).name.lower()
    return name in {"python", "python3", "python.exe"}


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode == 0:
        return
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
