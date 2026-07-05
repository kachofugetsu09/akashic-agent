from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.aka_descriptor import PluginDescriptor, load_plugin_descriptor
from agent.plugins.global_registry import upsert_plugin_registry_entry


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
        descriptor = load_plugin_descriptor(clone_root)
        if descriptor is None:
            raise ValueError("插件缺少 .aka-plugin/plugin.json")
        install_result = _activate_plugin_version(
            descriptor=descriptor,
            marketplace=marketplace,
            clone_root=clone_root,
            cache_root=cache_root,
            data_root=data_root,
        )
        plugin_id = f"{descriptor.name}@{marketplace}"
        _ = upsert_plugin_registry_entry(
            plugin_id,
            {
                "plugin_id": plugin_id,
                "name": descriptor.name,
                "marketplace": marketplace,
                "source_type": "installed",
                "version": descriptor.version,
                "description": descriptor.description,
                "enabled": True,
                "local_disabled": False,
                "active": False,
                "plugin_root": str(install_result.installed_path),
                "data_dir": str(install_result.data_path),
                "lifecycle_entry": str(descriptor.lifecycle_entry or ""),
                "capabilities": {
                    "lifecycle": bool(descriptor.lifecycle_entry),
                    "skills": bool(descriptor.skill_roots or descriptor.drift_skill_roots),
                    "mcp": bool(descriptor.mcp_servers),
                },
                "skills": _collect_skill_names(descriptor.skill_roots),
                "drift_skills": _collect_skill_names(descriptor.drift_skill_roots),
                "mcp_servers": sorted(descriptor.mcp_servers.keys()),
                "install_source": source,
            },
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
    descriptor: PluginDescriptor,
    marketplace: str,
    clone_root: Path,
    cache_root: Path,
    data_root: Path,
) -> PluginInstallResult:
    data_path = data_root / f"{descriptor.name}-{marketplace}"
    data_path.mkdir(parents=True, exist_ok=True)
    plugin_base = cache_root / descriptor.name
    target_root = plugin_base / descriptor.version
    plugin_base.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        shutil.rmtree(target_root)
    _ = shutil.copytree(clone_root, target_root)
    _prepare_plugin_mcp_runtimes(target_root, descriptor, data_path)
    _remove_old_versions(plugin_base, descriptor.version)
    return PluginInstallResult(
        plugin_name=descriptor.name,
        plugin_version=descriptor.version,
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
    descriptor: PluginDescriptor,
    data_path: Path,
) -> None:
    for config_relpath in _manifest_mcp_config_paths(descriptor):
        config_path = plugin_root / config_relpath
        if not config_path.exists():
            continue
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        loaded_dict = cast(dict[str, object], loaded)
        servers = loaded_dict.get("servers")
        if not isinstance(servers, dict):
            continue
        servers_dict = cast(dict[object, object], servers)
        changed = False
        for server_name, server_value in servers_dict.items():
            if not isinstance(server_value, dict):
                continue
            server_dict = cast(dict[str, object], server_value)
            _inject_plugin_env(server_dict, data_path)
            if _prepare_single_mcp_server(
                plugin_root=plugin_root,
                server_name=str(server_name),
                server=server_dict,
            ):
                changed = True
                continue
            changed = True
        if changed:
            _ = config_path.write_text(
                json.dumps(loaded, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def _manifest_mcp_config_paths(descriptor: PluginDescriptor) -> list[str]:
    raw_paths = descriptor.raw_manifest.get("paths")
    if not isinstance(raw_paths, dict):
        return []
    raw_paths_dict = cast(dict[str, object], raw_paths)
    configs = raw_paths_dict.get("mcp_servers")
    if isinstance(configs, str):
        stripped = configs.strip()
        return [stripped] if stripped else []
    if not isinstance(configs, list):
        return []
    result: list[str] = []
    for item in cast(list[object], configs):
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result


def _prepare_single_mcp_server(
    *,
    plugin_root: Path,
    server_name: str,
    server: dict[str, object],
) -> bool:
    command = server.get("command")
    if not isinstance(command, list) or not command:
        return False
    command_items = [str(item) for item in cast(list[object], command)]
    if not _is_python_command(command_items[0]):
        return False
    runtime_root = _resolve_mcp_runtime_root(plugin_root, server, command_items)
    if runtime_root is None:
        return False
    requirements = runtime_root / "requirements.txt"
    if not requirements.exists():
        return False
    venv_python = _ensure_python_runtime(runtime_root, requirements, server_name)
    if command_items[0] == str(venv_python):
        return False
    command_items[0] = str(venv_python)
    server["command"] = command_items
    return True


def _inject_plugin_env(server: dict[str, object], data_path: Path) -> None:
    env = server.get("env")
    if not isinstance(env, dict):
        env = {}
        server["env"] = env
    env_dict = cast(dict[str, object], env)
    _ = env_dict.setdefault("AKA_PLUGIN_DATA_DIR", str(data_path))


def _resolve_mcp_runtime_root(
    plugin_root: Path,
    server: dict[str, object],
    command_items: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if len(command_items) >= 2:
        script_path = Path(command_items[1])
        if not script_path.is_absolute():
            candidates.append((plugin_root / script_path).resolve(strict=False).parent)
    cwd_raw = str(server.get("cwd") or "").strip()
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


def _collect_skill_names(skill_roots: tuple[Path, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in skill_roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "SKILL.md").exists():
                continue
            if child.name in seen:
                continue
            seen.add(child.name)
            names.append(child.name)
    return names


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
