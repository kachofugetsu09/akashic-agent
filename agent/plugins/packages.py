from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class PluginPackage:
    id: str
    root: Path
    members: tuple[str, ...]
    dashboard: bool
    provides: tuple[str, ...]


def discover_plugin_packages(project_root: Path) -> dict[str, PluginPackage]:
    packages_root = project_root / "plugin_packages"
    if not packages_root.is_dir():
        return {}
    result: dict[str, PluginPackage] = {}
    for path in sorted(packages_root.glob("*/package.toml")):
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        package = raw.get("package")
        if not isinstance(package, dict):
            raise ValueError(f"插件包缺少 [package]: {path}")
        package = cast(dict[str, Any], package)
        package_id = package.get("id")
        members = package.get("members")
        if not isinstance(package_id, str) or not package_id:
            raise ValueError(f"插件包 id 无效: {path}")
        if not isinstance(members, list):
            raise ValueError(f"插件包 members 无效: {path}")
        member_values = cast(list[object], members)
        if not all(isinstance(item, str) and item for item in member_values):
            raise ValueError(f"插件包 members 无效: {path}")
        members = cast(list[str], member_values)
        dashboard = package.get("dashboard", False)
        if not isinstance(dashboard, bool):
            raise ValueError(f"插件包 dashboard 无效: {path}")
        provides = package.get("provides", [])
        if not isinstance(provides, list):
            raise ValueError(f"插件包 provides 无效: {path}")
        provide_values = cast(list[object], provides)
        if not all(isinstance(item, str) and item for item in provide_values):
            raise ValueError(f"插件包 provides 无效: {path}")
        provides = cast(list[str], provide_values)
        if package_id in result:
            raise ValueError(f"插件包 id 重复: {package_id}")
        result[package_id] = PluginPackage(
            id=package_id,
            root=path.parent,
            members=tuple(members),
            dashboard=dashboard,
            provides=tuple(provides),
        )
    _validate_packages(result)
    return result


def enabled_plugin_packages(
    project_root: Path,
    entries: dict[str, bool],
) -> dict[str, PluginPackage]:
    packages = discover_plugin_packages(project_root)
    enabled = {
        package_id: package
        for package_id, package in packages.items()
        if entries.get(package_id, False)
    }
    claimed: dict[str, str] = {}
    for package in enabled.values():
        for capability in package.provides:
            owner = claimed.get(capability)
            if owner is not None:
                raise ValueError(
                    f"插件包 capability 冲突: {capability}={owner},{package.id}"
                )
            claimed[capability] = package.id
    return enabled


def _validate_packages(packages: dict[str, PluginPackage]) -> None:
    owners: dict[str, str] = {}
    for package in packages.values():
        for member in package.members:
            owner = owners.get(member)
            if owner is not None:
                raise ValueError(f"插件模块属于多个包: {member}={owner},{package.id}")
            owners[member] = package.id
