from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.manifest import load_plugin_manifest, plugins_root
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
)


def run_plugin_doctor(
    *,
    plugin_id: str = "",
    workspace: Path,
    plugins_home: Path | None = None,
) -> dict[str, Any]:
    """只读安装制品；实际入口、依赖和资源检查留给完整 Root 装配。"""
    resolved_workspace = workspace
    manifest = load_plugin_manifest(plugins_home)
    selected = [plugin_id] if plugin_id else sorted(manifest)
    if plugin_id and plugin_id not in manifest:
        return {"status": "broken", "plugins": [], "error": f"插件不存在: {plugin_id}"}
    plugins = [
        _inspect_plugin(
            current_id,
            manifest[current_id],
            plugins_home,
        )
        for current_id in selected
    ]
    return {
        "status": _merge_status(item["status"] for item in plugins),
        "plugins": plugins,
        "workspace": str(resolved_workspace),
    }


def format_plugin_doctor_report(report: dict[str, Any]) -> str:
    error = str(report.get("error") or "").strip()
    if error:
        return error
    lines: list[str] = []
    for plugin in cast(list[dict[str, Any]], report.get("plugins") or []):
        lines.append(f"plugin doctor {plugin['plugin_id']}")
        for check in cast(list[dict[str, str]], plugin["checks"]):
            lines.append(f"- {check['name']}: {check['status']} - {check['detail']}")
        lines.extend([f"- result: {plugin['status']}", ""])
    return "\n".join(lines).rstrip() if lines else "没有发现任何插件。"


def _inspect_plugin(
    plugin_id: str,
    enabled: bool,
    plugins_home: Path | None,
) -> dict[str, Any]:
    checks = [
        _check("policy", "ok" if enabled else "warn", f"enabled={str(enabled).lower()}")
    ]
    resolution_error: str | None = None
    try:
        installed_root = _find_plugin_root(
            plugin_id,
            plugins_home,
        )
    except (OSError, RuntimeError, ValueError) as error:
        installed_root = None
        resolution_error = str(error)
    if resolution_error is not None:
        checks.append(_check("install", "error", resolution_error))
    elif installed_root is not None:
        checks.append(
            _check(
                "install",
                "ok",
                f"installed plugin.py: {installed_root}",
            )
        )
        try:
            load_static_plugin_manifest(installed_root)
            checks.append(_check("runtime", "deferred", "运行能力由实际装配确定"))
        except (OSError, RuntimeError, ValueError) as e:
            checks.append(_check("declaration", "error", str(e)))
    else:
        checks.append(_check("install", "error", "未找到插件目录"))
    return {
        "plugin_id": plugin_id,
        "status": _merge_status(check["status"] for check in checks),
        "checks": checks,
    }


def _find_plugin_root(
    plugin_id: str,
    plugins_home: Path | None,
) -> Path | None:
    """Read one exact installed artifact without scanning checkout sources."""

    name, separator, marketplace = plugin_id.partition("@")
    if not separator:
        return None

    base = plugins_root(plugins_home) / "cache" / marketplace / name
    pointers = read_pointers(base)
    if pointers is not None:
        if pointers.stable != pointers.latest:
            raise RuntimeError(f"插件仍有历史候选指针对，须先处理未决更新: {base}")
        return resolve_pointer(base, pointers.stable)

    return None


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _merge_status(statuses: Any) -> str:
    values = list(statuses)
    if any(value in {"error", "broken"} for value in values):
        return "broken"
    if any(value in {"warn", "degraded", "deferred"} for value in values):
        return "degraded"
    return "healthy"
