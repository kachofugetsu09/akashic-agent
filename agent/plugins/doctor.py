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
        stable_root, latest_root = _find_plugin_roots(
            plugin_id,
            plugins_home,
        )
    except (RuntimeError, ValueError) as error:
        stable_root, latest_root = None, None
        resolution_error = str(error)
    if resolution_error is not None:
        checks.append(_check("install", "error", resolution_error))
    elif stable_root is not None:
        checks.append(
            _check(
                "install",
                "ok",
                f"stable plugin.py: {stable_root}",
            )
        )
        try:
            load_static_plugin_manifest(stable_root)
            checks.append(_check("runtime", "deferred", "运行能力由实际装配确定"))
        except (OSError, RuntimeError, ValueError) as e:
            checks.append(_check("declaration", "error", str(e)))
    elif latest_root is None:
        checks.append(_check("install", "error", "未找到插件目录"))
    else:
        checks.append(
            _check(
                "install",
                "ok",
                f"latest candidate plugin.py: {latest_root}",
            )
        )
        try:
            load_static_plugin_manifest(latest_root)
            checks.append(_check("candidate_runtime", "deferred", "运行能力由实际装配确定"))
        except (OSError, RuntimeError, ValueError) as e:
            checks.append(_check("declaration", "error", str(e)))
    if (
        resolution_error is None
        and latest_root is not None
        and latest_root != stable_root
    ):
        checks.append(
            _check(
                "candidate",
                "deferred",
                "latest 候选尚未 promote；运行时继续以 stable 为准"
                f" (stable={stable_root}, latest={latest_root})",
            )
        )
    return {
        "plugin_id": plugin_id,
        "status": _merge_status(check["status"] for check in checks),
        "checks": checks,
    }


def _find_plugin_roots(
    plugin_id: str,
    plugins_home: Path | None,
) -> tuple[Path | None, Path | None]:
    """只读取正式安装的 stable/latest，不从 checkout 补齐缺少的插件。"""

    name, separator, marketplace = plugin_id.partition("@")
    if not separator:
        return None, None

    base = plugins_root(plugins_home) / "cache" / marketplace / name
    pointers = read_pointers(base)
    if pointers is not None:
        return (
            resolve_pointer(base, pointers.stable),
            resolve_pointer(base, pointers.latest),
        )

    # 3. 外部插件只认原子 pointer，不扫描旧版可见目录。
    return None, None


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _merge_status(statuses: Any) -> str:
    values = list(statuses)
    if any(value in {"error", "broken"} for value in values):
        return "broken"
    if any(value in {"warn", "degraded", "deferred"} for value in values):
        return "degraded"
    return "healthy"
