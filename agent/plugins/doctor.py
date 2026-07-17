from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, cast

from agent.config import Config
from agent.plugins.base import Plugin
from agent.plugins.manifest import load_plugin_manifest, plugins_root
from agent.plugins.registry import plugin_registry
from agent.plugins.specs import McpServerSpec


def run_plugin_doctor(
    *,
    plugin_id: str = "",
    config_path: str = "config.toml",
    workspace: Path,
    plugins_home: Path | None = None,
) -> dict[str, Any]:
    resolved_workspace = workspace
    config = Config.load(config_path, workspace=resolved_workspace)
    memory_engine = (config.memory.engine or "").strip() or "default"
    manifest = load_plugin_manifest(plugins_home)
    selected = [plugin_id] if plugin_id else sorted(manifest)
    if plugin_id and plugin_id not in manifest:
        return {"status": "broken", "plugins": [], "error": f"插件不存在: {plugin_id}"}
    plugins = [
        _inspect_plugin(
            current_id,
            manifest[current_id],
            resolved_workspace,
            plugins_home,
            memory_engine=memory_engine,
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
    workspace: Path,
    plugins_home: Path | None,
    *,
    memory_engine: str,
) -> dict[str, Any]:
    plugin_root = _find_plugin_root(plugin_id, plugins_home)
    checks = [_check("policy", "ok" if enabled else "warn", f"enabled={str(enabled).lower()}")]
    if plugin_root is None:
        checks.append(_check("install", "error", "未找到插件目录"))
    else:
        checks.append(_check("install", "ok", f"已发现 plugin.py: {plugin_root}"))
        try:
            plugin_class = _load_plugin_class(plugin_root)
            checks.extend(
                _check_capabilities(
                    plugin_class,
                    plugin_root,
                    workspace,
                    links_required=enabled and not (
                        plugin_id == "default_memory" and memory_engine != "default"
                    ),
                )
            )
        except Exception as e:
            checks.append(_check("declaration", "error", str(e)))
    return {
        "plugin_id": plugin_id,
        "status": _merge_status(check["status"] for check in checks),
        "checks": checks,
    }


def _find_plugin_root(plugin_id: str, plugins_home: Path | None) -> Path | None:
    name, separator, marketplace = plugin_id.partition("@")
    if not separator:
        root = Path(__file__).resolve().parents[2] / "plugins" / name
        return root if (root / "plugin.py").exists() else None
    base = plugins_root(plugins_home) / "cache" / marketplace / name
    versions = sorted(path for path in base.iterdir() if path.is_dir()) if base.is_dir() else []
    return versions[-1] if versions and (versions[-1] / "plugin.py").exists() else None


def _load_plugin_class(plugin_root: Path) -> type[Plugin]:
    module_name = f"akasic_plugin_doctor_{uuid.uuid4().hex}"
    path = plugin_root / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=[str(plugin_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        plugin_class = plugin_registry.get_class(module_name)
        if plugin_class is None:
            raise ValueError("plugin.py 未声明 Plugin 子类")
        if not issubclass(plugin_class, Plugin):
            raise TypeError("plugin.py 注册的类型不是 Plugin 子类")
        return cast(type[Plugin], plugin_class)
    finally:
        plugin_registry.remove_plugin(module_name)
        _ = sys.modules.pop(module_name, None)


def _check_capabilities(
    plugin_class: type[Plugin],
    plugin_root: Path,
    workspace: Path,
    *,
    links_required: bool,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for label, roots, target in (
        ("skills", plugin_class.skill_roots(), workspace / "skills"),
        ("drift_skills", plugin_class.drift_skill_roots(), workspace / "drift" / "skills"),
    ):
        missing = [raw for raw in roots if not (plugin_root / raw).is_dir()]
        links = [child.name for raw in roots for child in (plugin_root / raw).iterdir() if child.is_dir()]
        unlinked = [
            name
            for name in links
            if links_required and not (target / name).is_symlink()
        ]
        status = "error" if missing else "warn" if unlinked else "ok"
        checks.append(_check(label, status, f"roots={len(roots)} missing={missing} unlinked={unlinked}"))
    servers = plugin_class.mcp_servers()
    invalid = [item for item in servers if not isinstance(item, McpServerSpec)]
    checks.append(_check("mcp", "error" if invalid else "ok", f"servers={len(servers)}"))
    return checks


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _merge_status(statuses: Any) -> str:
    values = list(statuses)
    if any(value in {"error", "broken"} for value in values):
        return "broken"
    if any(value in {"warn", "degraded"} for value in values):
        return "degraded"
    return "healthy"
