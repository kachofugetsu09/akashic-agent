from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from agent.config import Config
from agent.plugins.aka_descriptor import PluginDescriptor, load_plugin_descriptor
from agent.plugins.global_registry import load_plugin_registry


def run_plugin_doctor(
    *,
    plugin_id: str = "",
    config_path: str = "config.toml",
    workspace: Path | None = None,
    plugins_home: Path | None = None,
) -> dict[str, Any]:
    resolved_workspace = workspace or (Path.home() / ".akashic" / "workspace")
    engine_name = _load_memory_engine_name(config_path)
    registry = load_plugin_registry(plugins_home)
    if plugin_id:
        entry = registry.get(plugin_id)
        if entry is None:
            return {
                "status": "broken",
                "plugins": [],
                "error": f"插件不存在: {plugin_id}",
            }
        plugins = [_inspect_plugin(plugin_id, entry, resolved_workspace, engine_name)]
    else:
        plugins = [
            _inspect_plugin(current_id, entry, resolved_workspace, engine_name)
            for current_id, entry in sorted(registry.items())
        ]
    return {
        "status": _merge_status([item["status"] for item in plugins]) if plugins else "healthy",
        "plugins": plugins,
        "workspace": str(resolved_workspace),
        "memory_engine": engine_name,
    }


def format_plugin_doctor_report(report: dict[str, Any]) -> str:
    error = str(report.get("error") or "").strip()
    if error:
        return error
    plugins = cast(list[dict[str, Any]], report.get("plugins") or [])
    lines: list[str] = []
    for plugin in plugins:
        lines.append(f"plugin doctor {plugin['plugin_id']}")
        for check in cast(list[dict[str, str]], plugin.get("checks") or []):
            lines.append(
                f"- {check['name']}: {check['status']} - {check['detail']}"
            )
        lines.append(f"- result: {plugin['status']}")
        lines.append("")
    if not lines:
        return "没有发现任何插件。"
    return "\n".join(lines).rstrip()


def _inspect_plugin(
    plugin_id: str,
    entry: dict[str, object],
    workspace: Path,
    memory_engine_name: str,
) -> dict[str, Any]:
    plugin_root_text = str(entry.get("plugin_root") or "").strip()
    plugin_root = Path(plugin_root_text) if plugin_root_text else None
    descriptor = (
        load_plugin_descriptor(plugin_root)
        if plugin_root is not None and plugin_root.exists()
        else None
    )
    checks = [
        _check_install(plugin_root, descriptor),
        _check_policy(entry),
        _check_lifecycle(entry),
        _check_skills(plugin_id, entry, workspace),
        _check_mcp(entry, descriptor),
    ]
    status = _merge_status(check["status"] for check in checks)
    return {
        "plugin_id": plugin_id,
        "status": status,
        "memory_engine": memory_engine_name,
        "checks": checks,
    }


def _check_install(
    plugin_root: Path | None,
    descriptor: PluginDescriptor | None,
) -> dict[str, str]:
    if plugin_root is None:
        return _check("install", "error", "缺少 plugin_root")
    if not plugin_root.exists():
        return _check("install", "error", f"plugin_root 不存在: {plugin_root}")
    if descriptor is not None:
        return _check("install", "ok", f"已发现 .aka-plugin/plugin.json: {plugin_root}")
    if (plugin_root / "plugin.py").exists():
        return _check("install", "ok", f"builtin 插件目录存在: {plugin_root}")
    return _check("install", "error", f"目录存在但缺少插件声明: {plugin_root}")


def _check_policy(entry: dict[str, object]) -> dict[str, str]:
    enabled = bool(entry.get("enabled", True))
    local_disabled = bool(entry.get("local_disabled", False))
    active = bool(entry.get("active", False))
    status = "ok"
    if not enabled or local_disabled or not active:
        status = "warn"
    detail = (
        f"enabled={str(enabled).lower()} "
        f"local_disabled={str(local_disabled).lower()} "
        f"active={str(active).lower()}"
    )
    return _check("policy", status, detail)


def _check_lifecycle(entry: dict[str, object]) -> dict[str, str]:
    capabilities = _as_dict(entry.get("capabilities"))
    if not bool(capabilities.get("lifecycle", False)):
        return _check("lifecycle", "n/a", "未声明 lifecycle")
    lifecycle_entry = str(entry.get("lifecycle_entry") or "").strip()
    if not lifecycle_entry:
        return _check("lifecycle", "error", "capabilities.lifecycle=true 但缺少 lifecycle_entry")
    if not Path(lifecycle_entry).exists():
        return _check("lifecycle", "error", f"lifecycle_entry 不存在: {lifecycle_entry}")
    if bool(entry.get("active", False)):
        return _check("lifecycle", "ok", "声明存在且当前 active=true")
    return _check("lifecycle", "warn", "声明存在但当前 active=false，可能需要重启或被运行态 gating")


def _check_skills(
    plugin_id: str,
    entry: dict[str, object],
    workspace: Path,
) -> dict[str, str]:
    capabilities = _as_dict(entry.get("capabilities"))
    if not bool(capabilities.get("skills", False)):
        return _check("skills", "n/a", "未声明 skills")
    normal_skills = _as_str_list(entry.get("skills"))
    drift_skills = _as_str_list(entry.get("drift_skills"))
    normal_missing = [
        name
        for name in normal_skills
        if not (workspace / "skills" / name).is_symlink()
    ]
    base_name = plugin_id.split("@", 1)[0]
    drift_missing = [
        name
        for name in drift_skills
        if not (workspace / "drift" / "skills" / f"{base_name}:{name}").is_symlink()
    ]
    expected = len(normal_skills) + len(drift_skills)
    if expected == 0:
        return _check("skills", "warn", "capabilities.skills=true 但 registry 里没有 skill")
    missing = normal_missing + [f"{base_name}:{name}" for name in drift_missing]
    if not missing:
        return _check("skills", "ok", f"共 {expected} 个 skill 软链接已就位")
    return _check("skills", "warn", f"缺少软链接: {', '.join(missing)}")


def _check_mcp(
    entry: dict[str, object],
    descriptor: PluginDescriptor | None,
) -> dict[str, str]:
    capabilities = _as_dict(entry.get("capabilities"))
    if not bool(capabilities.get("mcp", False)):
        return _check("mcp", "n/a", "未声明 MCP")
    server_names = _as_str_list(entry.get("mcp_servers"))
    if not server_names:
        return _check("mcp", "error", "capabilities.mcp=true 但 registry 里没有 mcp_servers")
    if descriptor is None:
        return _check("mcp", "warn", f"registry 声明了 {len(server_names)} 个 MCP server，未校验 manifest")
    declared_servers = set(descriptor.mcp_servers.keys())
    missing = [name for name in server_names if name not in declared_servers]
    if missing:
        return _check("mcp", "error", f"manifest 未声明 server: {', '.join(missing)}")
    return _check("mcp", "ok", f"manifest 已声明 {len(server_names)} 个 MCP server")


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
    }


def _merge_status(statuses: Any) -> str:
    values = list(statuses)
    if any(value in {"error", "broken"} for value in values):
        return "broken"
    if any(value in {"warn", "degraded"} for value in values):
        return "degraded"
    return "healthy"


def _load_memory_engine_name(config_path: str) -> str:
    try:
        config = Config.load(config_path)
    except Exception:
        return ""
    return (config.memory.engine or "").strip() or "default"


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result
