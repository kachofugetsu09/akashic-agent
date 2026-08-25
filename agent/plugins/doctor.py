from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any, cast

from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manifest import load_plugin_manifest, plugins_root
from agent.plugins.static_manifest import (
    StaticPluginManifest,
    load_static_plugin_manifest,
    validate_module_exports,
)


def run_plugin_doctor(
    *,
    plugin_id: str = "",
    config_path: str = "config.toml",
    workspace: Path,
    plugins_home: Path | None = None,
) -> dict[str, Any]:
    resolved_workspace = workspace
    _ = config_path
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
) -> dict[str, Any]:
    stable_root, latest_root, projection_root = _find_plugin_roots(
        plugin_id,
        plugins_home,
    )
    checks = [
        _check("policy", "ok" if enabled else "warn", f"enabled={str(enabled).lower()}")
    ]
    links_required = enabled
    if stable_root is not None:
        checks.append(
            _check(
                "install",
                "ok",
                f"stable {_entrypoint_label(stable_root)}: {stable_root}",
            )
        )
        try:
            declaration = _load_plugin_declaration(
                stable_root,
                require_static="@" in plugin_id,
            )
            checks.extend(
                _check_capabilities(
                    declaration,
                    stable_root,
                    workspace,
                    projection_root=projection_root,
                    links_required=links_required,
                )
            )
        except Exception as e:
            checks.append(_check("declaration", "error", str(e)))
    elif latest_root is None:
        checks.append(_check("install", "error", "未找到插件目录"))
    else:
        checks.append(
            _check(
                "install",
                "ok",
                f"latest candidate {_entrypoint_label(latest_root)}: {latest_root}",
            )
        )
        try:
            declaration = _load_plugin_declaration(
                latest_root,
                require_static="@" in plugin_id,
            )
            checks.extend(_check_candidate_declaration(declaration, latest_root))
            checks.extend(
                _check_empty_projection(
                    workspace,
                    projection_root=projection_root,
                    links_required=links_required,
                )
            )
        except Exception as e:
            checks.append(_check("declaration", "error", str(e)))
    if latest_root is not None and latest_root != stable_root:
        checks.append(
            _check(
                "candidate",
                "deferred",
                "latest 候选尚未 promote；workspace skill 投影继续以 stable 为准"
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
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve stable/latest artifacts and their shared projection owner."""

    # 1. Builtin 插件仍由仓库固定目录拥有。
    name, separator, marketplace = plugin_id.partition("@")
    if not separator:
        root = Path(__file__).resolve().parents[2] / "plugins" / name
        resolved = root if _plugin_root_has_entrypoint(root) else None
        return resolved, resolved, resolved

    # 2. 新安装布局以原子 pointer 为准；投影只能跟随 stable。
    base = plugins_root(plugins_home) / "cache" / marketplace / name
    pointers = read_pointers(base)
    if pointers is not None:
        return (
            resolve_pointer(base, pointers.stable),
            resolve_pointer(base, pointers.latest),
            base,
        )

    # 3. 外部插件只认原子 pointer，不扫描旧版可见目录。
    return None, None, base


def _plugin_root_has_entrypoint(root: Path) -> bool:
    """Recognize builtin plugin.py and static-manifest custom entrypoints."""

    manifest_path = root / "akashic.plugin.toml"
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = load_static_plugin_manifest(root)
        return (root / manifest.entrypoint).is_file()
    return (root / "plugin.py").is_file()


def _entrypoint_label(root: Path) -> str:
    manifest_path = root / "akashic.plugin.toml"
    if manifest_path.exists() or manifest_path.is_symlink():
        return load_static_plugin_manifest(root).entrypoint
    return "plugin.py"


def _load_plugin_declaration(
    plugin_root: Path,
    *,
    require_static: bool,
) -> ComposablePlugin:
    """读取并校验一个 v3 namespace。"""

    static_manifest = _load_optional_static_manifest(plugin_root)
    if require_static and static_manifest is None:
        raise ValueError(f"installed v3 插件缺少静态 manifest: {plugin_root}")
    module_name = f"akasic_plugin_doctor_{uuid.uuid4().hex}"
    path = plugin_root / (
        static_manifest.entrypoint if static_manifest is not None else "plugin.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=[str(path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if static_manifest is not None and getattr(module, "api_version", None) != 3:
            raise ValueError("静态 v3 manifest 与 module api_version 不一致")
        if getattr(module, "api_version", None) != 3:
            raise ValueError("plugin.py 必须声明 api_version = 3")
        if static_manifest is not None:
            validate_module_exports(
                static_manifest,
                module,
                plugin_root=plugin_root,
            )
        return ComposablePlugin.from_module(module)
    finally:
        _ = sys.modules.pop(module_name, None)


def _load_optional_static_manifest(
    plugin_root: Path,
) -> StaticPluginManifest | None:
    """读取可选的 v3 static manifest；内建源码插件可以没有 manifest。"""

    path = plugin_root / "akashic.plugin.toml"
    if not path.exists() and not path.is_symlink():
        return None
    return load_static_plugin_manifest(plugin_root)


def _check_capabilities(
    declaration: ComposablePlugin,
    plugin_root: Path,
    workspace: Path,
    *,
    projection_root: Path | None,
    links_required: bool,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for label, roots, target, subpath in (
        (
            "skills",
            _declared_skill_roots(declaration, drift=False),
            workspace / "skills",
            ("skills",),
        ),
        (
            "drift_skills",
            _declared_skill_roots(declaration, drift=True),
            workspace / "drift" / "skills",
            ("drift", "skills"),
        ),
    ):
        missing = [raw for raw in roots if not (plugin_root / raw).is_dir()]
        expected = _expected_skill_links(plugin_root, roots)
        unlinked: list[str] = []
        misdirected: list[str] = []
        stale: list[str] = []
        if links_required:
            unlinked, misdirected = _check_expected_links(target, expected)
            stale = _stale_link_names(target, expected, projection_root, subpath)
        status = (
            "error"
            if missing
            else "warn" if (unlinked or misdirected or stale) else "ok"
        )
        checks.append(
            _check(
                label,
                status,
                "roots="
                f"{len(roots)} missing={missing} unlinked={unlinked} "
                f"misdirected={misdirected} stale={stale}",
            )
        )
    servers = _declared_mcp_servers(plugin_root)
    checks.append(
        _check(
            "mcp",
            "ok",
            f"declared_servers={len(servers)} names={list(servers)}",
        )
    )
    return checks


def _check_candidate_declaration(
    declaration: ComposablePlugin,
    plugin_root: Path,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for label, roots in (
        ("candidate_skills", _declared_skill_roots(declaration, drift=False)),
        ("candidate_drift_skills", _declared_skill_roots(declaration, drift=True)),
    ):
        missing = [raw for raw in roots if not (plugin_root / raw).is_dir()]
        checks.append(
            _check(
                label,
                "error" if missing else "ok",
                f"roots={len(roots)} missing={missing}",
            )
        )
    servers = _declared_mcp_servers(plugin_root)
    checks.append(
        _check(
            "candidate_mcp",
            "ok",
            f"declared_servers={len(servers)} names={list(servers)}",
        )
    )
    return checks


def _declared_skill_roots(
    declaration: ComposablePlugin,
    *,
    drift: bool,
) -> tuple[str, ...]:
    return declaration.drift_skill_roots if drift else declaration.skill_roots


def _declared_mcp_servers(
    plugin_root: Path,
) -> tuple[str, ...]:
    """Return import-free MCP names declared by the artifact."""

    manifest = _load_optional_static_manifest(plugin_root)
    if manifest is None:
        return ()
    return tuple(server.name for server in manifest.mcp_servers)


def _check_empty_projection(
    workspace: Path,
    *,
    projection_root: Path | None,
    links_required: bool,
) -> list[dict[str, str]]:
    if not links_required:
        return []
    return [
        _check(
            label,
            "warn" if stale else "ok",
            f"roots=0 missing=[] unlinked=[] misdirected=[] stale={stale}",
        )
        for label, target, subpath in (
            ("skills", workspace / "skills", ("skills",)),
            (
                "drift_skills",
                workspace / "drift" / "skills",
                ("drift", "skills"),
            ),
        )
        for stale in [_stale_link_names(target, {}, projection_root, subpath)]
    ]


def _expected_skill_links(
    plugin_root: Path,
    roots: tuple[str, ...],
) -> dict[str, Path]:
    expected: dict[str, Path] = {}
    for raw in roots:
        skills_dir = plugin_root / raw
        if not skills_dir.is_dir():
            continue
        for child in sorted(skills_dir.iterdir(), key=lambda item: item.name):
            if child.is_dir() and (child / "SKILL.md").exists():
                if child.name not in expected:
                    expected[child.name] = child.resolve(strict=False)
    return expected


def _check_expected_links(
    target: Path,
    expected: dict[str, Path],
) -> tuple[list[str], list[str]]:
    unlinked: list[str] = []
    misdirected: list[str] = []
    for name, expected_target in expected.items():
        link = target / name
        if not link.is_symlink():
            unlinked.append(name)
        elif _link_target(link) != expected_target:
            misdirected.append(name)
    return unlinked, misdirected


def _stale_link_names(
    target: Path,
    expected: dict[str, Path],
    projection_root: Path | None,
    subpath: tuple[str, ...],
) -> list[str]:
    if projection_root is None or not target.is_dir():
        return []
    stale: list[str] = []
    for link in target.iterdir():
        if link.name in expected or not link.is_symlink():
            continue
        if _is_managed_projection_target(
            _link_target(link),
            projection_root,
            subpath,
        ):
            stale.append(link.name)
    return stale


def _link_target(link: Path) -> Path:
    raw = link.readlink()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (link.parent / raw).resolve(strict=False)


def _is_managed_projection_target(
    target: Path,
    projection_root: Path,
    subpath: tuple[str, ...],
) -> bool:
    try:
        relative = target.resolve(strict=False).relative_to(
            projection_root.resolve(strict=False)
        )
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) >= len(subpath) + 1 and parts[-(len(subpath) + 1) : -1] == subpath


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _merge_status(statuses: Any) -> str:
    values = list(statuses)
    if any(value in {"error", "broken"} for value in values):
        return "broken"
    if any(value in {"warn", "degraded", "deferred"} for value in values):
        return "degraded"
    return "healthy"
