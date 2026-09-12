#!/usr/bin/env python3
"""Build an auditable inventory of plugin packages and implementation edges.

The inventory intentionally includes packages without a V3 manifest.  Such a
package is reported as a support package and is not silently treated as an
external-installable plugin.  ``--strict`` is a static boundary check: it
fails on the current historical import debt instead of turning that debt into
an implicit allowlist.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PLUGIN_ROOT = "plugins"
CORE_PREFIXES = (
    "agent",
    "bootstrap",
    "bus",
    "config",
    "infra",
    "main",
    "session",
)
PUBLIC_PLUGIN_API_PREFIXES = ("agent.plugin_composition",)
PROJECT_ROOTS = CORE_PREFIXES + ("plugins",)
DYNAMIC_IMPORT_NAMES = frozenset(
    {
        "__import__",
        "import_module",
        "importlib.import_module",
        "importlib.util.spec_from_file_location",
        "importlib.machinery.SourceFileLoader",
    }
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()


def _module_name(path: str) -> str:
    """Return the import-like name of one source target."""

    return path.replace("/", ".").removesuffix(".py")


def _is_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _literal(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return "<dynamic>"


def _target_from_import(
    node: ast.Import | ast.ImportFrom, alias: ast.alias
) -> tuple[str, str]:
    if isinstance(node, ast.Import):
        return alias.name, "import"
    if node.level:
        dots = "." * node.level
        return (
            f"{dots}{node.module or ''}{('.' + alias.name) if alias.name != '*' else ''}",
            "relative",
        )
    module = node.module or ""
    return (
        f"{module}.{alias.name}" if module and alias.name != "*" else module
    ), "from"


def _edge(
    path: Path, root: Path, node: ast.AST, target: str, kind: str
) -> dict[str, Any]:
    return {
        "file": _relative(path, root),
        "line": int(getattr(node, "lineno", 0)),
        "target": target,
        "kind": kind,
    }


def _scan_python_file(
    path: Path, root: Path, package: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "implementation_dependencies": [],
        "static_imports": [],
        "relative_imports": [],
        "sibling_plugin_imports": [],
        "core_imports": [],
        "private_core_imports": [],
        "dynamic_dependencies": [],
    }
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        result["dynamic_dependencies"].append(
            {
                "file": _relative(path, root),
                "line": 0,
                "kind": "parse_error",
                "target": type(error).__name__,
                "detail": str(error),
            }
        )
        return result

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            aliases = node.names
            for alias in aliases:
                target, kind = _target_from_import(node, alias)
                record = _edge(path, root, node, target, kind)
                result["static_imports"].append(record)
                if kind == "relative":
                    result["relative_imports"].append(record)
                if target.startswith("plugins."):
                    result["implementation_dependencies"].append(record)
                    sibling = target.split(".", 2)[1]
                    if sibling != package:
                        result["sibling_plugin_imports"].append(
                            {**record, "sibling_package": sibling}
                        )
                elif _is_prefix(target, CORE_PREFIXES):
                    result["implementation_dependencies"].append(record)
                    result["core_imports"].append(record)
                    if not _is_prefix(target, PUBLIC_PLUGIN_API_PREFIXES):
                        result["private_core_imports"].append(record)
                elif target and not target.startswith("."):
                    # Keep non-stdlib project-looking imports visible without
                    # guessing whether a third-party package is truly safe.
                    first = target.split(".", 1)[0]
                    if first in PROJECT_ROOTS:
                        result["implementation_dependencies"].append(record)

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in DYNAMIC_IMPORT_NAMES:
                result["dynamic_dependencies"].append(
                    {
                        "file": _relative(path, root),
                        "line": int(getattr(node, "lineno", 0)),
                        "kind": "dynamic_import",
                        "target": _literal(node.args[0]) if node.args else None,
                        "expression": name,
                    }
                )
        if isinstance(node, ast.Attribute) and node.attr == "path":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                result["dynamic_dependencies"].append(
                    {
                        "file": _relative(path, root),
                        "line": int(getattr(node, "lineno", 0)),
                        "kind": "sys_path_mutation_or_read",
                        "target": "sys.path",
                    }
                )
    return result


def _merge_edges(target: list[dict[str, Any]], values: list[dict[str, Any]]) -> None:
    seen = {
        (item.get("file"), item.get("line"), item.get("kind"), item.get("target"))
        for item in target
    }
    for value in values:
        key = (
            value.get("file"),
            value.get("line"),
            value.get("kind"),
            value.get("target"),
        )
        if key not in seen:
            target.append(value)
            seen.add(key)


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "akashic.plugin.toml"
    if not path.exists():
        return {
            "path": None,
            "valid": False,
            "error": "missing_static_manifest",
        }
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        required = ("schema_version", "name", "version", "api_version", "entrypoint")
        missing = [key for key in required if key not in raw]
        if missing:
            return {
                "path": path.name,
                "valid": False,
                "present": True,
                "error": f"manifest_missing_fields:{','.join(missing)}",
            }
        return {
            "path": path.name,
            "valid": True,
            "present": True,
            "name": raw.get("name"),
            "version": raw.get("version"),
            "api_version": raw.get("api_version"),
            "entrypoint": raw.get("entrypoint"),
            "identity_fields": {
                key: raw.get(key)
                for key in (
                    "schema_version",
                    "name",
                    "version",
                    "api_version",
                    "entrypoint",
                )
            },
        }
    except (OSError, tomllib.TOMLDecodeError, TypeError) as error:
        return {
            "path": path.name,
            "valid": False,
            "present": True,
            "error": f"manifest_parse_error:{type(error).__name__}:{error}",
        }


def _entry_digest(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(_relative(path, root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _inventory_entry(root: Path, package_root: Path) -> dict[str, Any]:
    package = package_root.name
    py_files = sorted(package_root.rglob("*.py"))
    manifest = _manifest(package_root)
    edges: dict[str, list[dict[str, Any]]] = {
        "implementation_dependencies": [],
        "static_imports": [],
        "relative_imports": [],
        "sibling_plugin_imports": [],
        "core_imports": [],
        "private_core_imports": [],
        "dynamic_dependencies": [],
    }
    for path in py_files:
        scanned = _scan_python_file(path, root, package)
        for key, values in scanned.items():
            _merge_edges(edges[key], values)
    entrypoint = manifest.get("entrypoint") if manifest.get("valid") else None
    conventional = [
        name
        for name in ("plugin.py", "message_plugin.py")
        if (package_root / name).is_file()
    ]
    if (
        isinstance(entrypoint, str)
        and (package_root / entrypoint).is_file()
        and entrypoint not in conventional
    ):
        conventional.append(entrypoint)
    entrypoint_exists = (
        isinstance(entrypoint, str) and (package_root / entrypoint).is_file()
    )
    return {
        "package": package,
        "root": _relative(package_root, root),
        "classification": (
            "manifest-plugin"
            if manifest.get("valid")
            else "invalid-manifest" if manifest.get("present") else "support-package"
        ),
        "manifest": manifest,
        "declared_entrypoint_exists": entrypoint_exists,
        "entrypoints": sorted(conventional),
        "python_files": [_relative(path, root) for path in py_files],
        "source_digest": _entry_digest(py_files, root),
        **{
            key: sorted(
                values,
                key=lambda item: (
                    str(item.get("file")),
                    int(item.get("line", 0)),
                    str(item.get("target")),
                ),
            )
            for key, values in edges.items()
        },
    }


def _core_consumers(root: Path) -> list[dict[str, Any]]:
    """Record Core-side imports of plugin implementation packages."""

    records: list[dict[str, Any]] = []
    paths = [root / name for name in CORE_PREFIXES if (root / name).exists()]
    paths.extend([root / "main.py", root / "config.py"])
    for base in paths:
        candidates = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in candidates:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for alias in node.names:
                    target, _ = _target_from_import(node, alias)
                    if target.startswith("plugins."):
                        records.append(
                            {
                                "file": _relative(path, root),
                                "line": int(getattr(node, "lineno", 0)),
                                "target": target,
                            }
                        )
    unique = {(item["file"], item["line"], item["target"]): item for item in records}
    return [
        unique[key]
        for key in sorted(unique, key=lambda item: (item[0], item[1], item[2]))
    ]


def build_inventory(repo_root: Path) -> dict[str, Any]:
    """Build a deterministic inventory from the checkout's current source."""

    root = repo_root.resolve(strict=True)
    plugin_root = root / PLUGIN_ROOT
    if not plugin_root.is_dir():
        raise ValueError(f"插件根目录不存在: {plugin_root}")
    entries = [
        _inventory_entry(root, path)
        for path in sorted(plugin_root.iterdir())
        if path.is_dir() and not path.name.startswith("__")
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest_entries = [
        entry for entry in entries if entry["classification"] == "manifest-plugin"
    ]
    violations: list[dict[str, Any]] = []
    for entry in entries:
        if entry["classification"] != "manifest-plugin":
            violations.append(
                {
                    "package": entry["package"],
                    "kind": "not_installable_artifact",
                    "reason": (
                        "package has an invalid static V3 manifest"
                        if entry["classification"] == "invalid-manifest"
                        else "package has no static V3 manifest; retained as support-package evidence"
                    ),
                }
            )
        if entry["manifest"].get("valid") and not entry["declared_entrypoint_exists"]:
            violations.append(
                {
                    "package": entry["package"],
                    "kind": "missing_entrypoint",
                    "reason": "manifest declares no existing entrypoint",
                }
            )
        for edge in entry["sibling_plugin_imports"]:
            violations.append(
                {
                    **edge,
                    "package": entry["package"],
                    "kind": "sibling_plugin_import",
                }
            )
        for edge in entry["private_core_imports"]:
            violations.append(
                {
                    **edge,
                    "package": entry["package"],
                    "kind": "private_core_import",
                }
            )
        for edge in entry["dynamic_dependencies"]:
            violations.append(
                {
                    **edge,
                    "package": entry["package"],
                    "kind": "dynamic_implementation_dependency",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": str(root),
        "git_head": commit,
        "plugin_root": PLUGIN_ROOT,
        "packages": entries,
        "core_consumer_imports": _core_consumers(root),
        "summary": {
            "package_count": len(entries),
            "manifest_plugin_count": len(manifest_entries),
            "support_package_count": len(entries) - len(manifest_entries),
            "manifest_plugins": [entry["package"] for entry in manifest_entries],
            "support_packages": [
                entry["package"]
                for entry in entries
                if entry["classification"] == "support-package"
            ],
            "invalid_manifest_packages": [
                entry["package"]
                for entry in entries
                if entry["classification"] == "invalid-manifest"
            ],
            "packages_with_sibling_imports": [
                entry["package"] for entry in entries if entry["sibling_plugin_imports"]
            ],
            "packages_with_private_core_imports": [
                entry["package"] for entry in entries if entry["private_core_imports"]
            ],
            "packages_with_dynamic_dependencies": [
                entry["package"] for entry in entries if entry["dynamic_dependencies"]
            ],
            "core_consumer_import_count": len(_core_consumers(root)),
            "static_boundary_violation_count": len(violations),
        },
        "static_boundary_violations": sorted(
            violations,
            key=lambda item: (
                str(item.get("package")),
                str(item.get("file", "")),
                int(item.get("line", 0)),
                str(item.get("kind")),
            ),
        ),
    }


def _text_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"inventory head={report['git_head']}",
        f"packages={summary['package_count']} manifest_plugins={summary['manifest_plugin_count']} support_packages={summary['support_package_count']}",
        f"static_boundary_violations={summary['static_boundary_violation_count']}",
    ]
    for violation in report["static_boundary_violations"]:
        location = (
            f"{violation['file']}:{violation['line']}"
            if "file" in violation
            else violation["package"]
        )
        target = f" target={violation.get('target')}" if violation.get("target") else ""
        lines.append(f"FAIL {violation['kind']} {location}{target}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--strict", action="store_true", help="静态边界存在一项欠账即返回 1"
    )
    args = parser.parse_args(argv)
    try:
        report = build_inventory(args.repo_root)
    except (OSError, ValueError) as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.format == "text":
        rendered = _text_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if args.strict and report["static_boundary_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
