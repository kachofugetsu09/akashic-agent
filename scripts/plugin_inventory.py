#!/usr/bin/env python3
"""登记全部插件包与真实导入欠账，复用同一份静态边界规则。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.plugins.static_manifest import load_static_plugin_manifest
from scripts import plugin_boundary as boundary


def build_inventory(repo_root: Path) -> dict[str, object]:
    """按 Git 跟踪范围登记发布身份、实现依赖和不能静态证明的动态入口。"""
    root = repo_root.resolve()
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines()
    files = [name for name in tracked if name.endswith(".py")]
    sources = {name: (root / name).read_text() for name in files}
    imports = boundary.collect_imports(files, sources)
    findings = boundary.import_findings(imports)
    packages = sorted({name.split("/")[1] for name in tracked
                       if name.startswith("plugins/") and len(name.split("/")) >= 3})
    rows = []
    violations = [{"kind": rule, "file": item.importer, "target": item.module}
                  for rule, items in findings.items() for item in items]
    for package in packages:
        prefix = "plugins/" + package + "/"
        package_files = [name for name in tracked if name.startswith(prefix)]
        digest = hashlib.sha256()
        for name in package_files:
            digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
        manifest_path = prefix + "akashic.plugin.toml"
        manifest = None
        error = None
        if manifest_path not in tracked:
            classification = "support-package"
        else:
            try:
                parsed = load_static_plugin_manifest(root / prefix)
                manifest = {"name": parsed.name, "version": parsed.version,
                            "entrypoint": parsed.entrypoint, "api_version": parsed.api_version}
                classification = "manifest-plugin"
            except ValueError as exc:
                classification, error = "invalid-manifest", str(exc)
        if classification != "manifest-plugin":
            violations.append({"kind": "not_installable_artifact", "file": prefix, "error": error})
        package_imports = [item for item in imports if item.importer.startswith(prefix)]
        dynamic = []
        for name in files:
            if not name.startswith(prefix):
                continue
            tree = ast.parse(sources[name])
            # 字面动态导入由同一 checker 解析；计算式路径仍要求运行时证据。
            aliases = {"__import__", "import_module"}
            aliases.update(alias.asname or alias.name for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom) and node.module == "importlib"
                           for alias in node.names if alias.name == "import_module")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Name) and node.func.id in aliases
                    or isinstance(node.func, ast.Attribute) and node.func.attr in {"import_module", "spec_from_file_location"}
                ):
                    dynamic.append({"file": name, "line": node.lineno,
                                    "expression": ast.unparse(node)})
        rows.append({"package": package, "classification": classification, "manifest": manifest,
                     "manifest_error": error, "source_digest": digest.hexdigest(),
                     "static_imports": [{"file": item.importer, "target": item.module} for item in package_imports],
                     "dynamic_dependencies": dynamic,
                     "implementation_dependencies": [item for item in violations if item["file"].startswith(prefix)]})
    return {"schema_version": 1, "repository": str(root), "packages": rows,
            "summary": {"package_count": len(rows),
                "manifest_plugin_count": sum(row["classification"] == "manifest-plugin" for row in rows),
                "support_package_count": sum(row["classification"] == "support-package" for row in rows),
                "static_boundary_violation_count": len(violations)},
            "core_consumer_imports": [{"file": item.importer, "target": item.module} for item in findings["R1"]],
            "static_boundary_violations": violations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = build_inventory(args.repo_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")
    return int(args.strict and bool(report["static_boundary_violations"]))


if __name__ == "__main__":
    raise SystemExit(main())
