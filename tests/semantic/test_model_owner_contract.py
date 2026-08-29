from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONSUMER_ROOTS = (
    ROOT / "agent",
    ROOT / "bootstrap",
    ROOT / "core",
    ROOT / "docker/debug",
    ROOT / "eval",
    ROOT / "benchmark",
)
ORDINARY_PLUGIN_PACKAGES = (
    "plugins.models",
    "plugins.openai_compatible",
    "plugins.codex",
    "plugins.opencode_go",
)
PUBLIC_CONTROL_PROBES = (
    ROOT / "docker/debug/akasha_v2_runtime_probe.py",
    ROOT / "docker/debug/programmatic_control_probe.py",
)


def test_model_consumers_do_not_import_ordinary_plugin_sources() -> None:
    """Keep Core and operational consumers independent of repository plugin copies."""

    violations: list[str] = []
    for root in CONSUMER_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                    if imported.startswith(ORDINARY_PLUGIN_PACKAGES):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    if any(
                        alias.name.startswith(ORDINARY_PLUGIN_PACKAGES)
                        for alias in node.names
                    ):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def test_model_debug_probes_use_public_control_fixture() -> None:
    """Keep formal probes on the same public model-control boundary as clients."""

    for path in PUBLIC_CONTROL_PROBES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports_public_helper = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "docker.debug.model_plugin_fixture"
            and any(alias.name == "add_openai_models" for alias in node.names)
            for node in ast.walk(tree)
        )
        calls_public_helper = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "add_openai_models"
            for node in ast.walk(tree)
        )
        assert imports_public_helper, path.relative_to(ROOT)
        assert calls_public_helper, path.relative_to(ROOT)
