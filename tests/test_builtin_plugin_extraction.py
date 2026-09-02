from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source_name", "module_name", "plugin_name"),
    (
        ("eventmail", "ordinary_eventmail", "eventmail"),
        ("wake", "ordinary_wake", "wake"),
    ),
)
def test_builtin_plugin_entrypoint_loads_from_an_external_directory(
    tmp_path: Path,
    source_name: str,
    module_name: str,
    plugin_name: str,
) -> None:
    """Prove runtime imports resolve inside the copied plugin, not `plugins.*`."""

    external_root = tmp_path / "installed" / source_name
    shutil.copytree(ROOT / "plugins" / source_name, external_root)
    spec = importlib.util.spec_from_file_location(
        module_name,
        external_root / "plugin.py",
        submodule_search_locations=[str(external_root)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for imported in tuple(sys.modules):
            if imported == module_name or imported.startswith(module_name + "."):
                del sys.modules[imported]

    assert module.api_version == 3
    assert module.name == plugin_name
