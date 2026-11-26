from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_boundary_module():
    script = Path(__file__).resolve().parent.parent / "scripts" / "plugin_boundary.py"
    spec = importlib.util.spec_from_file_location("plugin_boundary", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r15_accepts_lower_triangular_dag() -> None:
    pb = _load_boundary_module()
    imports = [
        pb.Import("plugins/b/plugin.py", "plugins.a.contract", True),
        pb.Import("plugins/c/plugin.py", "plugins.b.contract", True),
        pb.Import("plugins/c/plugin.py", "plugins.a.contract", True),
    ]
    assert pb.check_plugin_dag(imports) == []


def test_r15_rejects_two_node_and_three_node_cycles() -> None:
    pb = _load_boundary_module()
    imports = [
        pb.Import("plugins/a/contract.py", "plugins.b.contract", True),
        pb.Import("plugins/b/contract.py", "plugins.a.contract", True),
        pb.Import("plugins/x/plugin.py", "plugins.y.contract", True),
        pb.Import("plugins/y/plugin.py", "plugins.z.contract", True),
        pb.Import("plugins/z/plugin.py", "plugins.x.contract", True),
    ]
    errors = pb.check_plugin_dag(imports)
    assert len(errors) == 2
    assert any("a, b" in err and "a->b" in err and "b->a" in err for err in errors)
    assert any("x, y, z" in err for err in errors)
