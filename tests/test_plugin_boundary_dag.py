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


def test_ledger_generation_attachments_use_public_contract_identity() -> None:
    """generation 私有加载 ledger 时，附件边界类型必须与公开 plugins.ledger.contract 同身份。"""
    import importlib

    from agent.plugins.importer import FreshPluginImporter
    from plugins.ledger.contract import AttachmentKind, AttachmentRef

    ledger_root = Path(__file__).resolve().parent.parent / "plugins" / "ledger"
    gen_pkg = "_test_ledger_gen_identity"
    importer = FreshPluginImporter()
    importer.register(gen_pkg, ledger_root)
    try:
        spec = importer.root_spec(gen_pkg, ledger_root / "plugin.py")
        assert spec is not None and spec.loader is not None
        pkg_mod = importlib.util.module_from_spec(spec)
        sys.modules[gen_pkg] = pkg_mod
        spec.loader.exec_module(pkg_mod)

        attachments_mod = importlib.import_module(f"{gen_pkg}.attachments")
        import_mod = importlib.import_module(f"{gen_pkg}.attachment_import")
        assert attachments_mod.AttachmentKind is AttachmentKind
        assert attachments_mod.AttachmentRef is AttachmentRef
        assert import_mod.AttachmentKind is AttachmentKind
        assert import_mod.AttachmentRef is AttachmentRef
    finally:
        importer.unregister(gen_pkg)
        for name in [k for k in sys.modules if k == gen_pkg or k.startswith(f"{gen_pkg}.")]:
            sys.modules.pop(name, None)
