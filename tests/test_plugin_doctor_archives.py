from pathlib import Path


from agent.plugins.archive import PluginArchive


def test_asset_doctor_checks_arbitrary_categories_without_skill_links(tmp_path):
    from types import ModuleType
    from agent.plugins.composable import ComposablePlugin
    from agent.plugins.doctor import _check_capabilities

    module = ModuleType("fixture")
    exec("api_version=3\nname='fixture'\nversion='1.0.0'\n"
         "asset_roots={'abde': ('records',)}\nasync def apply(ctx, config): pass\n", module.__dict__)
    declaration = ComposablePlugin.from_module(module)
    (tmp_path / "records").mkdir()
    checks = _check_capabilities(declaration, tmp_path)
    assert next(item for item in checks if item["name"] == "assets:abde")["status"] == "ok"
    (tmp_path / "records").rmdir()
    checks = _check_capabilities(declaration, tmp_path)
    assert next(item for item in checks if item["name"] == "assets:abde")["status"] == "error"


def test_read_only_archive_does_not_create_directory(tmp_path):
    root = tmp_path / "missing"
    PluginArchive(root, create=False)
    assert not root.exists()
