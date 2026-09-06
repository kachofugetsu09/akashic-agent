from pathlib import Path
import os

import pytest
from yoyo import get_backend, read_migrations

from agent.migrations.context import bind_migration_context
from bootstrap.init_workspace import init_workspace


def test_yoyo_installs_grants_without_overwriting_operator_choice(tmp_path):
    """真实 yoyo 发布完整默认授权，重试及已有配置均保留原选择。"""
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260906_06_model_call_timing.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    source = Path(__file__).parents[1] / "migrations/yoyo/20260907_01_context_material_grants.py"
    (directory / source.name).write_bytes(source.read_bytes())
    workspace = tmp_path / "workspace"
    path = workspace / "plugin-data/context-builtin/config.local.toml"
    backend = get_backend(f"sqlite:///{tmp_path / 'ledger.db'}")
    migrations = read_migrations(str(directory))
    with backend, bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
        backend.apply_migrations(backend.to_apply(migrations))
        assert not backend.to_apply(migrations)
        initialized = tmp_path / "initialized"
        init_workspace(config_path=tmp_path / "init-config.toml", workspace=initialized)
        assert path.read_bytes() == (initialized / "plugin-data/context-builtin/config.local.toml").read_bytes()
        assert path.stat().st_mode & 0o777 == 0o600
        path.write_text('prompt_sources = {custom = "custom"}\n')
        before = (path.stat().st_ino, path.read_bytes())
        migrations[-1].module.install_context_grants(None)
        assert (path.stat().st_ino, path.read_bytes()) == before


def test_failed_grants_publish_leaves_no_partial_config(tmp_path, monkeypatch):
    """目录发布失败只清理本次临时文件，重试能正常建立授权。"""
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260906_06_model_call_timing.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    source = Path(__file__).parents[1] / "migrations/yoyo/20260907_01_context_material_grants.py"
    (directory / source.name).write_bytes(source.read_bytes())
    migration = read_migrations(str(directory))[-1]
    migration.load()
    module = migration.module
    workspace = tmp_path / "workspace"
    path = workspace / "plugin-data/context-builtin/config.local.toml"
    original = os.link
    def reject(source, destination):
        raise OSError("fixture link failure")
    with bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
        monkeypatch.setattr(os, "link", reject)
        with pytest.raises(OSError, match="fixture link failure"):
            module.install_context_grants(None)
        assert not path.exists() and not list(path.parent.iterdir())
        monkeypatch.setattr(os, "link", original)
        module.install_context_grants(None)
        assert path.exists()


def test_force_init_backs_up_config_and_preserves_owned_assets(tmp_path):
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.toml"
    init_workspace(config_path=config, workspace=workspace)
    config.write_text(config.read_text() + '\n# retained operator credential\n')
    original = config.read_bytes()
    assets = [workspace / "memory/VEDA.md", workspace / "memes/manifest.json",
              workspace / "plugin-data/context-builtin/config.local.toml"]
    for path in assets:
        path.write_text("operator owned bytes\n")
    init_workspace(config_path=config, workspace=workspace, force=True)
    backups = list(tmp_path.glob("config.toml.before-init-*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert backups[0].stat().st_mode & 0o777 == 0o600
    assert all(path.read_text() == "operator owned bytes\n" for path in assets)
