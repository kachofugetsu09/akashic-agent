from pathlib import Path
import os
import tomllib

import pytest
from yoyo import get_backend

from agent.migrations.context import bind_migration_context
from bootstrap.init_workspace import init_workspace
from tests.legacy_migration_loader import load_bundle_migrations


def test_yoyo_installs_grants_without_overwriting_operator_choice(tmp_path):
    """真实 yoyo 发布完整默认授权，重试及已有配置均保留原选择。"""
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260906_06_model_call_timing.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    source = Path(__file__).parents[1] / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260907_01_context_material_grants.py"
    (directory / source.name).write_bytes(source.read_bytes())
    workspace = tmp_path / "workspace"
    path = workspace / "plugin-data/context-builtin/config.local.toml"
    backend = get_backend(f"sqlite:///{tmp_path / 'ledger.db'}")
    migrations = load_bundle_migrations(directory)
    with backend, bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
        backend.apply_migrations(backend.to_apply(migrations))
        assert not backend.to_apply(migrations)
        initialized = tmp_path / "initialized"
        init_workspace(config_path=tmp_path / "init-config.toml", workspace=initialized)
        old = tomllib.loads(path.read_text())
        assert old == {
            "prompt_sources": {
                "default_prompt": "prompt",
                "markdown_memory": "markdown_memory",
            },
            "summary_source": ["compaction", "compaction"],
        }
        current = tomllib.loads(
            (initialized / "plugin-data/context-builtin/config.local.toml").read_text()
        )
        assert current["prompt_sources"]["skills"] == "standard_tools"
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
    source = Path(__file__).parents[1] / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260907_01_context_material_grants.py"
    (directory / source.name).write_bytes(source.read_bytes())
    migration = load_bundle_migrations(directory)[-1]
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


@pytest.mark.parametrize("custom", ["default", "custom", "comment"])
def test_skill_prompt_grant_backs_up_defaults_and_keeps_custom_choice(tmp_path, custom):
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260907_02_retire_legacy_agent_config.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    source = Path(__file__).parents[1] / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260907_03_skill_prompt_grant.py"
    (directory / source.name).write_bytes(source.read_bytes())
    workspace = tmp_path / "workspace"
    path = workspace / "plugin-data/context-builtin/config.local.toml"
    path.parent.mkdir(parents=True)
    before = ('prompt_sources = {custom = "custom"}\n' if custom == "custom" else
              'prompt_sources = {default_prompt = "prompt", markdown_memory = "markdown_memory"}\nsummary_source = ["compaction", "compaction"]\n')
    if custom == "comment":
        before += "# operator note\n"
    path.write_text(before)
    backend = get_backend(f"sqlite:///{tmp_path / 'ledger.db'}")
    migrations = load_bundle_migrations(directory)
    with backend, bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
        backend.apply_migrations(backend.to_apply(migrations))
        migrations[-1].module.grant_skills(None)
    backup = path.with_name("config.before-skill-prompt-grant.toml")
    if custom != "default":
        assert path.read_text() == before and not backup.exists()
    else:
        assert backup.read_text() == before
        assert tomllib.loads(path.read_text())["prompt_sources"]["skills"] == "skills"


def test_skill_prompt_grant_rejects_symlinked_plugin_data_without_external_write(tmp_path):
    """父目录越界时在读取或备份配置前失败，外部文件保持不变。"""
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260907_02_retire_legacy_agent_config.py").write_text(
        'from yoyo import step\nsteps = [step("SELECT 1")]\n'
    )
    source = Path(__file__).parents[1] / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260907_03_skill_prompt_grant.py"
    (directory / source.name).write_bytes(source.read_bytes())
    migration = load_bundle_migrations(directory)[-1]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external-plugin-data"
    config_dir = external / "context-builtin"
    config_dir.mkdir(parents=True)
    config = config_dir / "config.local.toml"
    config.write_text(
        'prompt_sources = {default_prompt = "prompt", markdown_memory = "markdown_memory"}\n'
        'summary_source = ["compaction", "compaction"]\n'
    )
    before = config.read_bytes()
    (workspace / "plugin-data").symlink_to(external, target_is_directory=True)

    with bind_migration_context(
        config_path=tmp_path / "config.toml", workspace=workspace
    ):
        with pytest.raises(ValueError, match="不能穿过符号链接"):
            migration.module.grant_skills(None)

    assert config.read_bytes() == before
    assert not config.with_name("config.before-skill-prompt-grant.toml").exists()
