import tomllib

from agent.migrations.context import bind_migration_context
from tests.legacy_migration_loader import load_migration_namespace


def _migration():
    return load_migration_namespace("20260909_01_tool_provider_views")


def test_tool_provider_migration_moves_only_exact_skill_owner_and_keeps_backup(tmp_path):
    module = _migration()
    workspace = tmp_path / "workspace"
    path = workspace / "plugin-data/context-builtin/config.local.toml"
    path.parent.mkdir(parents=True)
    before = (
        'prompt_sources = {default_prompt = "prompt", skills = "skills"}\n'
        'summary_source = ["compaction", "compaction"]\n'
    )
    path.write_text(before)
    with bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
        module.move_skill_prompt_owner(None)
        backup = path.with_name("config.before-tool-provider-views.toml")
        assert backup.read_text() == before
        assert tomllib.loads(path.read_text())["prompt_sources"]["skills"] == "standard_tools"
        after = path.read_bytes()
        module.move_skill_prompt_owner(None)
        assert path.read_bytes() == after
        assert backup.read_text() == before


def test_tool_provider_migration_keeps_missing_and_custom_skill_owner(tmp_path):
    module = _migration()
    for index, before in enumerate((
        'summary_source = ["compaction", "compaction"]\n',
        'prompt_sources = {skills = "my_skills", custom = "custom"}\n',
    )):
        workspace = tmp_path / f"workspace-{index}"
        path = workspace / "plugin-data/context-builtin/config.local.toml"
        path.parent.mkdir(parents=True)
        path.write_text(before)
        with bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
            module.move_skill_prompt_owner(None)
        assert path.read_text() == before
        assert not path.with_name("config.before-tool-provider-views.toml").exists()
