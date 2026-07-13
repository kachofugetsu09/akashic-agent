from __future__ import annotations

from pathlib import Path

from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.manifest import upsert_plugin_manifest
from bootstrap.init_workspace import init_workspace


def _init_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    _ = init_workspace(config_path=config_path, workspace=tmp_path / "workspace")
    return config_path


def test_plugin_doctor_reads_programmatic_capabilities(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_root = plugins_home / "cache" / "github" / "demo" / "1.0.0"
    skill_dir = plugin_root / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("skill", encoding="utf-8")
    (plugin_root / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class DemoPlugin(Plugin):\n"
        "    name = 'demo'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills',)\n",
        encoding="utf-8",
    )
    (workspace / "skills").mkdir(parents=True)
    (workspace / "skills" / "demo-skill").symlink_to(skill_dir, target_is_directory=True)
    upsert_plugin_manifest("demo@github", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "healthy"
    assert "plugin doctor demo@github" in format_plugin_doctor_report(report)


def test_plugin_doctor_reports_broken_declaration(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    plugin_root = plugins_home / "cache" / "github" / "demo" / "1.0.0"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text("class X: pass\n", encoding="utf-8")
    upsert_plugin_manifest("demo@github", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
    )

    assert report["status"] == "broken"


def test_plugin_doctor_finds_builtin_plugin(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    upsert_plugin_manifest("default_proactive", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="default_proactive",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "healthy"


def test_plugin_doctor_skips_inactive_default_memory_drift_links(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    _ = init_workspace(
        config_path=config_path,
        workspace=workspace,
    )
    config_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace('engine = ""', 'engine = "akasha"', 1),
        encoding="utf-8",
    )
    upsert_plugin_manifest("default_memory", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="default_memory",
        config_path=str(config_path),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "healthy"
