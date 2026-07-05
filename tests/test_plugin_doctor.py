from __future__ import annotations

import json
from pathlib import Path

from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.global_registry import upsert_plugin_registry_entry


def test_plugin_doctor_reports_healthy_skill_plugin(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_root = plugins_home / "cache" / "github" / "demo" / "0.1.0"
    skill_dir = plugin_root / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (plugin_root / ".aka-plugin").mkdir()
    (plugin_root / ".aka-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "0.1.0",
                "description": "demo plugin",
                "paths": {"skills": ["skills"]},
            }
        ),
        encoding="utf-8",
    )
    (workspace / "skills").mkdir(parents=True)
    (workspace / "skills" / "demo-skill").symlink_to(skill_dir, target_is_directory=True)
    upsert_plugin_registry_entry(
        "demo@github",
        {
            "plugin_id": "demo@github",
            "plugin_root": str(plugin_root),
            "enabled": True,
            "local_disabled": False,
            "active": True,
            "capabilities": {"lifecycle": False, "skills": True, "mcp": False},
            "skills": ["demo-skill"],
            "drift_skills": [],
            "mcp_servers": [],
            "lifecycle_entry": "",
        },
        plugins_home=plugins_home,
    )

    report = run_plugin_doctor(
        plugin_id="demo@github",
        plugins_home=plugins_home,
        workspace=workspace,
        config_path=str(tmp_path / "missing.toml"),
    )

    assert report["status"] == "healthy"
    checks = {item["name"]: item for item in report["plugins"][0]["checks"]}
    assert checks["install"]["status"] == "ok"
    assert checks["skills"]["status"] == "ok"


def test_plugin_doctor_reports_missing_skill_and_inactive_lifecycle(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_root = plugins_home / "cache" / "github" / "demo" / "0.1.0"
    plugin_root.mkdir(parents=True)
    (plugin_root / ".aka-plugin").mkdir()
    (plugin_root / ".aka-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "0.1.0",
                "description": "demo plugin",
                "paths": {"skills": ["skills"]},
                "akashic": {"lifecycle": {"entry": "plugin.py"}},
            }
        ),
        encoding="utf-8",
    )
    (plugin_root / "plugin.py").write_text("class X: pass\n", encoding="utf-8")
    (plugin_root / "skills" / "demo-skill").mkdir(parents=True)
    upsert_plugin_registry_entry(
        "demo@github",
        {
            "plugin_id": "demo@github",
            "plugin_root": str(plugin_root),
            "enabled": True,
            "local_disabled": False,
            "active": False,
            "capabilities": {"lifecycle": True, "skills": True, "mcp": False},
            "skills": ["demo-skill"],
            "drift_skills": [],
            "mcp_servers": [],
            "lifecycle_entry": str(plugin_root / "plugin.py"),
        },
        plugins_home=plugins_home,
    )

    report = run_plugin_doctor(
        plugin_id="demo@github",
        plugins_home=plugins_home,
        workspace=workspace,
        config_path=str(tmp_path / "missing.toml"),
    )

    assert report["status"] == "degraded"
    checks = {item["name"]: item for item in report["plugins"][0]["checks"]}
    assert checks["lifecycle"]["status"] == "warn"
    assert checks["skills"]["status"] == "warn"
    text = format_plugin_doctor_report(report)
    assert "plugin doctor demo@github" in text
