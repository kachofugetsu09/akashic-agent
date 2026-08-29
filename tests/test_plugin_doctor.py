from __future__ import annotations

from pathlib import Path

import pytest

import agent.plugins.doctor as plugin_doctor
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.manifest import upsert_plugin_manifest
from bootstrap.init_workspace import init_workspace


def _init_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    _ = init_workspace(config_path=config_path, workspace=tmp_path / "workspace")
    return config_path


def _write_artifact_plugin(
    plugin_base: Path,
    artifact_id: str,
    *,
    skills: dict[str, str],
) -> Path:
    plugin_root = plugin_base / ".artifacts" / artifact_id
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'demo'\n"
        "version = '1.0.0'\n"
        "skill_roots = ('skills',)\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    _write_static_manifest(plugin_root, name="demo")
    for name, body in skills.items():
        skill_dir = plugin_root / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return plugin_root


def _write_static_manifest(
    plugin_root: Path,
    *,
    name: str,
    entrypoint: str = "plugin.py",
    mcp_names: tuple[str, ...] = (),
) -> None:
    mcp_manifest = ""
    for mcp_name in mcp_names:
        runner = plugin_root / "mcp" / f"{mcp_name}.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("", encoding="utf-8")
        mcp_manifest += (
            "\n[[mcp]]\n" f"name = {mcp_name!r}\n" f"command = ['mcp/{mcp_name}.py']\n"
        )
    (plugin_root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {name!r}\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        f"entrypoint = {entrypoint!r}\n"
        f"{mcp_manifest}",
        encoding="utf-8",
    )


def _write_builtin_plugin(
    builtin_root: Path,
    folder: str,
    declared_name: str,
    *,
    static: bool = True,
) -> Path:
    plugin_root = builtin_root / folder
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text(
        "api_version = 3\n"
        f"name = {declared_name!r}\n"
        "version = '1.0.0'\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    if static:
        _write_static_manifest(plugin_root, name=declared_name)
    return plugin_root


def _check(report: dict[str, object], name: str) -> dict[str, str]:
    plugins = report["plugins"]
    assert isinstance(plugins, list)
    checks = plugins[0]["checks"]
    assert isinstance(checks, list)
    return next(check for check in checks if check["name"] == name)


def test_plugin_doctor_reads_programmatic_capabilities(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "github" / "demo"
    plugin_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    skill_dir = plugin_root / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("skill", encoding="utf-8")
    (plugin_root / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'demo'\n"
        "version = '1.0.0'\n"
        "skill_roots = ('skills',)\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    _write_static_manifest(plugin_root, name="demo")
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )
    (workspace / "skills").mkdir(parents=True)
    (workspace / "skills" / "demo-skill").symlink_to(
        skill_dir, target_is_directory=True
    )
    upsert_plugin_manifest("demo@github", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "healthy"
    assert "plugin doctor demo@github" in format_plugin_doctor_report(report)


def test_plugin_doctor_reads_v3_namespace_declaration(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "github" / "v3_demo"
    plugin_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'v3_demo'\n"
        "version = '1.0.0'\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    _write_static_manifest(
        plugin_root,
        name="v3_demo",
        mcp_names=("fitbit", "steam"),
    )
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )
    upsert_plugin_manifest("v3_demo@github", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="v3_demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "healthy"
    assert _check(report, "skills")["detail"].startswith("roots=0")
    assert _check(report, "mcp")["detail"] == (
        "declared_servers=2 names=['fitbit', 'steam']"
    )


def test_plugin_doctor_uses_static_custom_entrypoint(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "github" / "static_demo"
    plugin_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    plugin_root.mkdir(parents=True)
    (plugin_root / "entry.py").write_text(
        "api_version = 3\n"
        "name = 'static_demo'\n"
        "version = '1.0.0'\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'static_demo'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'entry.py'\n",
        encoding="utf-8",
    )
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )
    upsert_plugin_manifest(
        "static_demo@github",
        enabled=True,
        plugins_home=plugins_home,
    )

    report = run_plugin_doctor(
        plugin_id="static_demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "healthy"
    assert "entry.py" in format_plugin_doctor_report(report)


def test_plugin_doctor_custom_entrypoint_uses_its_relative_import_root(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "github" / "nested_demo"
    plugin_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    (plugin_root / "src").mkdir(parents=True)
    (plugin_root / "src" / "constants.py").write_text(
        "VERSION = '1.0.0'\n",
        encoding="utf-8",
    )
    (plugin_root / "src" / "entry.py").write_text(
        "from .constants import VERSION\n"
        "api_version = 3\n"
        "name = 'nested_demo'\n"
        "version = VERSION\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'nested_demo'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'src/entry.py'\n",
        encoding="utf-8",
    )
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )
    upsert_plugin_manifest(
        "nested_demo@github",
        enabled=True,
        plugins_home=plugins_home,
    )

    report = run_plugin_doctor(
        plugin_id="nested_demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "healthy"
    assert "src/entry.py" in format_plugin_doctor_report(report)


def test_plugin_doctor_reads_latest_artifact_candidate(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "local" / "demo"
    plugin_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'demo'\n"
        "version = '1.0.0'\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    _write_static_manifest(plugin_root, name="demo")
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(None),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )
    upsert_plugin_manifest("demo@local", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@local",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "degraded"
    assert str(plugin_root) in format_plugin_doctor_report(report)
    assert _check(report, "candidate")["status"] == "deferred"


def test_plugin_doctor_rejects_legacy_visible_version_without_pointer(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    plugin_root = plugins_home / "cache/github/demo/1.0.0"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text(
        "api_version = 3\nname = 'demo'\nversion = '1.0.0'\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    _write_static_manifest(plugin_root, name="demo")
    upsert_plugin_manifest("demo@github", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "broken"
    assert _check(report, "install")["detail"] == "未找到插件目录"


def test_plugin_doctor_defers_candidate_projection_until_promotion(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "local" / "demo"
    stable_root = _write_artifact_plugin(
        plugin_base,
        "1.0.0-aaaa",
        skills={"stable-skill": "stable\n"},
    )
    latest_root = _write_artifact_plugin(
        plugin_base,
        "2.0.0-bbbb",
        skills={"candidate-skill": "candidate\n"},
    )
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/2.0.0-bbbb"),
    )
    link = workspace / "skills" / "stable-skill"
    link.parent.mkdir(parents=True)
    link.symlink_to(stable_root / "skills" / "stable-skill", target_is_directory=True)
    upsert_plugin_manifest("demo@local", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@local",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    assert report["status"] == "degraded"
    assert _check(report, "skills")["status"] == "ok"
    assert _check(report, "candidate")["status"] == "deferred"
    assert str(latest_root) in _check(report, "candidate")["detail"]


def test_plugin_doctor_reports_misdirected_and_stale_stable_projection(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    workspace = tmp_path / "workspace"
    plugin_base = plugins_home / "cache" / "local" / "demo"
    old_root = _write_artifact_plugin(
        plugin_base,
        "1.0.0-aaaa",
        skills={"current-skill": "old\n", "removed-skill": "removed\n"},
    )
    stable_root = _write_artifact_plugin(
        plugin_base,
        "2.0.0-bbbb",
        skills={"current-skill": "current\n"},
    )
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/2.0.0-bbbb"),
        latest=ArtifactPointer(".artifacts/2.0.0-bbbb"),
    )
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "current-skill").symlink_to(
        old_root / "skills" / "current-skill",
        target_is_directory=True,
    )
    (skills_dir / "removed-skill").symlink_to(
        old_root / "skills" / "removed-skill",
        target_is_directory=True,
    )
    upsert_plugin_manifest("demo@local", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@local",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=workspace,
    )

    skills = _check(report, "skills")
    assert report["status"] == "degraded"
    assert skills["status"] == "warn"
    assert "misdirected=['current-skill']" in skills["detail"]
    assert "stale=['removed-skill']" in skills["detail"]
    assert stable_root != old_root


def test_plugin_doctor_reports_broken_declaration(tmp_path: Path) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    plugin_base = plugins_home / "cache" / "github" / "demo"
    plugin_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.py").write_text("class X: pass\n", encoding="utf-8")
    _write_static_manifest(plugin_root, name="demo")
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )
    upsert_plugin_manifest("demo@github", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="demo@github",
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "broken"


@pytest.mark.parametrize("plugin_id", ["wake", "openai-compatible", "opencode-go"])
def test_plugin_doctor_finds_builtin_plugin(
    tmp_path: Path,
    plugin_id: str,
) -> None:
    plugins_home = tmp_path / ".akashic-plugin"
    upsert_plugin_manifest(plugin_id, enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id=plugin_id,
        config_path=str(_init_config(tmp_path)),
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "healthy"


def test_builtin_doctor_uses_one_declared_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_doctor = tmp_path / "repo" / "agent" / "plugins" / "doctor.py"
    builtin_root = tmp_path / "repo" / "plugins"
    monkeypatch.setattr(plugin_doctor, "__file__", str(fake_doctor))
    for folder, declared_name in (("wake", "other"), ("custom", "wake")):
        _write_builtin_plugin(builtin_root, folder, declared_name)
    plugins_home = tmp_path / ".akashic-plugin"
    upsert_plugin_manifest("wake", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="wake",
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "healthy"
    assert str(builtin_root / "custom") in _check(report, "install")["detail"]

    duplicate = builtin_root / "duplicate"
    duplicate.mkdir()
    (duplicate / "plugin.py").write_text("", encoding="utf-8")
    _write_static_manifest(duplicate, name="wake")
    duplicate_report = run_plugin_doctor(
        plugin_id="wake",
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert duplicate_report["status"] == "broken"
    assert (
        "多个内置插件声明了相同 name" in _check(duplicate_report, "install")["detail"]
    )


def test_builtin_doctor_reports_any_invalid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_doctor = tmp_path / "repo" / "agent" / "plugins" / "doctor.py"
    plugin_root = tmp_path / "repo" / "plugins" / "custom"
    plugin_root.mkdir(parents=True)
    (plugin_root / "akashic.plugin.toml").symlink_to("missing.toml")
    monkeypatch.setattr(plugin_doctor, "__file__", str(fake_doctor))
    plugins_home = tmp_path / ".akashic-plugin"
    upsert_plugin_manifest("wake", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="wake",
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "broken"
    assert "静态 manifest" in _check(report, "install")["detail"]


def test_builtin_doctor_ignores_symlink_plugin_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_doctor = tmp_path / "repo" / "agent" / "plugins" / "doctor.py"
    builtin_root = tmp_path / "repo" / "plugins"
    monkeypatch.setattr(plugin_doctor, "__file__", str(fake_doctor))
    legacy_root = _write_builtin_plugin(
        builtin_root,
        "wake",
        "wake",
        static=False,
    )
    symlink_target = _write_builtin_plugin(tmp_path, "outside", "shadow")
    (builtin_root / "custom").symlink_to(symlink_target, target_is_directory=True)
    plugins_home = tmp_path / ".akashic-plugin"
    upsert_plugin_manifest("wake", enabled=True, plugins_home=plugins_home)

    report = run_plugin_doctor(
        plugin_id="wake",
        plugins_home=plugins_home,
        workspace=tmp_path / "workspace",
    )

    assert report["status"] == "healthy"
    assert str(legacy_root) in _check(report, "install")["detail"]
