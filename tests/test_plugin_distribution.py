"""发布制品只能含选定宿主路径与各插件自己的源码。"""
import io
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any

import pytest
import toml
import yaml

from agent.plugin_composition import FiberState, ServiceKey
from agent.plugin_composition.config_input import load_config, save_config
from agent.plugins.install import (
    finalize_uninstall_plugin,
    install_git_plugin,
    set_installed_plugin_enabled,
)
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.manager import PluginManager
from agent.plugins.selection import PluginSelection, SelectionConflictError, SelectionWriteError
from agent.plugins.artifacts import ArtifactPointer, read_pointers
from agent.plugin_composition.archive import PluginArchive
from agent.plugins.input_preparation import prepare_plugin_input
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.static_manifest import PluginSourceCompileError, load_static_plugin_manifest
from bootstrap.workspace_lock import WorkspaceMaintenanceLock
from bus.event_bus import EventBus
import scripts.build_host_runtime_release as host_runtime_release
from scripts.build_host_runtime_release import _create_context
from scripts.build_plugin_distribution import (
    _append_tree,
    _build_web_assets,
    _bundle_plugin,
    build,
)
from scripts.install_plugin_distribution import (
    _preflight_bundle,
    _write_receipt,
    ensure_profile,
    extract_core,
    install_profile,
    verify_distribution,
    adopt_bundled_distribution,
    upgrade_bundled_distribution,
    main as distribution_main,
)
import scripts.install_plugin_distribution as distribution_installer
from agent.plugins.manifest import workspace_plugin_data_dir
from scripts.rollback_plugin_install import rollback_plugin_install


def test_default_profile_installs_akashic_sender() -> None:
    """The default Akashic channel must include its matching delivery sender."""

    profile = json.loads(
        Path("docker/host-runtime/profiles/default.json").read_text(encoding="utf-8")
    )
    plugins = {item["name"]: item for item in profile["plugins"]}

    assert plugins["akashic_sender"]["depends_on"] == ["delivery"]


def test_web_build_includes_plugin_ui_assets(tmp_path, monkeypatch) -> None:
    """发行 Web 构建必须生成插件 UI，不能只构建两个宿主页面。"""

    source = tmp_path / "source"
    for path in (
        "frontend/chat/vite.config.ts",
        "frontend/dashboard/vite.config.ts",
    ):
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("export default {}\n", encoding="utf-8")
    (source / "package.json").write_text("{}\n", encoding="utf-8")
    (source / "package-lock.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
        "commit", "-m", "source",
    ], check=True, capture_output=True)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, env):
        commands.append(command)
        if command == ["npm", "run", "build:dashboard"]:
            (cwd / "static/dashboard").mkdir(parents=True)
            (cwd / "static/dashboard/index.html").write_text("dashboard")
        elif command == ["npm", "run", "build:chat"]:
            (cwd / "static/chat").mkdir(parents=True)
            (cwd / "static/chat/index.html").write_text("chat")
        elif command == ["npm", "run", "build:web-plugins"]:
            (cwd / "plugins/akasha").mkdir(parents=True)
            (cwd / "plugins/akasha/message_ui.js").write_text("new module")

    monkeypatch.setattr(
        "scripts.build_plugin_distribution._run_web_command", fake_run
    )
    temporary = tmp_path / "build"
    temporary.mkdir()
    assets, plugins, report = _build_web_assets(source, "HEAD", temporary)

    assert assets is not None
    assert plugins is not None
    assert (plugins / "akasha/message_ui.js").read_text() == "new module"
    assert commands[-1] == ["npm", "run", "build:web-plugins"]
    build_commands = report["build_commands"]
    assert isinstance(build_commands, list)
    assert build_commands[-1] == "npm run build:web-plugins"


def test_plugin_bundle_uses_generated_ui_asset(tmp_path) -> None:
    """插件 bundle 必须覆盖固定提交中陈旧的生成资产。"""

    source = tmp_path / "source"
    plugin = source / "plugins/akasha"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        'api_version = 3\nname = "akasha"\nversion = "1"\nasync def apply(ctx): pass\n'
    )
    (plugin / "akashic.plugin.toml").write_text(
        'schema_version = 1\napi_version = 3\nname = "akasha"\n'
        'version = "1"\nentrypoint = "plugin.py"\n'
    )
    (plugin / "message_ui.js").write_text("stale module\n")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
        "commit", "-m", "source",
    ], check=True, capture_output=True)
    generated = tmp_path / "generated/plugins/akasha"
    generated.mkdir(parents=True)
    (generated / "message_ui.js").write_text("fresh module\n")
    output = tmp_path / "release"
    output.mkdir()

    row = _bundle_plugin(
        source,
        "HEAD",
        "2026-09-15T00:00:00+00:00",
        "plugins/akasha",
        output,
        set(),
        generated.parent,
    )
    installed = tmp_path / "installed"
    subprocess.run(
        ["git", "clone", str(output / row["file"]), str(installed)],
        check=True,
        capture_output=True,
    )
    assert (installed / "message_ui.js").read_text() == "fresh module\n"


def test_workload_controller_imports_core_from_distribution_source() -> None:
    """发行 workload controller 必须能导入 image 中的 Core 模块。"""

    compose = yaml.safe_load(
        Path("docker/host-runtime/compose.experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    environment = compose["services"]["workload-controller"]["environment"]

    assert environment["PYTHONPATH"] == "/opt/akashic/source"


def test_core_mounts_host_python_prefix_read_only() -> None:
    """Core 必须只读访问创建持久插件环境时固定的宿主 Python。"""

    compose = yaml.safe_load(
        Path("docker/host-runtime/compose.experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    volumes = compose["services"]["akashic-core"]["volumes"]

    assert {
        "type": "bind",
        "source": "${AKASHIC_HOST_PYTHON_PREFIX:?AKASHIC_HOST_PYTHON_PREFIX is required}",
        "target": "${AKASHIC_HOST_PYTHON_PREFIX:?AKASHIC_HOST_PYTHON_PREFIX is required}",
        "read_only": True,
    } in volumes


def test_distribution_installs_isolated_git_sources_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("one", "two", "unused"):
        root = source / "plugins" / name
        root.mkdir(parents=True)
        (root / "plugin.py").write_text(f'api_version = 3\nname = "{name}"\nversion = "1"\ndef apply(ctx): pass\n')
    (source / "main.py").write_text('print("core")\n')
    legacy_memory = source / "memory2"
    legacy_memory.mkdir()
    (legacy_memory / "embedder.py").write_text("legacy memory must stay external\n")
    (source / "config.example.toml").write_text("[runtime]\nworkspace = \"workspace\"\n")
    (source / "private.txt").write_text("must not ship")
    runtime = source / "docker" / "host-runtime"
    profile = runtime / "profiles"
    profile.mkdir(parents=True)
    (runtime / "Dockerfile.distribution").write_text("FROM scratch\n")
    (runtime / "distribution-entrypoint.sh").write_text("#!/bin/sh\n")
    (profile / "default.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "fixture",
        "marketplace": "distribution",
        "initialization": {
            "plugin_configs": [{
                "owner": "one",
                "config": {
                    "prompt_sources": {"fixture": "one@distribution"},
                    "summary_source": ["summary", "one@distribution"],
                    "unsafe": "x\"\n[unexpected]\nvalue = \"bad\"",
                },
            }],
        },
        "plugins": [
            {"name": "one", "depends_on": [], "reason": "fixture capability"},
            {"name": "two", "depends_on": ["one"], "reason": "fixture consumer"},
        ],
    }) + "\n")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgSign=false", "commit", "-m", "source"], check=True, capture_output=True)
    output = tmp_path / "release"
    report = build(source, "HEAD", output)
    plugins = report["plugins"]
    wiring = report["runtime_wiring"]
    assert isinstance(plugins, list)
    assert isinstance(wiring, list)
    with tarfile.open(fileobj=io.BytesIO((output / "core.tar").read_bytes())) as archive:
        assert set(archive.getnames()) == {
            "config.example.toml", "main.py", "runtime-dependencies.json",
            "docker", "docker/host-runtime", "docker/host-runtime/profiles",
            "docker/host-runtime/profiles/default.json",
            "docker/host-runtime/Dockerfile.distribution",
            "docker/host-runtime/distribution-entrypoint.sh",
        }
        assert not any(name == "plugins" or name.startswith("plugins/") for name in archive.getnames())
        assert not any(name == "memory2" or name.startswith("memory2/") for name in archive.getnames())
    assert {row["name"] for row in plugins} == {"one", "two", "unused"}
    one_row = next(row for row in plugins if row["name"] == "one")
    repository_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    _preflight_bundle(
        output / one_row["file"],
        row=one_row,
        source_commit=str(report["source_commit"]),
    )
    monkeypatch.chdir(repository_cwd)
    assert {row["path"] for row in wiring} == {
        "Dockerfile.distribution",
        "distribution-entrypoint.sh",
    }
    assert (output / "Dockerfile.distribution").is_file()
    dockerfile = Path("docker/host-runtime/Dockerfile.distribution").read_text()
    assert "COPY core.tar /opt/akashic/distribution/core.tar" in dockerfile
    assert (
        "COPY Dockerfile.distribution /opt/akashic/distribution/Dockerfile.distribution"
        in dockerfile
    )
    assert (
        "COPY distribution-entrypoint.sh /opt/akashic/distribution/distribution-entrypoint.sh"
        in dockerfile
    )
    assert "--ensure-profile" in (
        Path("docker/host-runtime/distribution-entrypoint.sh").read_text()
    )
    verified = verify_distribution(output)
    core_root = extract_core(output, tmp_path / "core", report=verified)
    assert (core_root / "config.example.toml").is_file()
    assert not (core_root / "plugins").exists()
    config = tmp_path / "profile-config.toml"
    config.write_text("[runtime]\nworkspace = \"profile-workspace\"\n", encoding="utf-8")
    profile_receipt = install_profile(
        output,
        output / "profiles/default.json",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
        config_path=tmp_path / "profile-config.toml",
    )
    assert [item["name"] for item in profile_receipt["installed"]] == ["one", "two"]
    assert (tmp_path / "profile-workspace/migrations.sqlite3").is_file()
    assert config.read_text() == "[runtime]\nworkspace = \"profile-workspace\"\n"
    context_config = tmp_path / "profile-workspace/plugin-data/one-distribution"
    assert load_config(context_config)[0] == {
        "prompt_sources": {"fixture": "one@distribution"},
        "summary_source": ["summary", "one@distribution"],
        "unsafe": "x\"\n[unexpected]\nvalue = \"bad\"",
    }
    assert not (tmp_path / "profile-home" / "cache" / "distribution" / "unused").exists()
    receipt_path = tmp_path / "profile-workspace/runtime/distribution-install.json"
    _write_receipt(receipt_path, profile_receipt)
    save_config(context_config, {"custom": "keep"})
    set_installed_plugin_enabled(
        "one@distribution", enabled=False, plugins_home=tmp_path / "profile-home"
    )
    existing = ensure_profile(
        output,
        output / "profiles/default.json",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
        config_path=config,
        receipt_path=receipt_path,
    )
    assert existing["status"] == "existing"
    assert existing["installed"][0]["source_revision"] == profile_receipt["installed"][0]["source_revision"]
    assert load_config(context_config)[0] == {"custom": "keep"}

    external = tmp_path / "external-one"
    external.mkdir()
    (external / "plugin.py").write_text(
        'api_version = 3\nname = "one"\nversion = "2"\ndef apply(ctx): pass\n'
    )
    subprocess.run(["git", "init", str(external)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(external), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
        "commit", "-m", "replacement",
    ], check=True, capture_output=True)
    replacement = install_git_plugin(
        workspace=tmp_path / "profile-workspace",
        source=str(external),
        marketplace="distribution",
        plugins_home=tmp_path / "profile-home",
    )
    assert replacement.source_revision != profile_receipt["installed"][0]["source_revision"]
    set_installed_plugin_enabled(
        "one@distribution", enabled=False, plugins_home=tmp_path / "profile-home"
    )
    replaced = ensure_profile(
        output,
        output / "profiles/default.json",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
        config_path=config,
        receipt_path=receipt_path,
    )
    assert replaced["status"] == "existing"
    assert load_config(context_config)[0] == {"custom": "keep"}
    cli_restart = subprocess.run(
        [
            sys.executable,
            "scripts/install_plugin_distribution.py",
            "--distribution",
            str(output),
            "--profile",
            str(output / "profiles/default.json"),
            "--workspace",
            str(tmp_path / "profile-workspace"),
            "--plugins-home",
            str(tmp_path / "profile-home"),
            "--config",
            str(config),
            "--ensure-profile",
            "--receipt",
            str(receipt_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(cli_restart.stdout)["status"] == "existing"

    # A new Core/profile generation may remove the original providers and
    # install a different-name replacement.  The old receipt remains a
    # historical record; ensure_profile must validate the current composition
    # without reinstalling or rewriting that record.
    receipt_before_upgrade = receipt_path.read_bytes()
    replacement_source = source / "plugins" / "replacement"
    replacement_source.mkdir(parents=True)
    (replacement_source / "plugin.py").write_text(
        'api_version = 3\nname = "replacement"\nversion = "1"\ndef apply(ctx): pass\n'
    )
    (source / "main.py").write_text('print("core-v2")\n')
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
        "commit", "-m", "core and provider upgrade",
    ], check=True, capture_output=True)
    output_v2 = tmp_path / "release-v2"
    report_v2 = build(source, "HEAD", output_v2)
    plugins_v2 = report_v2["plugins"]
    assert isinstance(plugins_v2, list)
    assert report_v2["source_commit"] != report["source_commit"]
    replacement_row = next(
        row for row in plugins_v2 if row["name"] == "replacement"
    )
    installed_alias = install_git_plugin(
        workspace=tmp_path / "profile-workspace",
        source=str(output_v2 / replacement_row["file"]),
        marketplace="distribution",
        ref_name=replacement_row["source_revision"],
        plugins_home=tmp_path / "profile-home",
    )
    replacement_provenance = (
        installed_alias.installed_path / ".akashic-source.json"
    ).read_bytes()
    finalize_uninstall_plugin(
        "one@distribution",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
    )
    finalize_uninstall_plugin(
        "two@distribution",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
    )
    assert load_plugin_manifest(tmp_path / "profile-home") == {
        "replacement@distribution": True
    }
    upgraded = ensure_profile(
        output_v2,
        output / "profiles/default.json",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
        config_path=config,
        receipt_path=receipt_path,
    )
    assert upgraded["status"] == "existing"
    assert upgraded["profile"] == "fixture"
    assert receipt_path.read_bytes() == receipt_before_upgrade
    assert load_plugin_manifest(tmp_path / "profile-home") == {
        "replacement@distribution": True
    }
    assert replacement_provenance == (
        installed_alias.installed_path / ".akashic-source.json"
    ).read_bytes()
    assert not (tmp_path / "profile-home" / "cache" / "distribution" / "one").exists()
    assert not (tmp_path / "profile-home" / "cache" / "distribution" / "two").exists()

    invalid_receipt = tmp_path / "invalid-receipt.json"
    invalid_receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        ensure_profile(
            output,
            output / "profiles/default.json",
            workspace=tmp_path / "profile-workspace",
            plugins_home=tmp_path / "profile-home",
            config_path=config,
            receipt_path=invalid_receipt,
        )
    for row in plugins:
        installed = install_git_plugin(workspace=tmp_path / "workspace", plugins_home=tmp_path / "home",
            source=str(output / row["file"]), marketplace="distribution")
        assert installed.source_revision == row["source_revision"]
        assert not (installed.installed_path / "plugins").exists()
        assert not (installed.installed_path / "private.txt").exists()
        provenance = json.loads((installed.installed_path / ".akashic-source.json").read_text())
        assert provenance == {"commit": report["source_commit"], "path": row["source_path"]}
    with pytest.raises(FileExistsError):
        build(source, "HEAD", output)


@pytest.mark.asyncio
async def test_new_distribution_keeps_selected_archive_until_public_install(tmp_path):
    """A new image is inert until the public install selects its exact bundle."""
    source = tmp_path / "source"
    target = source / "plugins/target/plugin.py"
    peer = source / "plugins/peer/plugin.py"
    target.parent.mkdir(parents=True)
    peer.parent.mkdir(parents=True)
    peer.write_text(
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\nname = 'peer'\nversion = '1'\n"
        "PEER = ServiceKey('fixture.peer')\n"
        "async def apply(ctx):\n"
        "    state = {'started': 0, 'closed': 0}\n"
        "    async def start():\n"
        "        state['started'] += 1\n"
        "        async def close():\n"
        "            state['closed'] += 1\n"
        "        return close\n"
        "    await ctx.effect(start, label='peer-effect')\n"
        "    await ctx.provide(PEER, state)\n",
        encoding="utf-8",
    )
    profile = source / "docker/host-runtime/profiles/default.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({
        "schema_version": 1,
        "name": "fixture",
        "marketplace": "release",
        "initialization": {"plugin_configs": []},
        "plugins": [
            {"name": "peer", "depends_on": [], "reason": "retained service"},
            {"name": "target", "depends_on": [], "reason": "versioned input"},
        ],
    }), encoding="utf-8")
    (profile.parent.parent / "Dockerfile.distribution").write_text("FROM scratch\n")
    (profile.parent.parent / "distribution-entrypoint.sh").write_text("#!/bin/sh\n")
    (source / "config.example.toml").write_text("[runtime]\n", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text("[runtime]\n", encoding="utf-8")

    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)

    def make_distribution(version: str) -> tuple[Path, dict[str, object]]:
        target.write_text(
            "api_version = 3\nname = 'target'\n"
            f"version = {version!r}\nasync def apply(ctx):\n    return None\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
        subprocess.run([
            "git", "-C", str(source), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
            "-c", "core.hooksPath=/dev/null", "commit", "-m", f"version {version}",
        ], check=True, capture_output=True)
        output = tmp_path / f"distribution-{version}"
        return output, build(source, "HEAD", output)

    distribution_a, report_a = make_distribution("1")
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    receipt = install_profile(
        distribution_a, distribution_a / "profiles/default.json",
        workspace=workspace, plugins_home=home, config_path=config,
    )
    receipt_path = workspace / "runtime/distribution-install.json"
    _write_receipt(receipt_path, receipt)
    receipt_bytes = receipt_path.read_bytes()
    PluginSelection(workspace).initialize()
    data_file = workspace / "plugin-data/target-release/user.txt"
    data_file.write_text("preserve user data", encoding="utf-8")

    def manager() -> PluginManager:
        return PluginManager(
            [], event_bus=EventBus(), workspace=workspace,
            installed_cache_root=home / "cache",
        )

    first = manager()
    try:
        await first.load_all()
        selected_a = first._selection.read()
        target_a = first.generation("target@release")
        assert selected_a is not None and target_a is not None
        assert target_a.fiber is not None and target_a.fiber.state is FiberState.ACTIVE
        assert target_a.instance.version == "1"
        archive_a = target_a.archive_ref
        assert archive_a is not None
        descriptor_a = first._archive.read_descriptor(archive_a)
        assert descriptor_a["source_revision"] == target_a.source_revision
        assert archive_a in first._selection_components(selected_a)
    finally:
        await first.terminate_all()

    distribution_b, report_b = make_distribution("2")
    bundle_rows = report_b["plugins"]
    assert isinstance(bundle_rows, list)
    bundle_b = next(
        row for row in bundle_rows
        if isinstance(row, dict) and row.get("name") == "target"
    )
    existing = ensure_profile(
        distribution_b, distribution_b / "profiles/default.json",
        workspace=workspace, plugins_home=home, config_path=config,
        receipt_path=receipt_path,
    )
    assert existing["status"] == "existing"
    assert existing["distribution_source_commit"] == report_a["source_commit"]
    assert receipt_path.read_bytes() == receipt_bytes
    assert PluginSelection(workspace).read() == selected_a

    second = manager()
    try:
        await second.load_all()
        still_a = second.generation("target@release")
        assert still_a is not None and still_a.archive_ref == archive_a
        assert still_a.instance.version == "1"
        assert second._selection.read() == selected_a
        root = second.live_root
        peer = second.generation("peer@release")
        assert root is not None and peer is not None and peer.fiber is not None
        peer_fiber = peer.fiber
        peer_effects = tuple(peer_fiber.effects)
        peer_context = peer_fiber.context
        async with peer_context.runtime_scope():
            peer_state = peer_context.require(ServiceKey("fixture.peer"))
        assert peer_state == {"started": 1, "closed": 0}

        # This is the online public install path. It does not establish an
        # offline distribution upgrade command or update the whole fleet.
        accepted = await second.install(
            source=str(distribution_b / bundle_b["file"]), marketplace="release",
            ref_name=str(bundle_b["source_revision"]), sparse_paths=[],
            update_id="explicit-target-b",
        )
        assert accepted.state == "accepted"
        operation = second._operation
        assert operation is not None
        await operation.task
        active = second.read_update("explicit-target-b")
        target_b = second.generation("target@release")
        assert active.state == "active" and target_b is not None
        assert target_b is not still_a and target_b.fiber is not None
        assert target_b.fiber.state is FiberState.ACTIVE
        assert target_b.instance.version == "2"
        installed_revision = subprocess.run(
            ["git", "-C", str(target_b.plugin_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert installed_revision == bundle_b["source_revision"]
        assert target_b.archive_ref == active.input_ref == active.archive_ref
        selected_b = second._selection.read()
        assert selected_b is not None
        assert target_b.archive_ref in second._selection_components(selected_b)
        assert selected_b != selected_a
        assert second._archive.read_descriptor(selected_b)["previous"] == selected_a
        descriptor_b = second._archive.read_descriptor(target_b.archive_ref)
        assert descriptor_b["source_revision"] == target_b.source_revision
        assert descriptor_b["code"] != descriptor_a["code"]
        provenance = json.loads((target_b.plugin_dir / ".akashic-source.json").read_text())
        assert provenance == {"commit": report_b["source_commit"], "path": "plugins/target"}
        status = second.plugin_status()
        status_rows = status["plugins"]
        assert isinstance(status_rows, list)
        target_status = next(
            item for item in status_rows
            if isinstance(item, dict) and item.get("plugin_id") == "target@release"
        )
        assert target_status["selected_ref"] == target_b.archive_ref
        assert target_status["archive_ref"] == target_b.archive_ref
        assert target_status["fiber_state"] == "active"
        assert second.live_root is root
        assert second.generation("peer@release") is peer
        assert peer.fiber is peer_fiber and tuple(peer_fiber.effects) == peer_effects
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("fixture.peer")) is peer_state
        assert peer_state == {"started": 1, "closed": 0}
        archive_b = target_b.archive_ref
    finally:
        await second.terminate_all()

    assert peer_state == {"started": 1, "closed": 1}
    assert receipt_path.read_bytes() == receipt_bytes
    assert data_file.read_text(encoding="utf-8") == "preserve user data"
    assert (workspace / "runtime/plugin-archives" / f"{archive_a}.json").is_file()

    third = manager()
    try:
        await third.load_all()
        restored = third.generation("target@release")
        assert restored is not None and restored.fiber is not None
        assert restored.fiber.state is FiberState.ACTIVE
        assert restored.instance.version == "2"
        assert restored.archive_ref == archive_b
        assert third._selection.read() == selected_b
        assert receipt_path.read_bytes() == receipt_bytes
        assert data_file.read_text(encoding="utf-8") == "preserve user data"
    finally:
        await third.terminate_all()


def _offline_case(tmp_path: Path) -> dict[str, Any]:
    """Build one real two-plugin distribution and its first selected Root."""
    source = tmp_path / "source"
    for name in ("peer", "target"):
        (source / "plugins" / name).mkdir(parents=True)
    profile = source / "docker/host-runtime/profiles/default.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({
        "schema_version": 1, "name": "fixture", "marketplace": "release",
        "initialization": {"plugin_configs": []},
        "plugins": [
            {"name": "peer", "depends_on": [], "reason": "retained peer"},
            {"name": "target", "depends_on": [], "reason": "versioned target"},
        ],
    }))
    (profile.parent.parent / "Dockerfile.distribution").write_text("FROM scratch\n")
    (profile.parent.parent / "distribution-entrypoint.sh").write_text("#!/bin/sh\n")
    (source / "config.example.toml").write_text("[runtime]\n")
    config = tmp_path / "config.toml"
    config.write_text("[runtime]\n")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    case: dict[str, Any] = {"source": source, "workspace": tmp_path / "workspace", "home": tmp_path / "home",
            "config": config, "receipt": tmp_path / "workspace/runtime/distribution-install.json"}
    first, first_report = _offline_release(case, "1")
    case["release_commit"] = first_report["source_commit"]
    installed = install_profile(first, first / "profiles/default.json", workspace=case["workspace"],
                                plugins_home=case["home"], config_path=config)
    _write_receipt(case["receipt"], installed)
    selection = PluginSelection(case["workspace"])
    selection.initialize()
    archive = PluginArchive(case["workspace"] / "runtime/plugin-archives")
    refs = []
    for item in installed["installed"]:
        root = Path(item["installed_path"])
        identity = load_static_plugin_manifest(root)
        prepared = prepare_plugin_input({
            "name": item["name"], "marketplace": "release", "plugin_root": str(root),
            "module_path": str(root / "plugin.py"), "manifest_digest": identity.identity_digest,
            "source_type": "installed",
        }, workspace=case["workspace"], archive=archive)
        refs.append(prepared.archive_ref)
    case["root"] = selection.commit(tuple(refs), expected_ref=None)
    return case


def _offline_release(case: dict[str, Any], version: str) -> tuple[Path, dict[str, object]]:
    source = case["source"]
    for name in ("peer", "target"):
        content = (
            "api_version = 3\n" + f"name = {name!r}\nversion = {version!r}\n"
            "async def apply(ctx):\n"
            + ("    assert ctx.config['mode'] == 'v2'\n"
               if case.get("assert_target_config") and name == "target" and version == "2"
               else "    return None\n")
        )
        compile(content, f"{name}/plugin.py", "exec")
        (source / "plugins" / name / "plugin.py").write_text(content)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgSign=false", "-c", "core.hooksPath=/dev/null",
        "commit", "-m", f"version {version}",
    ], check=True, capture_output=True)
    output = source.parent / f"distribution-{version}"
    return output, build(source, "HEAD", output)


def _add_target_data_migration(source: Path, *, fail_once: bool = False) -> None:
    """Make B carry a real plugin-owned Yoyo data and config change."""

    plugin = source / "plugins/target"
    migrations = plugin / "target_migrations"
    migrations.mkdir()
    (migrations / "__init__.py").write_text("\n")
    step = migrations / "target_data_v2.py"
    step.write_text(
        "from yoyo import step\n"
        "from agent.migrations.context import current_migration_context\n"
        "import json\n"
        "__depends__ = set()\n"
        "__transactional__ = False\n"
        "def apply(connection):\n"
        "    data = current_migration_context().bundle_data_roots['target_upgrade']\n"
        + ("    marker = current_migration_context().workspace / 'migration-attempted'\n"
           "    if not marker.exists():\n"
           "        marker.write_text('attempted')\n"
           "        raise RuntimeError('injected migration failure')\n"
           if fail_once else "")
        +
        "    (data / 'data.txt').write_text('v2')\n"
        "    config = data / 'config.input.json'\n"
        "    value = json.loads(config.read_text())\n"
        "    assert value['config'][1]['mode'] == 'v1'\n"
        "    value['config'][1]['mode'] = 'v2'\n"
        "    config.write_text(json.dumps(value))\n"
        "steps = [step(apply)]\n"
    )
    files = [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (migrations / "__init__.py", step)
    ]
    (plugin / "migration.catalog.toml").write_text(toml.dumps({
        "schema_version": 1, "bundle_id": "target_upgrade", "version": "2",
        "migration_root": "target_migrations", "package_name": "target_migrations",
        "files": files,
        "migrations": [{"id": "target_data_v2", "path": step.name,
                        "depends": [], "transactional": False,
                        "sha256": hashlib.sha256(step.read_bytes()).hexdigest()}],
    }))


@pytest.mark.asyncio
async def test_release_upgrade_migrates_before_freezing_target_input(tmp_path, monkeypatch):
    """The target artifact must first see B data/config and then own the new Root."""

    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    data = workspace_plugin_data_dir(case["workspace"], "target", "release")
    data.mkdir(parents=True, exist_ok=True)
    (data / "data.txt").write_text("v1")
    save_config(data, {"mode": "v1"})
    _add_target_data_migration(case["source"])
    case["assert_target_config"] = True
    release_b, report_b = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    monkeypatch.setattr(distribution_installer, "_SOURCE_ROOT", repo)
    backup = tmp_path / "recovery"

    result = upgrade_bundled_distribution(
        distribution=release_b, profile=release_b / "profiles/default.json",
        workspace=case["workspace"], plugins_home=case["home"],
        config_path=case["config"], receipt_path=case["receipt"],
        backup_dir=backup, expected_root_ref=case["root"],
        previous_source_commit=case["release_commit"],
    )

    assert result["migration_ids"] == ["target_data_v2"]
    assert result["new_root_ref"] != case["root"]
    assert (data / "data.txt").read_text() == "v2"
    assert load_config(data)[0] == {"mode": "v2"}
    assert (backup / "state/workspace" / data.relative_to(case["workspace"]) / "data.txt").read_text() == "v1"
    assert PluginSelection(case["workspace"]).read() == result["new_root_ref"]
    assert (backup / "manifest.json").is_file()
    owner = PluginManager([], event_bus=EventBus(), workspace=case["workspace"],
                          installed_cache_root=case["home"] / "cache")
    try:
        await owner.load_all()
        target = owner.generation("target@release")
        assert target is not None and target.fiber is not None
        assert target.fiber.state is FiberState.ACTIVE
        assert target.instance.version == "2"
        assert owner.plugin_status()["selection_ref"] == result["new_root_ref"]
    finally:
        await owner.terminate_all()
    repeat = upgrade_bundled_distribution(
        distribution=release_b, profile=release_b / "profiles/default.json",
        workspace=case["workspace"], plugins_home=case["home"],
        config_path=case["config"], receipt_path=case["receipt"],
        backup_dir=tmp_path / "repeat-recovery", expected_root_ref=result["new_root_ref"],
        previous_source_commit=str(report_b["source_commit"]),
    )
    assert repeat["migration_ids"] == []
    assert repeat["new_root_ref"] == result["new_root_ref"]


def test_release_upgrade_failed_migration_keeps_old_selection_and_retries(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    data = workspace_plugin_data_dir(case["workspace"], "target", "release")
    data.mkdir(parents=True, exist_ok=True)
    (data / "data.txt").write_text("v1")
    save_config(data, {"mode": "v1"})
    _add_target_data_migration(case["source"], fail_once=True)
    release_b, _ = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    monkeypatch.setattr(distribution_installer, "_SOURCE_ROOT", repo)
    args: dict[str, Any] = dict(distribution=release_b, profile=release_b / "profiles/default.json",
                workspace=case["workspace"], plugins_home=case["home"],
                config_path=case["config"], receipt_path=case["receipt"],
                expected_root_ref=case["root"],
                previous_source_commit=case["release_commit"])

    with pytest.raises(RuntimeError, match="injected migration failure"):
        upgrade_bundled_distribution(**args, backup_dir=tmp_path / "failed-recovery")
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert (data / "data.txt").read_text() == "v1"
    assert (tmp_path / "failed-recovery/manifest.json").is_file()
    assert (tmp_path / "failed-recovery/state/workspace" / data.relative_to(case["workspace"]) / "data.txt").read_text() == "v1"

    retried = upgrade_bundled_distribution(**args, backup_dir=tmp_path / "retry-recovery")
    assert retried["migration_ids"] == ["target_data_v2"]
    assert PluginSelection(case["workspace"]).read() == retried["new_root_ref"]


def test_release_upgrade_preparation_failure_keeps_migrated_data_and_old_root(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    data = workspace_plugin_data_dir(case["workspace"], "target", "release")
    data.mkdir(parents=True, exist_ok=True)
    (data / "data.txt").write_text("v1")
    save_config(data, {"mode": "v1"})
    _add_target_data_migration(case["source"])
    release_b, _ = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    monkeypatch.setattr(distribution_installer, "_SOURCE_ROOT", repo)
    original = distribution_installer.prepare_plugin_input

    def fail_target(mod, *, workspace, archive):
        if mod["name"] == "target":
            raise SyntaxError("injected prepare failure")
        return original(mod, workspace=workspace, archive=archive)

    monkeypatch.setattr(distribution_installer, "prepare_plugin_input", fail_target)
    args: dict[str, Any] = dict(distribution=release_b, profile=release_b / "profiles/default.json",
                workspace=case["workspace"], plugins_home=case["home"],
                config_path=case["config"], receipt_path=case["receipt"],
                expected_root_ref=case["root"],
                previous_source_commit=case["release_commit"])
    with pytest.raises(SyntaxError, match="injected prepare failure"):
        upgrade_bundled_distribution(**args, backup_dir=tmp_path / "failed-recovery")
    assert (data / "data.txt").read_text() == "v2"
    assert load_config(data)[0] == {"mode": "v2"}
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert (tmp_path / "failed-recovery/state/workspace" / data.relative_to(case["workspace"]) / "data.txt").read_text() == "v1"
    monkeypatch.setattr(distribution_installer, "prepare_plugin_input", original)
    retried = upgrade_bundled_distribution(**args, backup_dir=tmp_path / "retry-recovery")
    assert retried["migration_ids"] == []
    assert PluginSelection(case["workspace"]).read() == retried["new_root_ref"]


def test_release_upgrade_public_cli_runs_real_migration_and_selection(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    data = workspace_plugin_data_dir(case["workspace"], "target", "release")
    data.mkdir(parents=True, exist_ok=True)
    (data / "data.txt").write_text("v1")
    save_config(data, {"mode": "v1"})
    _add_target_data_migration(case["source"])
    release_b, _ = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    command = [sys.executable, "-B", "-c",
               "import sys; from pathlib import Path; import scripts.install_plugin_distribution as m; "
               "m._SOURCE_ROOT = Path(sys.argv[1]); sys.argv = ['install_plugin_distribution.py', *sys.argv[2:]]; m.main()",
               str(repo), "--distribution", str(release_b),
               "--profile", str(release_b / "profiles/default.json"),
               "--workspace", str(case["workspace"]),
               "--plugins-home", str(case["home"]),
               "--config", str(case["config"]), "--receipt", str(case["receipt"]),
               "--upgrade-bundled", "--expected-root-ref", case["root"],
               "--previous-source-commit", case["release_commit"],
               "--backup-dir", str(tmp_path / "cli-recovery")]
    result = subprocess.run(command, cwd=Path(__file__).parents[1],
                            check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    assert receipt["migration_ids"] == ["target_data_v2"]
    assert receipt["new_root_ref"] == PluginSelection(case["workspace"]).read()
    assert load_config(data)[0] == {"mode": "v2"}
    assert (tmp_path / "cli-recovery/manifest.json").is_file()


def test_release_upgrade_selection_conflict_after_migration_stays_recoverable(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    data = workspace_plugin_data_dir(case["workspace"], "target", "release")
    data.mkdir(parents=True, exist_ok=True)
    (data / "data.txt").write_text("v1")
    save_config(data, {"mode": "v1"})
    _add_target_data_migration(case["source"])
    release_b, _ = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    monkeypatch.setattr(distribution_installer, "_SOURCE_ROOT", repo)
    original = PluginSelection.commit

    def conflict(self, components, *, expected_ref):
        raise SelectionConflictError("injected CAS conflict")

    monkeypatch.setattr(PluginSelection, "commit", conflict)
    args: dict[str, Any] = dict(distribution=release_b, profile=release_b / "profiles/default.json",
                                workspace=case["workspace"], plugins_home=case["home"],
                                config_path=case["config"], receipt_path=case["receipt"],
                                expected_root_ref=case["root"],
                                previous_source_commit=case["release_commit"])
    with pytest.raises(SelectionConflictError, match="injected CAS conflict"):
        upgrade_bundled_distribution(**args, backup_dir=tmp_path / "failed-recovery")
    assert (data / "data.txt").read_text() == "v2"
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert (tmp_path / "failed-recovery/manifest.json").is_file()
    monkeypatch.setattr(PluginSelection, "commit", original)
    retried = upgrade_bundled_distribution(**args, backup_dir=tmp_path / "retry-recovery")
    assert retried["migration_ids"] == []
    assert PluginSelection(case["workspace"]).read() == retried["new_root_ref"]


def test_release_upgrade_retains_same_path_external_override(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    external = tmp_path / "external-target"
    external.mkdir()
    (external / "plugin.py").write_text(
        "api_version = 3\nname = 'target'\nversion = 'external'\n"
        "async def apply(ctx):\n    return None\n")
    (external / ".akashic-source.json").write_text(json.dumps({
        "commit": "f" * 40, "path": "plugins/target",
    }))
    subprocess.run(["git", "init", str(external)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
                    "-c", "core.hooksPath=/dev/null", "commit", "-m", "external"],
                   check=True, capture_output=True)
    installed = install_git_plugin(workspace=case["workspace"], plugins_home=case["home"],
                                   source=str(external), marketplace="release")
    identity = load_static_plugin_manifest(installed.installed_path)
    prepared = prepare_plugin_input({
        "name": "target", "marketplace": "release", "plugin_root": str(installed.installed_path),
        "module_path": str(installed.installed_path / "plugin.py"),
        "manifest_digest": identity.identity_digest, "source_type": "installed",
    }, workspace=case["workspace"], archive=PluginArchive(case["workspace"] / "runtime/plugin-archives"))
    selection = PluginSelection(case["workspace"])
    external_root = selection.commit(tuple(
        prepared.archive_ref if selection.archive.read_descriptor(ref)["plugin_id"] == "target@release" else ref
        for ref in _selection_refs(selection, case["root"])
    ), expected_ref=case["root"])
    release_b, _ = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    monkeypatch.setattr(distribution_installer, "_SOURCE_ROOT", repo)
    upgraded = upgrade_bundled_distribution(
        distribution=release_b, profile=release_b / "profiles/default.json",
        workspace=case["workspace"], plugins_home=case["home"],
        config_path=case["config"], receipt_path=case["receipt"],
        backup_dir=tmp_path / "recovery", expected_root_ref=external_root,
        previous_source_commit=case["release_commit"],
    )
    assert upgraded["status"] == "partial_selected_not_started"
    assert upgraded["adoption"]["skipped_external"] == ["target@release"]
    assert prepared.archive_ref in _selection_refs(selection, upgraded["new_root_ref"])


def test_external_plan_rejects_duplicate_and_unsafe_paths(tmp_path):
    plan = tmp_path / "plan.json"
    target = {"plugin_id": "outside@external", "bundle_relative_path": "bundles/outside.bundle",
              "bundle_sha256": "a" * 64, "target_commit": "b" * 40}
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": "c" * 64,
                                "targets": [target, target]}))
    with pytest.raises(ValueError, match="重复"):
        distribution_installer._external_plan(plan)
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": "c" * 64,
                                "targets": [{**target, "bundle_relative_path": "../outside.bundle"}]}))
    with pytest.raises(ValueError, match="路径"):
        distribution_installer._external_plan(plan)


def test_external_upgrade_combines_bundled_and_explicit_into_one_root(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    case = _offline_case(state)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "plugin.py"
    source.write_text("api_version = 3\nname = 'outside'\nversion = '1'\nasync def apply(ctx):\n    return None\n")
    subprocess.run(["git", "init", str(external)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgSign=false", "commit", "-m", "one"], check=True, capture_output=True)
    installed = install_git_plugin(workspace=case["workspace"], plugins_home=case["home"],
                                   source=str(external), marketplace="external")
    identity = load_static_plugin_manifest(installed.installed_path)
    selection = PluginSelection(case["workspace"])
    prepared = prepare_plugin_input({"name": "outside", "marketplace": "external",
                                     "plugin_root": str(installed.installed_path),
                                     "module_path": str(installed.installed_path / "plugin.py"),
                                     "manifest_digest": identity.identity_digest,
                                     "source_type": "installed"},
                                    workspace=case["workspace"], archive=selection.archive)
    old = selection.commit((*_selection_refs(selection, case["root"]), prepared.archive_ref),
                           expected_ref=case["root"])
    source.write_text("api_version = 3\nname = 'outside'\nversion = '2'\nasync def apply(ctx):\n    return None\n")
    subprocess.run(["git", "-C", str(external), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgSign=false", "commit", "-m", "two"], check=True, capture_output=True)
    commit = subprocess.run(["git", "-C", str(external), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    inputs = tmp_path / "inputs"
    (inputs / "bundles").mkdir(parents=True)
    bundle = inputs / "bundles/outside.bundle"
    subprocess.run(["git", "-C", str(external), "bundle", "create", str(bundle), "HEAD"],
                   check=True, capture_output=True)
    plan = inputs / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": old,
                                "targets": [{"plugin_id": "outside@external",
                                             "bundle_relative_path": "bundles/outside.bundle",
                                             "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                                             "target_commit": commit}]}))
    release_b, _ = _offline_release(case, "2")
    repo = tmp_path / "migration-host"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text("schema_version = 1\nmigrations = []\n")
    monkeypatch.setattr(distribution_installer, "_SOURCE_ROOT", repo)
    before = (case["workspace"] / "runtime/plugin-stable.json").read_bytes()
    preflight = upgrade_bundled_distribution(
        distribution=release_b, profile=release_b / "profiles/default.json",
        workspace=case["workspace"], plugins_home=case["home"],
        config_path=case["config"], receipt_path=case["receipt"],
        backup_dir=tmp_path / "preflight-recovery", expected_root_ref=old,
        previous_source_commit=case["release_commit"], external_plan=plan,
        external_inputs=inputs, preflight_only=True)
    assert preflight["status"] == "preflight_ok"
    assert (case["workspace"] / "runtime/plugin-stable.json").read_bytes() == before
    assert not (tmp_path / "preflight-recovery").exists()
    commits: list[tuple[str, ...]] = []
    original_commit = PluginSelection.commit

    def count_commit(self, components, *, expected_ref):
        commits.append(components)
        return original_commit(self, components, expected_ref=expected_ref)

    monkeypatch.setattr(PluginSelection, "commit", count_commit)
    result = upgrade_bundled_distribution(
        distribution=release_b, profile=release_b / "profiles/default.json",
        workspace=case["workspace"], plugins_home=case["home"],
        config_path=case["config"], receipt_path=case["receipt"],
        backup_dir=tmp_path / "recovery", expected_root_ref=old,
        previous_source_commit=case["release_commit"], external_plan=plan,
        external_inputs=inputs)
    assert len(commits) == 1
    assert result["new_root_ref"] == selection.read()
    assert len(result["ordered_components"]) == 3
    assert result["external_plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert (tmp_path / "recovery/manifest.json").is_file()


def _adopt(case: dict[str, Any], distribution: Path, expected: str, suffix: str) -> dict[str, Any]:
    return adopt_bundled_distribution(
        distribution=distribution, profile=distribution / "profiles/default.json",
        workspace=case["workspace"], plugins_home=case["home"], config_path=case["config"],
        receipt_path=case["receipt"], backup_dir=case["source"].parent / f"backup-{suffix}",
        expected_root_ref=expected,
    )


def _selection_refs(selection: PluginSelection, root_ref: str) -> tuple[str, ...]:
    """Read string refs from the archive's generic JSON descriptor."""
    components = selection.archive.read_descriptor(root_ref)["components"]
    assert isinstance(components, tuple)
    refs: list[str] = []
    for ref in components:
        assert isinstance(ref, str)
        refs.append(ref)
    return tuple(refs)


@pytest.mark.asyncio
async def test_adopt_bundled_distribution_selects_once_and_boots_exact_input(tmp_path):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    receipt = case["receipt"].read_bytes()
    old_selection_bytes = PluginSelection(case["workspace"]).path.read_bytes()
    selected = _adopt(case, release_b, case["root"], "b")
    assert selected["status"] == "selected_not_started"
    assert selected["new_root_ref"] != case["root"]
    assert PluginSelection(case["workspace"]).read() == selected["new_root_ref"]
    assert (tmp_path / "backup-b/workspace/runtime/plugin-stable.json").read_bytes() == old_selection_bytes
    assert (tmp_path / "backup-b/workspace/runtime/plugin-reloads.sqlite3").is_file()
    recovery = json.loads((tmp_path / "backup-b/recovery.json").read_text())
    assert set(recovery["bundle_sha256"]) == {"peer@release", "target@release"}
    assert case["receipt"].read_bytes() == receipt
    assert _adopt(case, release_b, selected["new_root_ref"], "repeat")["status"] == "already_selected_not_started"
    assert not (tmp_path / "backup-repeat").exists()
    owner = PluginManager([], event_bus=EventBus(), workspace=case["workspace"],
                          installed_cache_root=case["home"] / "cache")
    try:
        await owner.load_all()
        assert owner.generation("target@release").instance.version == "2"
        assert owner.generation("peer@release").instance.version == "2"
    finally:
        await owner.terminate_all()
    release_c, _ = _offline_release(case, "3")
    next_result = _adopt(case, release_c, selected["new_root_ref"], "c")
    assert next_result["status"] == "selected_not_started"
    assert next_result["new_root_ref"] != selected["new_root_ref"]
    assert case["receipt"].read_bytes() == receipt


def test_adopt_bundled_distribution_reruns_committed_install_after_prepare_failure(tmp_path, monkeypatch):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    import scripts.install_plugin_distribution as offline
    original = offline.prepare_plugin_input

    def fail_target(mod, *, workspace, archive):
        if mod["name"] == "target":
            raise SyntaxError("injected after formal install")
        return original(mod, workspace=workspace, archive=archive)

    monkeypatch.setattr(offline, "prepare_plugin_input", fail_target)
    with pytest.raises(SyntaxError, match="injected"):
        _adopt(case, release_b, case["root"], "failed")
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert (tmp_path / "backup-failed/recovery.json").is_file()
    with closing(sqlite3.connect(case["workspace"] / "runtime/plugin-reloads.sqlite3")) as conn:
        old_ids = {row[0] for row in conn.execute(
            "SELECT update_id FROM plugin_updates WHERE plugin_id IN ('peer@release','target@release')"
        )}
        assert conn.execute(
            "SELECT count(*) FROM plugin_updates WHERE plugin_id='target@release' "
            "AND phase='committed' AND input_ref IS NULL"
        ).fetchone()[0] >= 1
    monkeypatch.setattr(offline, "prepare_plugin_input", original)
    result = _adopt(case, release_b, case["root"], "retry")
    assert result["status"] == "selected_not_started"
    assert {item["mode"] for item in result["changed"]} == {"resume_install_B"}
    assert PluginSelection(case["workspace"]).read() == result["new_root_ref"]
    with closing(sqlite3.connect(case["workspace"] / "runtime/plugin-reloads.sqlite3")) as conn:
        new_ids = {row[0] for row in conn.execute(
            "SELECT update_id FROM plugin_updates WHERE plugin_id IN ('peer@release','target@release')"
        )}
    assert old_ids < new_ids


def test_adopt_bundled_distribution_keeps_a_after_real_secondary_source_compile_error(tmp_path):
    case = _offline_case(tmp_path)
    broken = "def bad(:\n"
    with pytest.raises(SyntaxError):
        compile(broken, "target/helper.py", "exec")
    (case["source"] / "plugins/target/helper.py").write_text(broken)
    release_b, _ = _offline_release(case, "2")
    with pytest.raises(PluginSourceCompileError):
        _adopt(case, release_b, case["root"], "compile")
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert (tmp_path / "backup-compile/recovery.json").is_file()
    with closing(sqlite3.connect(case["workspace"] / "runtime/plugin-reloads.sqlite3")) as conn:
        assert conn.execute(
            "SELECT count(*) FROM plugin_updates WHERE plugin_id='target@release' "
            "AND phase='committed' AND input_ref IS NULL"
        ).fetchone()[0] >= 1


def test_adopt_bundled_distribution_rejects_bad_pointer_before_backup(tmp_path):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    pointer = case["home"] / "cache/release/target/.pointers.json"
    pointer.write_text("broken")
    with pytest.raises((ValueError, RuntimeError)):
        _adopt(case, release_b, case["root"], "bad")
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert not (tmp_path / "backup-bad").exists()


@pytest.mark.parametrize("damage", ["provenance", "archive", "manifest"])
def test_adopt_bundled_distribution_rejects_damaged_identity_before_backup(tmp_path, damage):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    if damage == "provenance":
        pointer = json.loads((case["home"] / "cache/release/target/.pointers.json").read_text())
        artifact = case["home"] / "cache/release/target" / pointer["stable"]
        (artifact / ".akashic-source.json").write_text("broken")
    elif damage == "archive":
        selection = PluginSelection(case["workspace"])
        refs = _selection_refs(selection, case["root"])
        target = next(ref for ref in refs if selection.archive.read_descriptor(ref)["plugin_id"] == "target@release")
        (selection.archive.path / f"{target}.json").write_text("broken")
    else:
        (case["home"] / "manifest.toml").write_text("broken")
    with pytest.raises((ValueError, RuntimeError)):
        _adopt(case, release_b, case["root"], damage)
    assert not (tmp_path / f"backup-{damage}").exists()


def test_adopt_bundled_distribution_rejects_pending_reload_and_armed_install(tmp_path):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    journal = ReloadJournal(case["workspace"])
    journal.arm_update(update_id="pending-b2", plugin_id="target@release",
                       plugin_base=case["home"] / "cache/release/target", previous=None,
                       candidate=ArtifactPointer(".artifacts/new"),
                       previous_enabled=True)
    with pytest.raises(RuntimeError, match="armed install"):
        _adopt(case, release_b, case["root"], "armed")
    assert not (tmp_path / "backup-armed").exists()


def test_stopped_exact_rollback_unblocks_bundled_adoption(tmp_path):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    base = case["home"] / "cache/release/target"
    previous = read_pointers(base)
    assert previous is not None
    journal = ReloadJournal(case["workspace"])
    journal.arm_update(update_id="interrupted-before-pointer", plugin_id="target@release",
                       plugin_base=base, previous=previous,
                       candidate=ArtifactPointer(".artifacts/new"), previous_enabled=True)
    with pytest.raises(RuntimeError, match="armed install"):
        _adopt(case, release_b, case["root"], "blocked")
    assert not (tmp_path / "backup-blocked").exists()

    settled = rollback_plugin_install(
        workspace=case["workspace"], plugins_home=case["home"],
        update_id="interrupted-before-pointer", expected_root_ref=case["root"],
        backup_dir=tmp_path / "rollback-target",
    )
    assert settled["status"] == "rolled_back"
    assert journal.update("interrupted-before-pointer").phase == "rolled_back"
    assert read_pointers(base) == previous
    assert PluginSelection(case["workspace"]).read() == case["root"]
    adopted = _adopt(case, release_b, case["root"], "after-rollback")
    assert adopted["status"] == "selected_not_started"
    assert adopted["new_root_ref"] != case["root"]


def test_adopt_bundled_distribution_rejects_pending_reload_before_backup(tmp_path):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    journal = ReloadJournal(case["workspace"])
    journal.begin(plugin_id="target@release", base_snapshot_id="base", generation_id="next",
                  source_revision="source", config_revision="config")
    with pytest.raises(RuntimeError, match="pending reload"):
        _adopt(case, release_b, case["root"], "pending")
    assert not (tmp_path / "backup-pending").exists()


@pytest.mark.parametrize("outcome", ["conflict", "uncertain"])
def test_adopt_bundled_distribution_preserves_prepared_install_after_cas_failure(tmp_path, monkeypatch, outcome):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    original = PluginSelection.commit

    def fail_commit(self, components, *, expected_ref):
        if outcome == "conflict":
            raise SelectionConflictError("injected conflict")
        raise SelectionWriteError(operation="commit", target_ref=None, outcome="uncertain",
                                  observed_ref=None, observation_error=RuntimeError("injected"))

    monkeypatch.setattr(PluginSelection, "commit", fail_commit)
    with pytest.raises(SelectionConflictError if outcome == "conflict" else SelectionWriteError):
        _adopt(case, release_b, case["root"], outcome)
    assert PluginSelection(case["workspace"]).read() == case["root"]
    with closing(sqlite3.connect(case["workspace"] / "runtime/plugin-reloads.sqlite3")) as conn:
        assert conn.execute(
            "SELECT count(*) FROM plugin_updates WHERE phase='committed' AND input_ref IS NOT NULL"
        ).fetchone()[0] >= 2
    monkeypatch.setattr(PluginSelection, "commit", original)
    result = _adopt(case, release_b, case["root"], f"retry-{outcome}")
    assert result["status"] == "selected_not_started"
    assert {item["mode"] for item in result["changed"]} == {"resume_install_B"}


def test_adopt_bundled_distribution_keeps_external_and_reports_partial(tmp_path, monkeypatch, capsys):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    external = tmp_path / "external-target"
    external.mkdir()
    source_text = (
        "api_version = 3\nname = 'target'\nversion = 'external'\n"
        "async def apply(ctx):\n    return None\n"
    )
    compile(source_text, "external-target/plugin.py", "exec")
    (external / "plugin.py").write_text(source_text)
    subprocess.run(["git", "init", str(external)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(external), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(external), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
        "-c", "core.hooksPath=/dev/null", "commit", "-m", "external",
    ], check=True, capture_output=True)
    installed = install_git_plugin(workspace=case["workspace"], plugins_home=case["home"],
                                   source=str(external), marketplace="release")
    identity = load_static_plugin_manifest(installed.installed_path)
    prepared = prepare_plugin_input({
        "name": "target", "marketplace": "release", "plugin_root": str(installed.installed_path),
        "module_path": str(installed.installed_path / "plugin.py"),
        "manifest_digest": identity.identity_digest, "source_type": "installed",
    }, workspace=case["workspace"], archive=PluginArchive(case["workspace"] / "runtime/plugin-archives"))
    selection = PluginSelection(case["workspace"])
    original = _selection_refs(selection, case["root"])
    external_root = selection.commit(tuple(
        prepared.archive_ref if selection.archive.read_descriptor(ref)["plugin_id"] == "target@release" else ref
        for ref in original
    ), expected_ref=case["root"])
    result = _adopt(case, release_b, external_root, "partial")
    assert result["status"] == "partial_selected_not_started"
    assert result["skipped_external"] == ["target@release"]
    assert [item["plugin_id"] for item in result["changed"]] == ["peer@release"]
    selected = _selection_refs(selection, result["new_root_ref"])
    assert prepared.archive_ref in selected
    monkeypatch.setattr(sys, "argv", [
        "distribution", "--distribution", str(release_b),
        "--profile", str(release_b / "profiles/default.json"),
        "--workspace", str(case["workspace"]), "--plugins-home", str(case["home"]),
        "--config", str(case["config"]), "--receipt", str(case["receipt"]),
        "--adopt-bundled", "--expected-root-ref", result["new_root_ref"],
        "--backup-dir", str(tmp_path / "unused"),
    ])
    with pytest.raises(SystemExit) as exit_info:
        distribution_main()
    assert exit_info.value.code == 3
    assert json.loads(capsys.readouterr().out)["status"] == "partial_selected_not_started"
    report = verify_distribution(release_b)
    target_row = next(row for row in report["plugins"] if row["name"] == "target")
    install_git_plugin(workspace=case["workspace"], plugins_home=case["home"],
                       source=str(release_b / target_row["file"]), marketplace="release",
                       ref_name=target_row["source_revision"])
    with pytest.raises(ValueError, match="不能覆盖外部选择"):
        _adopt(case, release_b, result["new_root_ref"], "external-drift")
    assert not (tmp_path / "backup-external-drift").exists()


def test_adopt_bundled_distribution_reports_no_eligible_targets(tmp_path, monkeypatch, capsys):
    case = _offline_case(tmp_path)
    new = case["source"] / "plugins/new/plugin.py"
    new.parent.mkdir(parents=True)
    source_text = "api_version = 3\nname = 'new'\nversion = '1'\nasync def apply(ctx):\n    return None\n"
    compile(source_text, "new/plugin.py", "exec")
    new.write_text(source_text)
    profile = case["source"] / "docker/host-runtime/profiles/default.json"
    document = json.loads(profile.read_text())
    document["plugins"] = [{"name": "new", "depends_on": [], "reason": "new source"}]
    profile.write_text(json.dumps(document))
    release, _ = _offline_release(case, "2")
    monkeypatch.setattr(sys, "argv", [
        "distribution", "--distribution", str(release),
        "--profile", str(release / "profiles/default.json"),
        "--workspace", str(case["workspace"]), "--plugins-home", str(case["home"]),
        "--config", str(case["config"]), "--receipt", str(case["receipt"]),
        "--adopt-bundled", "--expected-root-ref", case["root"],
        "--backup-dir", str(tmp_path / "unused"),
    ])
    with pytest.raises(SystemExit) as exit_info:
        distribution_main()
    assert exit_info.value.code == 4
    assert json.loads(capsys.readouterr().out)["status"] == "no_eligible_targets"
    assert PluginSelection(case["workspace"]).read() == case["root"]
    assert not (tmp_path / "unused").exists()


def test_adopt_bundled_distribution_cli_reports_selected_and_already_with_exit_zero(tmp_path, monkeypatch, capsys):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    argv = ["distribution", "--distribution", str(release_b),
            "--profile", str(release_b / "profiles/default.json"),
            "--workspace", str(case["workspace"]), "--plugins-home", str(case["home"]),
            "--config", str(case["config"]), "--receipt", str(case["receipt"]),
            "--adopt-bundled", "--expected-root-ref", case["root"],
            "--backup-dir", str(tmp_path / "cli-b")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as first:
        distribution_main()
    assert first.value.code == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected["status"] == "selected_not_started"
    argv[-3] = selected["new_root_ref"]
    argv[-1] = str(tmp_path / "cli-again")
    with pytest.raises(SystemExit) as again:
        distribution_main()
    assert again.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_selected_not_started"


def test_adopt_bundled_distribution_cli_reports_failure_with_exit_one(tmp_path, monkeypatch, capsys):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    (case["home"] / "cache/release/target/.pointers.json").write_text("broken")
    monkeypatch.setattr(sys, "argv", [
        "distribution", "--distribution", str(release_b),
        "--profile", str(release_b / "profiles/default.json"),
        "--workspace", str(case["workspace"]), "--plugins-home", str(case["home"]),
        "--config", str(case["config"]), "--receipt", str(case["receipt"]),
        "--adopt-bundled", "--expected-root-ref", case["root"],
        "--backup-dir", str(tmp_path / "failed-cli"),
    ])
    with pytest.raises(SystemExit) as failure:
        distribution_main()
    assert failure.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "failed-cli").exists()


def test_adopt_bundled_distribution_excludes_disabled_and_new_profile_members(tmp_path):
    case = _offline_case(tmp_path)
    selection = PluginSelection(case["workspace"])
    original = _selection_refs(selection, case["root"])
    peer_only = tuple(ref for ref in original if selection.archive.read_descriptor(ref)["plugin_id"] == "peer@release")
    current = selection.commit(peer_only, expected_ref=case["root"])
    set_installed_plugin_enabled("target@release", enabled=False, plugins_home=case["home"])
    new = case["source"] / "plugins/new/plugin.py"
    new.parent.mkdir(parents=True)
    source_text = "api_version = 3\nname = 'new'\nversion = '1'\nasync def apply(ctx):\n    return None\n"
    compile(source_text, "new/plugin.py", "exec")
    new.write_text(source_text)
    profile = case["source"] / "docker/host-runtime/profiles/default.json"
    document = json.loads(profile.read_text())
    document["plugins"].append({"name": "new", "depends_on": [], "reason": "new source"})
    profile.write_text(json.dumps(document))
    release_b, _ = _offline_release(case, "2")
    result = _adopt(case, release_b, current, "excluded")
    assert result["status"] == "selected_not_started"
    assert {item["plugin_id"] for item in result["excluded"]} == {"target@release", "new@release"}
    assert [item["plugin_id"] for item in result["changed"]] == ["peer@release"]
    assert load_plugin_manifest(case["home"])["target@release"] is False
    selected = _selection_refs(selection, result["new_root_ref"])
    assert len(selected) == 1


def test_adopt_bundled_distribution_lock_conflict_prevents_backup(tmp_path):
    case = _offline_case(tmp_path)
    release_b, _ = _offline_release(case, "2")
    lock = WorkspaceMaintenanceLock(case["workspace"])
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="workspace 仍有生命周期 owner"):
            _adopt(case, release_b, case["root"], "locked")
    finally:
        lock.release()
    assert not (tmp_path / "backup-locked").exists()
    assert PluginSelection(case["workspace"]).read() == case["root"]


def test_host_runtime_cli_defaults_to_distribution(monkeypatch, tmp_path, capsys):
    calls: list[str] = []

    def fake_distribution(**kwargs):
        calls.append("distribution")
        return {"mode": "distribution", "repository": str(kwargs["repository"])}

    def fail_legacy(**kwargs):
        pytest.fail("正式 CLI 不应默认选择旧 checkout builder")

    monkeypatch.setattr(host_runtime_release, "build_distribution_release", fake_distribution)
    monkeypatch.setattr(host_runtime_release, "build_release", fail_legacy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_host_runtime_release.py",
            "--repository", str(tmp_path / "repo"),
            "--commit", "a" * 40,
            "--image-tag", "akashic:test",
            "--output-manifest", str(tmp_path / "manifest.json"),
        ],
    )
    host_runtime_release.main()
    assert calls == ["distribution"]
    assert json.loads(capsys.readouterr().out)["mode"] == "distribution"


def test_public_release_image_uses_distribution_and_keeps_bridge_identity(
    monkeypatch, tmp_path
):
    import scripts.akashic_release.image as release_image

    calls: list[dict[str, object]] = []
    image_id = "sha256:" + "a" * 64

    def fake_distribution(**kwargs):
        calls.append(kwargs)
        return {
            "schemaVersion": 2,
            "imageId": image_id,
            "runtimeInfo": {"schemaVersion": 3},
        }

    bridge_identity = {"schemaVersion": 1, "toolchainDigest": "bridge-digest"}
    monkeypatch.setattr(release_image, "build_distribution_release", fake_distribution)
    monkeypatch.setattr(
        release_image,
        "declared_toolchain_identity",
        lambda commit, mise_config: bridge_identity,
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "mise.toml").write_text("[tools]\n", encoding="utf-8")
    manifest = tmp_path / "release.json"

    result = release_image.prepare_core_image(
        checkout=checkout,
        commit="b" * 40,
        manifest=manifest,
        image_tag="akashic:test",
    )

    assert calls[0]["repository"] == checkout
    assert calls[0]["requested_commit"] == "b" * 40
    assert result["hostToolchainIdentity"] == bridge_identity
    assert json.loads(manifest.read_text(encoding="utf-8"))["schemaVersion"] == 2


def test_distribution_release_manifest_passes_deployment_image_verifier(
    monkeypatch, tmp_path
):
    import scripts.verify_host_runtime_deployment as deployment

    image_id = "sha256:" + "c" * 64
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "imageId": image_id,
                "runtimeInfo": {"schemaVersion": 3},
            }
        ),
        encoding="utf-8",
    )

    def fake_inspect(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=image_id + "\n", stderr="")

    monkeypatch.setattr(deployment.subprocess, "run", fake_inspect)
    assert deployment.verify_deployment_image(manifest, image_id) == image_id


def test_release_environment_exports_distribution_tree(monkeypatch, tmp_path):
    from scripts.akashic_release import activate
    from scripts.akashic_release.model import ReleasePaths

    monkeypatch.setattr(activate, "docker_socket_gid", lambda: 961)
    host_python = tmp_path / "mise" / "python" / "3.14.6"
    host_python.mkdir(parents=True)
    monkeypatch.setattr(activate, "_base_python_prefix", lambda _: host_python)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-only-key")
    paths = ReleasePaths(tmp_path / "release")
    paths.create_layout()
    values = activate.release_environment(
        paths=paths,
        manifest={
            "sourceCommit": "a" * 40,
            "sourceTree": "b" * 40,
            "hostToolchainIdentity": {"toolchainDigest": "c" * 64},
            "imageId": "sha256:" + "d" * 64,
        },
        current={},
        mise=tmp_path / "mise",
    )

    assert values["AKASHIC_RUNTIME_COMMIT"] == "a" * 40
    assert values["AKASHIC_RUNTIME_TREE"] == "b" * 40
    assert values["AKASHIC_HOST_PYTHON_PREFIX"] == str(host_python)


def test_formal_host_context_contains_core_and_bundles_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    plugin = source / "plugins" / "one"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        'api_version = 3\nname = "one"\nversion = "1"\ndef apply(ctx): pass\n'
    )
    (source / "main.py").write_text("print('core')\n")
    runtime = source / "docker" / "host-runtime"
    profile = runtime / "profiles"
    profile.mkdir(parents=True)
    (runtime / "Dockerfile.distribution").write_text("FROM scratch\n")
    (runtime / "distribution-entrypoint.sh").write_text("#!/bin/sh\n")
    (profile / "default.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "fixture",
        "marketplace": "release",
        "initialization": {"plugin_configs": []},
        "plugins": [{"name": "one", "depends_on": [], "reason": "fixture"}],
    }) + "\n")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "-c", "commit.gpgSign=false",
        "commit", "-m", "source",
    ], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
    context = tmp_path / "formal-context"
    identity = _create_context(source, commit, tree, context)
    assert identity["sourceCommit"] == commit
    assert (context / "core.tar").is_file()
    assert (context / "one.bundle").is_file()
    assert not (context / "plugins").exists()
    assert not (context / "private.txt").exists()


def test_static_asset_directories_are_real_tar_directories(tmp_path):
    asset_root = tmp_path / "assets"
    (asset_root / "sdk").mkdir(parents=True)
    (asset_root / "sdk" / "react.js").write_text("export {}\n")
    with io.BytesIO() as stream:
        with tarfile.open(fileobj=stream, mode="w"):
            pass
        archive = _append_tree(
            stream.getvalue(), asset_root, "static/dashboard", mtime=1
        )
    with tarfile.open(fileobj=io.BytesIO(archive)) as result:
        assert result.getmember("static/dashboard/sdk/").isdir()
        assert result.getmember("static/dashboard/sdk/react.js").isfile()
