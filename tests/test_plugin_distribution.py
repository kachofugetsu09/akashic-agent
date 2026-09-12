"""发布制品只能含选定宿主路径与各插件自己的源码。"""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib

import pytest

from agent.plugins.install import install_git_plugin, set_installed_plugin_enabled
from scripts.build_host_runtime_release import _create_context
from scripts.build_plugin_distribution import _append_tree, build
from scripts.install_plugin_distribution import (
    _write_receipt,
    ensure_profile,
    extract_core,
    install_profile,
    verify_distribution,
)


def test_distribution_installs_isolated_git_sources_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("one", "two", "unused"):
        root = source / "plugins" / name
        root.mkdir(parents=True)
        (root / "plugin.py").write_text(f'api_version = 3\nname = "{name}"\nversion = "1"\ndef apply(ctx, config): pass\n')
        (root / "akashic.plugin.toml").write_text(f'schema_version = 1\napi_version = 3\nname = "{name}"\nversion = "1"\nentrypoint = "plugin.py"\n')
    (source / "main.py").write_text('print("core")\n')
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
    with tarfile.open(fileobj=io.BytesIO((output / "core.tar").read_bytes())) as archive:
        assert set(archive.getnames()) == {
            "config.example.toml", "main.py", "runtime-dependencies.json",
            "docker", "docker/host-runtime", "docker/host-runtime/profiles",
            "docker/host-runtime/profiles/default.json",
            "docker/host-runtime/Dockerfile.distribution",
            "docker/host-runtime/distribution-entrypoint.sh",
        }
        assert not any(name == "plugins" or name.startswith("plugins/") for name in archive.getnames())
    assert {row["name"] for row in report["plugins"]} == {"one", "two", "unused"}
    assert {row["path"] for row in report["runtime_wiring"]} == {
        "Dockerfile.distribution",
        "distribution-entrypoint.sh",
    }
    assert (output / "Dockerfile.distribution").is_file()
    assert "COPY core.tar /opt/akashic/distribution/core.tar" in (
        Path("docker/host-runtime/Dockerfile.distribution").read_text()
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
    context_config = tmp_path / "profile-workspace/plugin-data/one-distribution/config.local.toml"
    assert tomllib.loads(context_config.read_text()) == {
        "prompt_sources": {"fixture": "one@distribution"},
        "summary_source": ["summary", "one@distribution"],
        "unsafe": "x\"\n[unexpected]\nvalue = \"bad\"",
    }
    assert not (tmp_path / "profile-home" / "cache" / "distribution" / "unused").exists()
    receipt_path = tmp_path / "profile-workspace/runtime/distribution-install.json"
    _write_receipt(receipt_path, profile_receipt)
    context_config.write_text('custom = "keep"\n', encoding="utf-8")
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
    assert tomllib.loads(context_config.read_text()) == {"custom": "keep"}

    external = tmp_path / "external-one"
    external.mkdir()
    (external / "plugin.py").write_text(
        'api_version = 3\nname = "one"\nversion = "2"\ndef apply(ctx, config): pass\n'
    )
    (external / "akashic.plugin.toml").write_text(
        'schema_version = 1\napi_version = 3\nname = "one"\nversion = "2"\nentrypoint = "plugin.py"\n'
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
    assert tomllib.loads(context_config.read_text()) == {"custom": "keep"}
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
    for row in report["plugins"]:
        installed = install_git_plugin(workspace=tmp_path / "workspace", plugins_home=tmp_path / "home",
            source=str(output / row["file"]), marketplace="distribution")
        assert installed.source_revision == row["source_revision"]
        assert not (installed.installed_path / "plugins").exists()
        assert not (installed.installed_path / "private.txt").exists()
        provenance = json.loads((installed.installed_path / ".akashic-source.json").read_text())
        assert provenance == {"commit": report["source_commit"], "path": row["source_path"]}
    with pytest.raises(FileExistsError):
        build(source, "HEAD", output)


def test_formal_host_context_contains_core_and_bundles_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    plugin = source / "plugins" / "one"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        'api_version = 3\nname = "one"\nversion = "1"\ndef apply(ctx, config): pass\n'
    )
    (plugin / "akashic.plugin.toml").write_text(
        'schema_version = 1\napi_version = 3\nname = "one"\nversion = "1"\nentrypoint = "plugin.py"\n'
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
