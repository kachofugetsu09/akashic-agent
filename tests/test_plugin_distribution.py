"""发布制品只能含选定宿主路径与各插件自己的源码。"""
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from agent.plugins.install import install_git_plugin
from scripts.build_host_runtime_release import _create_context
from scripts.build_plugin_distribution import build
from scripts.install_plugin_distribution import extract_core, install_profile, verify_distribution


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
    profile = source / "docker" / "host-runtime" / "profiles"
    profile.mkdir(parents=True)
    (profile / "default.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "fixture",
        "marketplace": "distribution",
        "initialization": {
            "authorization": {"owner": "one"},
            "prompt": {"owner": "two"},
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
        }
        assert not any(name == "plugins" or name.startswith("plugins/") for name in archive.getnames())
    assert {row["name"] for row in report["plugins"]} == {"one", "two", "unused"}
    verified = verify_distribution(output)
    core_root = extract_core(output, tmp_path / "core", report=verified)
    assert (core_root / "config.example.toml").is_file()
    assert not (core_root / "plugins").exists()
    profile_receipt = install_profile(
        output,
        output / "profiles/default.json",
        workspace=tmp_path / "profile-workspace",
        plugins_home=tmp_path / "profile-home",
    )
    assert [item["name"] for item in profile_receipt["installed"]] == ["one", "two"]
    assert not (tmp_path / "profile-home" / "cache" / "distribution" / "unused").exists()
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
    profile = source / "docker" / "host-runtime" / "profiles"
    profile.mkdir(parents=True)
    (profile / "default.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "fixture",
        "marketplace": "release",
        "initialization": {
            "authorization": {"owner": "one"},
            "prompt": {"owner": "one"},
        },
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
