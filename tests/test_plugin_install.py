from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import agent.plugins.install as install_module
from agent.plugins.artifacts import (
    ArtifactPointer,
    discard_latest_pointer,
    read_pointer,
)
from agent.plugins.install import (
    install_git_plugin,
    set_installed_plugin_enabled,
    uninstall_plugin,
)
from agent.plugins.manifest import plugins_root
from agent.plugins.source_resolver import resolve_plugin_sources


def test_plugins_root_honors_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "isolated-plugin-home"
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(target))

    assert plugins_root() == target


def test_plugins_root_rejects_blank_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", "   ")

    with pytest.raises(ValueError, match="不能为空"):
        plugins_root()


def test_install_git_plugin_uses_static_v3_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "feed-mcp"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)

    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    data_dir = workspace / "plugin-data" / "feed-lab"
    data_dir.mkdir(parents=True)
    (data_dir / "state.json").write_text('{"keep":true}\n', encoding="utf-8")

    result = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    assert result.installed_path.parent == (
        home / "cache" / "lab" / "feed" / ".artifacts"
    )
    assert result.installed_path.name.startswith("1.0.0-")
    assert result.source_revision == _git_output(repo, "rev-parse", "HEAD")
    assert result.staged_candidate is False
    pointer_state = result.installed_path.parents[1] / ".pointers.json"
    assert pointer_state.is_file()
    assert not (pointer_state.parent / ".stable.json").exists()
    assert not (pointer_state.parent / ".latest.json").exists()
    assert (result.installed_path / "plugin.py").exists()
    installed_manifest = result.installed_path / "akashic.plugin.toml"
    assert installed_manifest.is_file()
    assert (
        tomllib.loads(installed_manifest.read_text(encoding="utf-8"))["name"] == "feed"
    )
    assert (result.data_path / "state.json").exists()
    manifest = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
    assert manifest == {"plugins": {"feed@lab": {"enabled": True}}}


def test_install_git_plugin_reads_static_v3_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "citation"
    _write_v3_plugin(
        repo,
        name="citation",
        version="2.0.0",
        module_source="raise RuntimeError('must not import during install')\n",
    )
    _commit(repo)

    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    assert result.plugin_name == "citation"
    assert result.plugin_version == "2.0.0"
    assert (result.installed_path / "plugin.py").is_file()
    assert (result.installed_path / "akashic.plugin.toml").is_file()


def test_install_git_plugin_prepares_declared_mcp_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "run_mcp.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "mcp" / "requirements.txt").write_text("requests\n", encoding="utf-8")
    _write_v3_plugin(repo, name="feed")
    (repo / "akashic.plugin.toml").write_text(
        (repo / "akashic.plugin.toml").read_text(encoding="utf-8")
        + "\n"
        + "[[python]]\n"
        + 'requirements = "mcp/requirements.txt"\n'
        + "\n"
        + "[[mcp]]\n"
        + 'name = "feed"\n'
        + 'command = ["python", "mcp/run_mcp.py"]\n'
        + 'cwd = "mcp"\n',
        encoding="utf-8",
    )
    _commit(repo)
    calls: list[tuple[str, Path]] = []

    def fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        calls.append((label, cwd))
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(install_module, "_run_command", fake_run)
    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    assert [label for label, _ in calls] == [
        "feed python[0] venv",
        "feed python[0] pip install",
    ]
    assert all(
        cwd.name == "mcp"
        and cwd.is_relative_to(result.installed_path.parents[2])
        and cwd != result.installed_path / "mcp"
        for _, cwd in calls
    )
    assert not (result.installed_path / "mcp" / "servers.json").exists()


def test_retry_reuses_artifact_with_core_generated_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "requirements.txt").write_text("", encoding="utf-8")
    _write_v3_plugin(repo, name="feed", marker="v1")
    (repo / "akashic.plugin.toml").write_text(
        (repo / "akashic.plugin.toml").read_text(encoding="utf-8")
        + '\n[[python]]\nrequirements = "mcp/requirements.txt"\n',
        encoding="utf-8",
    )
    _commit(repo)

    def fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.symlink_to(sys.executable)

    monkeypatch.setattr(install_module, "_run_command", fake_run)
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    _ = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    plugin_path = repo / "plugin.py"
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("v1", "v2"),
        encoding="utf-8",
    )
    _commit(repo)
    candidate = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
        stage_candidate=True,
    )
    _ = discard_latest_pointer(candidate.installed_path.parents[1])

    retried = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
        stage_candidate=True,
    )

    assert retried.installed_path == candidate.installed_path
    assert retried.staged_candidate is True


def test_plugin_enable_disable_and_uninstall_preserve_data(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    cache = home / "cache" / "github" / "fitbit" / "1.0.0"
    data = workspace / "plugin-data" / "fitbit-github"
    _write_v3_plugin(cache, name="fitbit")
    data.mkdir(parents=True)
    state = data / "sleep-model.bin"
    state.write_bytes(b"model")
    (home / "manifest.toml").write_text(
        '[plugins."fitbit@github"]\nenabled = true\n',
        encoding="utf-8",
    )

    set_installed_plugin_enabled(
        "fitbit@github",
        enabled=False,
        plugins_home=home,
    )
    manifest = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
    assert manifest["plugins"]["fitbit@github"]["enabled"] is False

    set_installed_plugin_enabled(
        "fitbit@github",
        enabled=True,
        plugins_home=home,
    )
    disabled_before_removal = False

    def wait_until_disabled(plugin_id: str) -> None:
        nonlocal disabled_before_removal
        current = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
        disabled_before_removal = (
            plugin_id == "fitbit@github"
            and current["plugins"][plugin_id]["enabled"] is False
            and cache.parent.exists()
            and state.exists()
        )

    removed_cache, retained_data = uninstall_plugin(
        "fitbit@github",
        workspace=workspace,
        plugins_home=home,
        wait_until_disabled=wait_until_disabled,
    )

    assert disabled_before_removal
    assert removed_cache == home / "cache" / "github" / "fitbit"
    assert not removed_cache.exists()
    assert retained_data == data
    assert state.read_bytes() == b"model"
    assert tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8")) == {
        "plugins": {}
    }


def test_plugin_management_rejects_non_installed_plugin_id(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    (home / "cache" / "github" / "fitbit").mkdir(parents=True)
    (home / "manifest.toml").parent.mkdir(parents=True, exist_ok=True)
    (home / "manifest.toml").write_text("", encoding="utf-8")

    for plugin_id in (
        "fitbit",
        "../fitbit@github",
        "fitbit@../github",
        "fitbit\ncorrupt@github",
    ):
        try:
            set_installed_plugin_enabled(
                plugin_id,
                enabled=False,
                plugins_home=home,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝插件 ID: {plugin_id}")


def test_install_failure_restores_previous_cache_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="old")
    _commit(repo)
    home = tmp_path / "plugins-home"
    first = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )
    old_content = (first.installed_path / "plugin.py").read_text(encoding="utf-8")

    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("'old'", "'new'"),
        encoding="utf-8",
    )
    _commit(repo)

    def fail_prepare(plugin_root: Path, static_manifest: object) -> None:
        resolved = resolve_plugin_sources(
            [],
            installed_cache_root=home / "cache",
        )
        assert len(resolved) == 1
        assert resolved[0].plugin_root == first.installed_path
        assert (first.installed_path / "plugin.py").read_text(
            encoding="utf-8"
        ) == old_content
        raise RuntimeError(f"prepare failed: {plugin_root}")

    monkeypatch.setattr(install_module, "_prepare_static_python_runtimes", fail_prepare)
    with pytest.raises(RuntimeError, match="prepare failed"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert (first.installed_path / "plugin.py").read_text(
        encoding="utf-8"
    ) == old_content
    plugin_base = home / "cache" / "lab" / "feed"
    assert read_pointer(plugin_base, "stable") == read_pointer(plugin_base, "latest")
    assert tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8")) == {
        "plugins": {"feed@lab": {"enabled": True}}
    }
    assert not any(
        child.name.startswith(".feed-install-")
        for child in (home / "cache" / "lab").iterdir()
    )

    monkeypatch.undo()

    def fail_manifest(*args, **kwargs) -> Path:
        raise OSError("manifest write failed")

    monkeypatch.setattr(install_module, "upsert_plugin_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest write failed"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )
    assert (first.installed_path / "plugin.py").read_text(
        encoding="utf-8"
    ) == old_content


def test_install_rejects_unsafe_path_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="../outside")
    _commit(repo)

    with pytest.raises(ValueError, match="安全的单一路径段"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="../outside",
            plugins_home=tmp_path / "plugins-home",
        )

    with pytest.raises(ValueError, match="静态 manifest name 无效"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )


def test_install_can_stage_one_latest_without_changing_stable(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="stable")
    _commit(repo)
    home = tmp_path / "plugins-home"
    first = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("stable", "latest"),
        encoding="utf-8",
    )
    _commit(repo)

    second = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
        stage_candidate=True,
    )

    stable = resolve_plugin_sources([], installed_cache_root=home / "cache")[0]
    latest = resolve_plugin_sources(
        [],
        installed_cache_root=home / "cache",
        installed_selector="latest",
    )[0]
    assert stable.plugin_root == first.installed_path
    assert latest.plugin_root == second.installed_path
    assert first.installed_path.exists()
    assert second.installed_path.exists()
    assert second.staged_candidate is True
    with pytest.raises(RuntimeError, match="等待 promote/discard"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
            stage_candidate=True,
        )


def test_first_staged_install_has_no_stable_until_promotion(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)
    home = tmp_path / "plugins-home"

    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
        stage_candidate=True,
    )

    assert resolve_plugin_sources([], installed_cache_root=home / "cache") == []
    assert (
        resolve_plugin_sources(
            [],
            installed_cache_root=home / "cache",
            installed_selector="latest",
        )[0].plugin_root
        == result.installed_path
    )

    _ = discard_latest_pointer(result.installed_path.parents[1])

    assert read_pointer(result.installed_path.parents[1], "stable") == ArtifactPointer(
        None
    )
    assert read_pointer(result.installed_path.parents[1], "latest") == ArtifactPointer(
        None
    )
    assert resolve_plugin_sources([], installed_cache_root=home / "cache") == []
    assert (
        resolve_plugin_sources(
            [],
            installed_cache_root=home / "cache",
            installed_selector="latest",
        )
        == []
    )


def test_default_update_keeps_immediate_stable_compatibility(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="v1")
    _commit(repo)
    home = tmp_path / "plugins-home"
    first = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("v1", "v2"),
        encoding="utf-8",
    )
    _commit(repo)

    second = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    resolved = resolve_plugin_sources([], installed_cache_root=home / "cache")
    assert resolved[0].plugin_root == second.installed_path
    assert second.staged_candidate is False
    assert first.installed_path.exists()


def test_install_rejects_visible_nonversion_cache_entry(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)
    home = tmp_path / "plugins-home"
    invalid_entry = home / "cache" / "lab" / "feed" / "unexpected.txt"
    invalid_entry.parent.mkdir(parents=True)
    invalid_entry.write_text("broken", encoding="utf-8")

    with pytest.raises(ValueError, match="cache 版本不是目录"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert invalid_entry.read_text(encoding="utf-8") == "broken"
    assert not (invalid_entry.parent / "1.0.0").exists()


def test_install_rejects_legacy_visible_version_directory(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    _commit(repo)
    home = tmp_path / "plugins-home"
    legacy = home / "cache/lab/feed/1.0.0"
    legacy.mkdir(parents=True)
    (legacy / "state.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="不受支持的旧版可见目录"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert (legacy / "state.txt").read_text(encoding="utf-8") == "keep"
    assert not (legacy.parent / ".pointers.json").exists()


def test_install_allows_internal_source_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    _write_v3_plugin(repo, name="feed")
    (repo / "helper.py").write_text("MARKER = 'inside'\n", encoding="utf-8")
    (repo / "linked_helper.py").symlink_to("helper.py")
    _commit(repo)

    result = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    linked_helper = result.installed_path / "linked_helper.py"
    assert linked_helper.read_text(encoding="utf-8") == "MARKER = 'inside'\n"
    assert not linked_helper.is_symlink()


def test_install_rejects_source_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    outside = tmp_path / "outside.py"
    outside.write_text("MARKER = 'outside'\n", encoding="utf-8")
    _write_v3_plugin(repo, name="feed")
    (repo / "linked_helper.py").symlink_to(outside)
    _commit(repo)

    with pytest.raises(ValueError, match="符号链接越界"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )


def test_install_accepts_branch_tag_and_commit_refs(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    plugin_path = repo / "plugin.py"
    _write_v3_plugin(repo, name="feed", marker="initial")
    _commit(repo)
    initial_sha = _git_output(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "release")
    plugin_path.write_text(
        plugin_path.read_text(encoding="utf-8").replace("initial", "head"),
        encoding="utf-8",
    )
    _commit(repo)
    _git(repo, "tag", "v2")

    for ref_name, marker in (
        ("release", "initial"),
        ("v2", "head"),
        (initial_sha, "initial"),
    ):
        result = install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            ref_name=ref_name,
            plugins_home=tmp_path / f"home-{ref_name}",
        )
        assert f"marker = '{marker}'" in (
            result.installed_path / "plugin.py"
        ).read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="命令选项"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            ref_name="-bad",
            plugins_home=tmp_path / "home-option",
        )
    with pytest.raises(ValueError, match="首尾空白"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            ref_name=" release",
            plugins_home=tmp_path / "home-whitespace",
        )


def test_uninstall_converges_when_cache_is_already_missing(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    data = workspace / "plugin-data" / "feed-github"
    data.mkdir(parents=True)
    state = data / "state.db"
    state.write_bytes(b"keep")
    home.mkdir(parents=True)
    (home / "manifest.toml").write_text(
        '[plugins."feed@github"]\nenabled = true\n',
        encoding="utf-8",
    )

    cache, retained = uninstall_plugin(
        "feed@github",
        workspace=workspace,
        plugins_home=home,
    )

    assert not cache.exists()
    assert retained == data
    assert state.read_bytes() == b"keep"
    assert tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8")) == {
        "plugins": {}
    }


def _write_v3_plugin(
    root: Path,
    *,
    name: str,
    version: str = "1.0.0",
    marker: str | None = None,
    module_source: str | None = None,
) -> None:
    """Create a static v3 artifact fixture with a matching entrypoint."""

    # 1. Write an import-free entrypoint whose optional marker tracks source refs.
    root.mkdir(parents=True, exist_ok=True)
    if module_source is None:
        lines = [
            "api_version = 3",
            f"name = {name!r}",
            f"version = {version!r}",
        ]
        if marker is not None:
            lines.append(f"marker = {marker!r}")
        module_source = "\n".join(lines) + "\n"
    (root / "plugin.py").write_text(module_source, encoding="utf-8")

    # 2. Write the immutable static identity consumed by the installer.
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {name!r}\n"
        f"version = {version!r}\n"
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )


def _commit(repo: Path) -> None:
    for args in (
        ["init"],
        ["config", "user.name", "test"],
        ["config", "user.email", "test@example.com"],
        ["add", "."],
        ["commit", "-m", "init"],
    ):
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr


def _git(repo: Path, *args: str) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()
