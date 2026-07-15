from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import agent.plugins.install as install_module
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


def test_install_git_plugin_uses_programmatic_declaration(tmp_path: Path) -> None:
    repo = tmp_path / "feed-mcp"
    repo.mkdir()
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills',)\n",
        encoding="utf-8",
    )
    (repo / "skills" / "feed-manage").mkdir(parents=True)
    (repo / "skills" / "feed-manage" / "SKILL.md").write_text("skill", encoding="utf-8")
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

    assert result.installed_path == home / "cache" / "lab" / "feed" / "1.0.0"
    assert (result.installed_path / "plugin.py").exists()
    assert (result.data_path / "state.json").exists()
    manifest = tomllib.loads((home / "manifest.toml").read_text(encoding="utf-8"))
    assert manifest == {"plugins": {"feed@lab": {"enabled": True}}}


def test_install_git_plugin_prepares_declared_mcp_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "run_mcp.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "mcp" / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (repo / "plugin.py").write_text(
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='feed', command=('python', 'mcp/run_mcp.py'))]\n",
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

    assert [label for label, _ in calls] == ["feed venv", "feed pip install"]
    assert all(
        cwd.name == "mcp"
        and cwd.is_relative_to(result.installed_path.parents[2])
        and cwd != result.installed_path / "mcp"
        for _, cwd in calls
    )
    assert not (result.installed_path / "mcp" / "servers.json").exists()


def test_plugin_enable_disable_and_uninstall_preserve_data(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    cache = home / "cache" / "github" / "fitbit" / "1.0.0"
    data = workspace / "plugin-data" / "fitbit-github"
    cache.mkdir(parents=True)
    data.mkdir(parents=True)
    (cache / "plugin.py").write_text("", encoding="utf-8")
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
    repo.mkdir()
    plugin_path = repo / "plugin.py"
    plugin_path.write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n"
        "    marker = 'old'\n",
        encoding="utf-8",
    )
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

    def fail_prepare(plugin_root: Path, servers) -> None:
        resolved = resolve_plugin_sources(
            [],
            installed_cache_root=home / "cache",
        )
        assert len(resolved) == 1
        assert resolved[0].plugin_root == first.installed_path
        assert (first.installed_path / "plugin.py").read_text(encoding="utf-8") == old_content
        raise RuntimeError(f"prepare failed: {plugin_root}")

    monkeypatch.setattr(install_module, "_prepare_plugin_mcp_runtimes", fail_prepare)
    with pytest.raises(RuntimeError, match="prepare failed"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=home,
        )

    assert (first.installed_path / "plugin.py").read_text(encoding="utf-8") == old_content
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
    assert (first.installed_path / "plugin.py").read_text(encoding="utf-8") == old_content


def test_install_rejects_unsafe_path_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    repo.mkdir()
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = '../outside'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
    _commit(repo)

    with pytest.raises(ValueError, match="安全的单一路径段"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="../outside",
            plugins_home=tmp_path / "plugins-home",
        )

    with pytest.raises(ValueError, match="安全的单一路径段"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )


def test_install_rejects_visible_nonversion_cache_entry(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    repo.mkdir()
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
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


def test_install_allows_internal_source_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    repo.mkdir()
    (repo / "helper.py").write_text("MARKER = 'inside'\n", encoding="utf-8")
    (repo / "linked_helper.py").symlink_to("helper.py")
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
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
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("MARKER = 'outside'\n", encoding="utf-8")
    (repo / "linked_helper.py").symlink_to(outside)
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
    _commit(repo)

    with pytest.raises(ValueError, match="符号链接越界"):
        install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="lab",
            plugins_home=tmp_path / "plugins-home",
        )


def test_plugin_probe_cleans_imported_submodules(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text("MARKER = 'ok'\n", encoding="utf-8")
    (tmp_path / "plugin.py").write_text(
        "from .helper import MARKER\n"
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n"
        "    marker = MARKER\n",
        encoding="utf-8",
    )

    _ = install_module._load_plugin_class(tmp_path)

    assert not any(name.startswith("akasic_plugin_install_") for name in sys.modules)


def test_mcp_runtime_path_cannot_escape_plugin_root(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()

    with pytest.raises(ValueError, match="MCP cwd 越界"):
        install_module._resolve_mcp_runtime_root(
            plugin_root,
            "../outside",
            ["python", "mcp/run_mcp.py"],
        )


def test_install_accepts_branch_tag_and_commit_refs(tmp_path: Path) -> None:
    repo = tmp_path / "feed"
    repo.mkdir()
    plugin_path = repo / "plugin.py"
    plugin_path.write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n"
        "    marker = 'initial'\n",
        encoding="utf-8",
    )
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
