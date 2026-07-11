from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import agent.plugins.install as install_module
from agent.plugins.install import (
    install_git_plugin,
    set_installed_plugin_enabled,
    uninstall_plugin,
)


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
    data_dir = home / "data" / "feed-lab"
    data_dir.mkdir(parents=True)
    (data_dir / "state.json").write_text('{"keep":true}\n', encoding="utf-8")

    result = install_git_plugin(source=str(repo), marketplace="lab", plugins_home=home)

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
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    assert calls == [
        ("feed venv", result.installed_path / "mcp"),
        ("feed pip install", result.installed_path / "mcp"),
    ]
    assert not (result.installed_path / "mcp" / "servers.json").exists()


def test_plugin_enable_disable_and_uninstall_preserve_data(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    cache = home / "cache" / "github" / "fitbit" / "1.0.0"
    data = home / "data" / "fitbit-github"
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


def test_uninstall_converges_when_cache_is_already_missing(tmp_path: Path) -> None:
    home = tmp_path / "plugins-home"
    data = home / "data" / "feed-github"
    data.mkdir(parents=True)
    state = data / "state.db"
    state.write_bytes(b"keep")
    (home / "manifest.toml").write_text(
        '[plugins."feed@github"]\nenabled = true\n',
        encoding="utf-8",
    )

    cache, retained = uninstall_plugin("feed@github", plugins_home=home)

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
