from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import agent.plugins.install as install_module
from agent.plugins.install import install_git_plugin


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
