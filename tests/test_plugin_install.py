from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import agent.plugins.install as install_module
from agent.plugins.install import install_git_plugin


def test_install_git_plugin_installs_into_cache_and_preserves_data(tmp_path: Path) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / ".aka-plugin").mkdir(parents=True)
    (repo / "skills" / "feed-manage").mkdir(parents=True)
    (repo / "skills" / "feed-manage" / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed\n---\nbody\n",
        encoding="utf-8",
    )
    (repo / ".aka-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "feed",
                "version": "0.1.0",
                "description": "feed plugin",
                "paths": {"skills": ["skills"]},
                "akashic": {"runtime": {"supports": ["skills"]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    home = tmp_path / "plugins-home"
    data_dir = home / "data" / "feed-lab"
    data_dir.mkdir(parents=True)
    (data_dir / "state.json").write_text('{"keep":true}\n', encoding="utf-8")

    result = install_git_plugin(
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    assert result.plugin_name == "feed"
    assert result.installed_path == home / "cache" / "lab" / "feed" / "0.1.0"
    assert (result.installed_path / ".aka-plugin" / "plugin.json").exists()
    assert (result.installed_path / "skills" / "feed-manage" / "SKILL.md").exists()
    assert (result.data_path / "state.json").read_text(encoding="utf-8").strip() == '{"keep":true}'
    registry = json.loads((home / "registry.json").read_text(encoding="utf-8"))
    entry = registry["plugins"]["feed@lab"]
    assert entry["plugin_id"] == "feed@lab"
    assert entry["install_source"] == str(repo)
    assert entry["skills"] == ["feed-manage"]
    assert entry["active"] is False


def test_install_git_plugin_prepares_mcp_venv_and_rewrites_python_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / ".aka-plugin").mkdir(parents=True)
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "run_mcp.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "mcp" / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (repo / "mcp" / "servers.json").write_text(
        json.dumps(
            {
                "servers": {
                    "feed": {
                        "command": ["python", "mcp/run_mcp.py"],
                        "cwd": ".",
                        "env": {},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (repo / ".aka-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "feed",
                "version": "0.1.0",
                "description": "feed plugin",
                "paths": {"mcp_servers": ["mcp/servers.json"]},
                "akashic": {"runtime": {"supports": ["mcp"]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    calls: list[tuple[str, Path]] = []

    def _fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        calls.append((label, cwd))
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(install_module, "_run_command", _fake_run)

    result = install_git_plugin(
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    servers = json.loads(
        (result.installed_path / "mcp" / "servers.json").read_text(encoding="utf-8")
    )["servers"]
    expected_python = install_module._venv_python_path(
        result.installed_path / "mcp" / ".venv"
    )

    assert servers["feed"]["command"][0] == str(expected_python)
    assert servers["feed"]["env"]["AKA_PLUGIN_DATA_DIR"] == str(result.data_path)
    assert calls == [
        ("feed venv", result.installed_path / "mcp"),
        ("feed pip install", result.installed_path / "mcp"),
    ]
    registry = json.loads(
        ((tmp_path / "plugins-home") / "registry.json").read_text(encoding="utf-8")
    )
    assert registry["plugins"]["feed@lab"]["mcp_servers"] == ["feed"]


def _run_git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode == 0:
        return
    raise AssertionError(result.stderr or result.stdout)
