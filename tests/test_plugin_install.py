from __future__ import annotations
import os
import subprocess
import tomllib
from pathlib import Path
from agent.plugins.install import finalize_uninstall_plugin, set_installed_plugin_enabled

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

    set_installed_plugin_enabled(
        "fitbit@github",
        enabled=False,
        plugins_home=home,
    )
    wait_until_disabled("fitbit@github")
    removed_cache, retained_data = finalize_uninstall_plugin(
        "fitbit@github",
        workspace=workspace,
        plugins_home=home,
    )

    assert disabled_before_removal
    assert removed_cache == home / "cache" / "github" / "fitbit"
    assert not removed_cache.exists()
    assert retained_data == data
    assert state.read_bytes() == b"model"
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
