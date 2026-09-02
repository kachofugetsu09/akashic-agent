import tomllib
from pathlib import Path

from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    materialize_static_command,
)


def test_replay_debug_manifest_binds_artifact_python_runtime() -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "docker"
        / "debug"
        / "plugins"
        / "replay_debug"
        / "akashic.plugin.toml"
    )
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))

    command = manifest["mcp"][0]["command"]

    assert command == ["python", "replay_mcp.py"]
    assert manifest["python"] == [{"requirements": "requirements.txt"}]


def test_replay_debug_materializes_its_staged_interpreter(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "docker"
        / "debug"
        / "plugins"
        / "replay_debug"
    )
    artifact = tmp_path / "replay_debug"
    artifact.mkdir()
    for name in (
        "akashic.plugin.toml",
        "plugin.py",
        "replay_mcp.py",
        "requirements.txt",
    ):
        (artifact / name).write_bytes((source / name).read_bytes())
    interpreter = artifact / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)

    manifest = load_static_plugin_manifest(artifact)
    command = materialize_static_command(
        artifact,
        manifest,
        manifest.mcp_servers[0],
    )

    assert command == (str(interpreter), "replay_mcp.py")
