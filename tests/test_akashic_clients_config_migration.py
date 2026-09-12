from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from agent.migrations.runner import MigrationRunner
from plugins.akashic_clients.config import AkashicClientsConfig


def _runner(root: Path, config: Path, workspace: Path) -> MigrationRunner:
    repo = root / "repo"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text(
        "schema_version = 1\nmigrations = []\n", encoding="utf-8"
    )
    return MigrationRunner(
        repo_root=repo,
        config_path=config,
        workspace=workspace,
        plugin_dirs=(Path(__file__).parents[1] / "plugins/akashic_clients",),
        installed_cache_root=root / "empty-plugin-cache",
    )


def test_akashic_clients_bundle_moves_legacy_config_before_startup(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    source = """\
[runtime]
workspace = "workspace"

[channels.chat]
enabled = false
channel_name = "web"

[mobile_realtime]
enabled = true
host = "127.0.0.1"
port = 6324
database = "data/mobile_realtime.db"
public_url = "wss://example.test/ws"

[mobile_realtime.key_encryption]
provider = "file"
master_key_file = "data/mobile/master-keys.json"
keyset_manifest = "data/mobile/keys/current.json"
"""
    config.write_text(source, encoding="utf-8")
    workspace = tmp_path / "workspace"
    runner = _runner(tmp_path, config, workspace)

    outcome = runner.run()

    assert outcome.migrations == ("20260913_01_akashic_clients_config",)
    target = workspace / "plugin-data/akashic_clients-builtin/config.local.toml"
    values = tomllib.loads(target.read_text(encoding="utf-8"))
    validated = AkashicClientsConfig.model_validate(values)
    assert validated.mobile_realtime.port == 6324
    assert values["enabled"] is True
    assert values["web"] == {"enabled": False}
    assert values["mobile_realtime"]["database"] == "data/mobile_realtime.db"
    assert values["mobile_realtime"]["key_encryption"]["provider"] == "file"
    assert "channels" not in tomllib.loads(config.read_text(encoding="utf-8"))
    assert "mobile_realtime" not in tomllib.loads(config.read_text(encoding="utf-8"))
    assert config.with_name(config.name + ".before-akashic-clients-plugin-migration.bak").read_text(
        encoding="utf-8"
    ) == source

    assert runner.run().state == "current"


def test_akashic_clients_bundle_refuses_conflicting_target(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[channels.chat]\nenabled = true\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "plugin-data/akashic_clients-builtin/config.local.toml"
    target.parent.mkdir(parents=True)
    target.write_text("enabled = false\n", encoding="utf-8")
    runner = _runner(tmp_path, config, workspace)

    with pytest.raises(RuntimeError, match="插件配置已存在且内容不同"):
        runner.run()

    assert "channels.chat" in config.read_text(encoding="utf-8")
    assert not config.with_name(config.name + ".before-akashic-clients-plugin-migration.bak").exists()
