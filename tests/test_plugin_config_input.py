"""固定配置的持久化边界；不依赖 runtime 或外部 provider。"""
from pathlib import Path
import tomllib

import pytest

from agent.plugin_composition.config_input import (
    CONFIG_INPUT, load_config, save_config, save_credential, upgrade_config,
)
from agent.plugins.manager import _copy_validation_tree


def test_explicit_upgrade_keeps_plaintext_only_in_private_recovery(tmp_path: Path):
    data = tmp_path / "plugin-data/example"
    data.mkdir(parents=True)
    original = b'token="old-secret"\nenabled=true\n'
    (data / "config.local.toml").write_bytes(original)
    (data / "config.local.toml.before-setup.bak").write_bytes(original)
    with pytest.raises(RuntimeError, match="升级"):
        load_config(data)

    def convert(content):
        values = tomllib.loads(content.decode())
        values["token"] = save_credential(data, values["token"])
        return values

    backup = upgrade_config(data, convert)
    assert (backup / "original/config.local.toml").read_bytes() == original
    assert (backup / "retired/config.local.toml.before-setup.bak").read_bytes() == original
    assert b"old-secret" not in (data / CONFIG_INPUT).read_bytes()
    target = tmp_path / "candidate/plugin-data/example"
    _copy_validation_tree(data, target, ())
    assert load_config(target)[0] == load_config(data)[0]
    assert not (tmp_path / "candidate/.plugin-credentials").exists()


def test_candidate_refuses_unknown_nonempty_data_and_legacy_backups(tmp_path: Path):
    data = tmp_path / "plugin-data/example"
    data.mkdir(parents=True)
    (data / "notes").write_text("unknown old data")
    target = tmp_path / "candidate/plugin-data/example"
    with pytest.raises(RuntimeError, match="缺少固定配置"):
        _copy_validation_tree(data, target, ())
    save_config(data, {})
    (data / "config.local.toml.bak").write_text('token="old-secret"')
    with pytest.raises(RuntimeError, match="升级"):
        _copy_validation_tree(data, target, ("config.local.toml.bak",))
    assert not target.exists()
