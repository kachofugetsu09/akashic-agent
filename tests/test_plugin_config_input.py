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


def test_unconfigured_plugin_keeps_empty_input_after_writing_business_data(tmp_path: Path):
    data = tmp_path / "plugin-data/example"
    initial = load_config(data)
    data.mkdir(parents=True)
    (data / "business-record").write_text("persisted by plugin")
    assert load_config(data) == initial
    assert not (data / CONFIG_INPUT).exists()


def test_config_reader_and_writer_leave_named_backups_to_owner(tmp_path: Path):
    data = tmp_path / "plugin-data/example"
    data.mkdir(parents=True)
    backup = data / "config.local.toml.bak"
    backup.write_text('token="old-secret"')
    assert load_config(data)[0] == {}
    save_config(data, {"enabled": True})
    assert load_config(data)[0] == {"enabled": True}
    assert backup.read_text() == 'token="old-secret"'


def test_exact_legacy_entry_requires_upgrade_even_with_fixed_input(tmp_path: Path):
    data = tmp_path / "plugin-data/example"
    save_config(data, {})
    (data / "config.local.toml").write_text('token="old-secret"')
    with pytest.raises(RuntimeError, match="升级"):
        load_config(data)
    with pytest.raises(RuntimeError, match="升级"):
        save_config(data, {"enabled": True})
