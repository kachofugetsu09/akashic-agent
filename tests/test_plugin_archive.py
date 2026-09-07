import os
import shutil
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest

from agent.plugins.archive import PluginArchive, encode_config, decode_config
from agent.plugin_composition.channels import CredentialRef
import agent.plugins.archive as archive_module


def source(tmp_path: Path) -> Path:
    path = tmp_path / "installed"
    path.mkdir()
    (path / "plugin.py").write_text("value = 'old'\n")
    (path / "assets").mkdir()
    (path / "assets" / "text.txt").write_text("fixed asset")
    (path / "alias").symlink_to("assets/text.txt")
    return path


def test_archive_survives_source_change_removal_and_new_store(tmp_path):
    original = source(tmp_path)
    archive = PluginArchive(tmp_path / "archive")
    old = archive.save(original)
    assert archive.save(original) == old
    (original / "plugin.py").write_text("value = 'new'\n")
    new = archive.save(original)
    assert new != old
    shutil.rmtree(original)
    reopened = PluginArchive(archive.path)
    assert (reopened.open(old) / "plugin.py").read_text() == "value = 'old'\n"
    assert (reopened.open(new) / "plugin.py").read_text() == "value = 'new'\n"
    assert (reopened.open(old) / "alias").read_text() == "fixed asset"


def test_corrupt_archive_is_not_repaired_or_rebound(tmp_path):
    original = source(tmp_path)
    archive = PluginArchive(tmp_path / "archive")
    identity = archive.save(original)
    module = archive.open(identity) / "plugin.py"
    module.chmod(0o600)
    module.write_text("changed")
    with pytest.raises(RuntimeError, match="损坏"):
        archive.open(identity)
    with pytest.raises(RuntimeError, match="损坏"):
        archive.save(original)
    assert module.read_text() == "changed"
    assert not list(archive.path.glob(".pending-*"))


@pytest.mark.parametrize("kind", ["escape", "absolute", "fifo"])
def test_archive_rejects_nonportable_source_before_publication(tmp_path, kind):
    original = source(tmp_path)
    invalid = original / "invalid"
    if kind == "escape":
        invalid.symlink_to("../outside")
    elif kind == "absolute":
        invalid.symlink_to(original / "plugin.py")
    else:
        os.mkfifo(invalid)
    archive = PluginArchive(tmp_path / "archive")
    with pytest.raises(ValueError):
        archive.save(original)
    assert list(archive.path.iterdir()) == []


def test_failed_final_sync_retains_published_recovery_material(tmp_path, monkeypatch):
    original = source(tmp_path)
    archive = PluginArchive(tmp_path / "archive")
    sync = archive_module._sync_directory

    def fail(path):
        if path == archive.path:
            raise OSError("disk sync failed")
        sync(path)

    monkeypatch.setattr(archive_module, "_sync_directory", fail)
    with pytest.raises(OSError, match="disk sync"):
        archive.save(original)
    (published,) = archive.path.iterdir()
    assert (archive.open(published.name) / "plugin.py").read_bytes() == (
        original / "plugin.py"
    ).read_bytes()
    monkeypatch.setattr(archive_module, "_sync_directory", sync)
    assert archive.save(original) == published.name


def test_source_change_during_copy_is_rejected(tmp_path, monkeypatch):
    original = source(tmp_path)
    archive = PluginArchive(tmp_path / "archive")
    copy = shutil.copytree

    def changed(src, dst, *args, **kwargs):
        if Path(src) == original:
            (original / "plugin.py").write_text("changed during capture")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(archive_module.shutil, "copytree", changed)
    with pytest.raises(RuntimeError, match="发生变化"):
        archive.save(original)
    assert list(archive.path.iterdir()) == []


def test_descriptor_freezes_toml_and_opaque_credentials(tmp_path):
    archive = PluginArchive(tmp_path / "archive")
    config = {
        "day": date(2026, 9, 5),
        "clock": time(12, 0),
        "instant": datetime(2026, 9, 5, tzinfo=timezone.utc),
        "credentials": CredentialRef(("service", "token")),
        "plain_list": ["date", "not a date"],
        "plain_map": {"kind": "credential", "value": "plain"},
    }
    identity = archive.save_descriptor({"config": encode_config(config)})
    read = archive.read_descriptor(identity)
    assert decode_config(read["config"]) == config
    config["day"] = date(2026, 9, 6)
    assert decode_config(archive.read_descriptor(identity)["config"])["day"] == date(
        2026, 9, 5
    )
    with pytest.raises(TypeError):
        read["config"] = {}
    target = archive.path / f"{identity}.json"
    target.write_text("{}")
    with pytest.raises(RuntimeError, match="损坏"):
        archive.read_descriptor(identity)
