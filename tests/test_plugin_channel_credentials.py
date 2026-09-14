from pathlib import Path

import pytest

from agent.plugin_composition.config_input import (
    CONFIG_INPUT, load_config, save_config, save_credential, revoke_credential,
)
from agent.plugins.channel_credentials import CoreProviderClientFactory


def _factory(data: Path) -> CoreProviderClientFactory:
    config, revision = load_config(data)
    return CoreProviderClientFactory(data, config, revision)


@pytest.mark.asyncio
async def test_factory_resolves_only_frozen_owner_refs(tmp_path: Path) -> None:
    data = tmp_path / "plugin-data/one"
    ref = save_credential(data, "secret-value")
    other = save_credential(data, "hidden")
    save_config(data, {"token": ref})
    factory = _factory(data)
    client = await factory.create({"arbitrary-plugin-alias": ref})
    assert client.credential(ref) == "secret-value"
    with pytest.raises(RuntimeError, match="frozen plugin"):
        await factory.create({"other": other})
    foreign = tmp_path / "plugin-data/two"
    save_config(foreign, {"token": ref})
    with pytest.raises(FileNotFoundError):
        await _factory(foreign).create({"token": ref})
    await factory.aclose()
    with pytest.raises(RuntimeError, match="已关闭"):
        client.credential(ref)


@pytest.mark.asyncio
async def test_factory_rejects_config_drift_and_revocation(tmp_path: Path) -> None:
    data = tmp_path / "plugin-data/one"
    ref = save_credential(data, "original")
    save_config(data, {"token": ref})
    factory = _factory(data)
    save_config(data, {})
    with pytest.raises(RuntimeError, match="revision 已漂移"):
        await factory.create({"token": ref})
    save_config(data, {"token": ref})
    factory = _factory(data)
    revoke_credential(data, ref)
    with pytest.raises(PermissionError, match="撤销"):
        await factory.create({"token": ref})


@pytest.mark.asyncio
async def test_config_change_after_read_keeps_exact_leased_version(tmp_path, monkeypatch):
    data = tmp_path / "plugin-data/one"
    ref = save_credential(data, "A")
    save_config(data, {"token": ref})
    factory = _factory(data)
    original = Path.read_bytes
    reads = []

    def read_then_change(current):
        content = original(current)
        if current == data / CONFIG_INPUT:
            reads.append(content)
            current.write_text('{"version": 1, "config": ["map", {}]}')
        return content

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    client = await factory.create({"token": ref})
    assert client.credential(ref) == "A"
    assert len(reads) == 1
    with pytest.raises(RuntimeError, match="漂移"):
        await factory.create({"token": ref})
    await factory.aclose()
