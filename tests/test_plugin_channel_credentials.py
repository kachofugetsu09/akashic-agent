from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.plugin_composition import CredentialRef
from agent.plugins.channel_credentials import CoreProviderClientFactory


def _revision(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve(strict=False)).encode())
    digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


@pytest.mark.asyncio
async def test_core_provider_factory_resolves_only_frozen_formal_refs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.local.toml"
    path.write_text("appSecret = 'secret-value'\nother = 'hidden'\n", encoding="utf-8")
    factory = CoreProviderClientFactory(
        path,
        ("appSecret",),
        _revision(path),
    )
    ref = CredentialRef(("appSecret",))

    client = await factory.create({"appSecret": ref})

    assert client.credential(ref) == "secret-value"
    with pytest.raises(RuntimeError, match="不属于当前 provider client"):
        client.credential(CredentialRef(("other",)))
    await factory.aclose()
    with pytest.raises(RuntimeError, match="已关闭"):
        client.credential(ref)


@pytest.mark.asyncio
async def test_core_provider_factory_rejects_raw_config_drift_before_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.local.toml"
    path.write_text("token = 'old'\n", encoding="utf-8")
    factory = CoreProviderClientFactory(path, ("token",), _revision(path))
    path.write_text("token = 'new'\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="revision 已漂移"):
        await factory.create({"token": CredentialRef(("token",))})
