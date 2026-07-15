from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugins.mobile_ui import MobileUiPluginUnavailable, PluginMobileUiProvider
from agent.plugins.generation import MobileUiAsset


class _MobilePlugin:
    async def mobile_ui_call(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        return {
            "method": method,
            "payload": payload,
            "session_id": session_id,
            "turn_id": turn_id,
        }


class _Lease:
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot

    async def __aenter__(self):
        return self.snapshot

    async def __aexit__(self, *args: object) -> None:
        return None


class _Store:
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot

    async def acquire(self) -> _Lease:
        return _Lease(self.snapshot)


def _provider() -> PluginMobileUiProvider:
    asset = MobileUiAsset(
        module="export default 1;",
        stylesheet=":host { color: red; }",
        sha256="a" * 64,
    )
    generation = SimpleNamespace(
        plugin_id="sample@github",
        source_revision="revision-1",
        instance=_MobilePlugin(),
        contributions=SimpleNamespace(
            mobile_ui_asset=asset,
        ),
    )
    snapshot = SimpleNamespace(
        generations=MappingProxyType({"sample@github": generation}),
        active_generations=lambda: (generation,),
    )
    manager = SimpleNamespace(current_snapshot=snapshot, snapshot_store=_Store(snapshot))
    return PluginMobileUiProvider(cast(Any, manager))


def test_mobile_ui_catalog_separates_metadata_from_asset() -> None:
    provider = _provider()

    catalog = provider.catalog()
    asset = provider.asset("sample@github")

    assert catalog == [{
        "id": "sample@github",
        "revision": "revision-1",
        "sha256": "a" * 64,
    }]
    assert asset["module"] == "export default 1;"
    assert asset["stylesheet"] == ":host { color: red; }"


@pytest.mark.asyncio
async def test_mobile_ui_rpc_receives_turn_context() -> None:
    provider = _provider()

    result = await provider.call(
        "sample@github",
        "recall.current",
        {"limit": 4},
        session_id="mobile:test",
        turn_id="turn-1",
    )

    assert result == {
        "method": "recall.current",
        "payload": {"limit": 4},
        "session_id": "mobile:test",
        "turn_id": "turn-1",
    }


def test_mobile_ui_rejects_inactive_plugin() -> None:
    provider = _provider()
    cast(Any, provider)._manager.current_snapshot.active_generations = lambda: ()

    with pytest.raises(MobileUiPluginUnavailable, match="sample"):
        provider.asset("sample@github")
