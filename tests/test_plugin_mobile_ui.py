from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast
import asyncio

import pytest

from agent.plugins.mobile_ui import (
    MobileUiPluginUnavailable,
    MobileUiRpcExecutionError,
    MobileUiRpcTimeout,
    PluginMobileUiProvider,
)
from agent.plugins.generation import MobileUiAsset
import agent.plugins.mobile_ui as mobile_ui_module


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


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("mapping iteration failed")

    def __len__(self) -> int:
        return 1


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


@pytest.mark.asyncio
async def test_mobile_ui_rpc_timeout_releases_snapshot_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    blocker = asyncio.Event()

    async def never_returns(*args: object, **kwargs: object) -> dict[str, object]:
        await blocker.wait()
        return {}

    cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ].instance.mobile_ui_call = never_returns
    monkeypatch.setattr(mobile_ui_module, "MOBILE_UI_RPC_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(MobileUiRpcTimeout):
        await provider.call(
            "sample@github",
            "recall.current",
            {},
            session_id="mobile:test",
            turn_id="turn-1",
        )


@pytest.mark.asyncio
async def test_mobile_ui_rpc_failure_isolated_from_transport() -> None:
    provider = _provider()

    async def fails(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("plugin bug detail")

    cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ].instance.mobile_ui_call = fails

    with pytest.raises(MobileUiRpcExecutionError, match="sample@github.recall.current"):
        await provider.call(
            "sample@github",
            "recall.current",
            {},
            session_id="mobile:test",
            turn_id="turn-1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    (
        [],
        {1: "value"},
        {"value": object()},
        {"value": "x" * (193 * 1024)},
        _ExplodingMapping(),
    ),
)
async def test_mobile_ui_rpc_invalid_result_isolated_from_transport(
    invalid_result: object,
) -> None:
    provider = _provider()

    async def returns_invalid(*args: object, **kwargs: object) -> object:
        return invalid_result

    cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ].instance.mobile_ui_call = returns_invalid

    with pytest.raises(MobileUiRpcExecutionError, match="sample@github.recall.current"):
        await provider.call(
            "sample@github",
            "recall.current",
            {},
            session_id="mobile:test",
            turn_id="turn-1",
        )
