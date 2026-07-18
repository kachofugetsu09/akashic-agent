from __future__ import annotations

from collections.abc import Iterator, Mapping
import asyncio
import hashlib
import threading
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

import agent.plugins.mobile_ui as mobile_ui_module
from agent.plugins.generation import MobileUiAsset
from agent.plugins.mobile_ui import (
    MobileUiPluginUnavailable,
    MobileUiQueryTimeout,
    MobileUiRpcExecutionError,
    MobileUiStaleRevision,
    PluginMobileUiProvider,
)


class _MobilePlugin:
    def mobile_ui_query(
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
    def __init__(self, store: "_Store", snapshot: object) -> None:
        self._store = store
        self.snapshot = snapshot

    async def __aenter__(self):
        self._store.entered += 1
        return self.snapshot

    async def __aexit__(self, *args: object) -> None:
        self._store.exited += 1


class _Store:
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot
        self.entered = 0
        self.exited = 0

    async def acquire(self) -> _Lease:
        return _Lease(self, self.snapshot)


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("mapping iteration failed")

    def __len__(self) -> int:
        return 1


def _provider() -> PluginMobileUiProvider:
    module = "export default 1;"
    stylesheet = ":host { color: red; }"
    asset = MobileUiAsset(
        module=module,
        module_sha256=hashlib.sha256(module.encode()).hexdigest(),
        module_bytes=len(module.encode()),
        stylesheet=stylesheet,
        stylesheet_sha256=hashlib.sha256(stylesheet.encode()).hexdigest(),
        stylesheet_bytes=len(stylesheet.encode()),
        navigation_label="Sample",
        navigation_description="Sample dashboard",
        slots=("turn.after_answer",),
    )
    generation = SimpleNamespace(
        plugin_id="sample@github",
        source_revision="revision-1",
        instance=_MobilePlugin(),
        contributions=SimpleNamespace(mobile_ui_asset=asset),
    )
    snapshot = SimpleNamespace(
        generations=MappingProxyType({"sample@github": generation}),
        active_generations=lambda: (generation,),
    )
    manager = SimpleNamespace(current_snapshot=snapshot, snapshot_store=_Store(snapshot))
    return PluginMobileUiProvider(cast(Any, manager))


def test_mobile_ui_catalog_separates_metadata_from_content_addressed_assets() -> None:
    provider = _provider()

    catalog = provider.catalog()
    item = cast(list[dict[str, object]], catalog["items"])[0]
    module = provider.asset(
        "sample@github",
        "revision-1",
        "module",
        cast(str, item["module_sha256"]),
    )
    stylesheet = provider.asset(
        "sample@github",
        "revision-1",
        "stylesheet",
        cast(str, item["stylesheet_sha256"]),
    )

    assert isinstance(catalog["catalog_revision"], str)
    assert "content" not in item
    assert item["navigation"] == {
        "label": "Sample",
        "description": "Sample dashboard",
    }
    assert item["slots"] == ["turn.after_answer"]
    assert module["content"] == "export default 1;"
    assert stylesheet["content"] == ":host { color: red; }"


@pytest.mark.asyncio
async def test_mobile_ui_query_receives_revision_and_turn_context() -> None:
    provider = _provider()

    result = await provider.query(
        "sample@github",
        "revision-1",
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


@pytest.mark.asyncio
async def test_mobile_ui_sync_query_never_blocks_event_loop() -> None:
    provider = _provider()
    entered = threading.Event()
    release = threading.Event()

    def block(*args: object, **kwargs: object) -> dict[str, object]:
        entered.set()
        release.wait(timeout=1)
        return {"status": "ready"}

    generation = cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ]
    generation.instance.mobile_ui_query = block
    query = asyncio.create_task(
        provider.query(
            "sample@github",
            "revision-1",
            "health.snapshot",
            {},
            session_id="mobile:test",
            turn_id="turn-1",
        )
    )

    assert await asyncio.to_thread(entered.wait, 1)
    heartbeat = asyncio.create_task(asyncio.sleep(0))
    await asyncio.wait_for(heartbeat, timeout=0.1)
    assert not query.done()
    release.set()
    assert await query == {"status": "ready"}


def test_mobile_ui_rejects_inactive_or_stale_plugin_assets() -> None:
    provider = _provider()
    item = cast(list[dict[str, object]], provider.catalog()["items"])[0]

    with pytest.raises(MobileUiStaleRevision, match="sample"):
        provider.asset(
            "sample@github",
            "old-revision",
            "module",
            cast(str, item["module_sha256"]),
        )

    cast(Any, provider)._manager.current_snapshot.active_generations = lambda: ()
    with pytest.raises(MobileUiPluginUnavailable, match="sample"):
        provider.asset(
            "sample@github",
            "revision-1",
            "module",
            cast(str, item["module_sha256"]),
        )


@pytest.mark.asyncio
async def test_mobile_ui_timeout_keeps_snapshot_lease_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    blocker = threading.Event()

    def block(*args: object, **kwargs: object) -> dict[str, object]:
        blocker.wait(timeout=1)
        return {}

    generation = cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ]
    generation.instance.mobile_ui_query = block
    store = cast(Any, provider)._manager.snapshot_store
    monkeypatch.setattr(mobile_ui_module, "MOBILE_UI_QUERY_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(MobileUiQueryTimeout):
        await provider.query(
            "sample@github",
            "revision-1",
            "recall.current",
            {},
            session_id="mobile:test",
            turn_id="turn-1",
        )

    assert store.entered == 1
    assert store.exited == 0
    blocker.set()
    for _ in range(20):
        if store.exited == 1:
            break
        await asyncio.sleep(0.01)
    assert store.exited == 1

@pytest.mark.asyncio
async def test_mobile_ui_rpc_failure_isolated_from_transport() -> None:
    provider = _provider()

    def fails(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("plugin bug detail")

    cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ].instance.mobile_ui_query = fails

    with pytest.raises(MobileUiRpcExecutionError, match="sample@github.recall.current"):
        await provider.query(
            "sample@github",
            "revision-1",
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

    def returns_invalid(*args: object, **kwargs: object) -> object:
        return invalid_result

    cast(Any, provider)._manager.current_snapshot.generations[
        "sample@github"
    ].instance.mobile_ui_query = returns_invalid

    with pytest.raises(MobileUiRpcExecutionError, match="sample@github.recall.current"):
        await provider.query(
            "sample@github",
            "revision-1",
            "recall.current",
            {},
            session_id="mobile:test",
            turn_id="turn-1",
        )
