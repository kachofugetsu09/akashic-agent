from __future__ import annotations

import asyncio
import hashlib
import shutil
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import agent.plugins.manager as plugin_manager_module
from agent.control.timer import TimerReceipt, TimerStatus
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.timers import PluginTimers
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.content.plugin import ContentSourceServices, ContentWakeServices
from plugins.content.store import ContentStore
from tests.fixtures.content_clock_source.plugin import (
    BoundContentSource,
    FixtureSourceStore,
    SourceRuntime,
)

CONTENT_SOURCE = ServiceKey[ContentSourceServices]("content.source.v1")
CONTENT_WAKE = ServiceKey[ContentWakeServices]("content.wake.v1")


class _TimerHandle:
    def __init__(self, timer_id: str, deadline: datetime, now: datetime) -> None:
        self._id = timer_id
        self.deadline = deadline
        self.now = now
        self.future: asyncio.Future[TimerReceipt] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def id(self) -> str:
        return self._id

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self.future)

    async def cancel(self) -> TimerReceipt:
        if not self.future.done():
            self.future.set_result(self._receipt(TimerStatus.CANCELLED))
        return await self.future

    async def cleanup(self) -> None:
        _ = await self.cancel()

    def fire(self) -> None:
        self.future.set_result(self._receipt(TimerStatus.FIRED))

    def _receipt(self, status: TimerStatus) -> TimerReceipt:
        return TimerReceipt(self.id, self.deadline, self.now, status)


class _Timer:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.handles: list[_TimerHandle] = []

    def schedule(self, deadline: datetime) -> _TimerHandle:
        handle = _TimerHandle(f"timer:{len(self.handles)}", deadline, self.now)
        self.handles.append(handle)
        return handle


async def _eventually(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


def _copy_plugins(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    content = tmp_path / "plugins" / "content"
    source = tmp_path / "plugins" / "content_clock_source"
    shutil.copytree(root / "plugins" / "content", content)
    shutil.copytree(
        root / "tests" / "fixtures" / "content_clock_source",
        source,
    )
    return content, source


def _sqlite_hashes(path: Path) -> dict[str, str]:
    return {
        candidate.name: hashlib.sha256(candidate.read_bytes()).hexdigest()
        for candidate in sorted(path.parent.glob(path.name + "*"))
    }


@pytest.mark.asyncio
async def test_real_v3_loader_timer_and_stores_submit_before_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    timer = _Timer(now)
    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", lambda: timer)
    content_dir, source_dir = _copy_plugins(tmp_path)
    workspace = tmp_path / "workspace"
    source_store = FixtureSourceStore(
        workspace / "plugin-data" / "content_clock_source-builtin" / "source.sqlite3"
    )
    source_store.seed(({"kind": "sleep", "score": 92},), now)
    manager = PluginManager(
        plugin_dirs=[content_dir, source_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: len(timer.handles) == 1)
        timer.handles[0].fire()
        await _eventually(lambda: source_store.state(now)["cursor"] == 1)

        content_store = ContentStore(
            workspace / "plugin-data" / "content-builtin" / "content.sqlite3"
        )
        assert content_store.state_counts() == {"pending": 1}
        assert source_store.state(now)["poll_count"] == 1
        assert len(timer.handles) == 2
        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        wake = snapshot.composition_root.context.require(CONTENT_WAKE)
        assert wake.snapshot(now)["wake_needed"] is True
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_submit_crash_before_cursor_repolls_without_duplicate_content(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    source_store = FixtureSourceStore(tmp_path / "source.sqlite3")
    source_store.seed(({"kind": "feed"},), now)
    content_store = ContentStore(tmp_path / "content.sqlite3")

    class Bound:
        def submit(self, batch_id, items):
            return content_store.submit("clock-feed", batch_id, items)

    first_timer = _Timer(now)

    def crash() -> None:
        raise RuntimeError("crash after submit")

    first = SourceRuntime(
        source_store,
        PluginTimers(first_timer),
        cast(BoundContentSource, Bound()),
        now=lambda: now,
        after_submit=crash,
    )
    await first.start()
    task = first._task
    assert task is not None
    first_timer.handles[0].fire()
    with pytest.raises(RuntimeError, match="crash after submit"):
        await task

    assert source_store.state(now)["cursor"] == 0
    assert content_store.state_counts() == {"pending": 1}

    second_timer = _Timer(now)
    second = SourceRuntime(
        source_store,
        PluginTimers(second_timer),
        cast(BoundContentSource, Bound()),
        now=lambda: now,
    )
    await second.start()
    second_timer.handles[0].fire()
    await _eventually(lambda: source_store.state(now)["cursor"] == 1)

    assert content_store.state_counts() == {"pending": 1}
    assert source_store.state(now)["poll_count"] == 1
    await second.close()


@pytest.mark.asyncio
async def test_candidate_root_has_no_timer_poll_or_formal_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    timers: list[_Timer] = []

    def timer_factory() -> _Timer:
        timer = _Timer(now)
        timers.append(timer)
        return timer

    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", timer_factory)
    content_dir, source_dir = _copy_plugins(tmp_path)
    workspace = tmp_path / "workspace"
    source_store = FixtureSourceStore(
        workspace / "plugin-data" / "content_clock_source-builtin" / "source.sqlite3"
    )
    source_store.seed(({"kind": "calendar"},), now + timedelta(hours=1))
    manager = PluginManager(
        plugin_dirs=[content_dir, source_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: sum(len(timer.handles) for timer in timers) == 1)
        before = source_store.state(now)
        content_path = workspace / "plugin-data" / "content-builtin" / "content.sqlite3"
        assert content_path.is_file()
        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        wake = snapshot.composition_root.context.require(CONTENT_WAKE)
        content = snapshot.composition_root.context.require(CONTENT_SOURCE).bind(
            "candidate-probe"
        )
        receipt = content.submit(
            "poll:1",
            (
                {
                    "item_id": "candidate-row",
                    "revision": "1",
                    "payload": {"kind": "candidate"},
                    "not_before": now,
                },
            ),
        )
        assert receipt["high_watermark"] == 1
        frozen = wake.snapshot(now)
        accepted = {
            "session_id": "wake:candidate",
            "turn_id": "turn:accepted",
        }
        selected = wake.select(
            frozen["items"][0]["ref"], frozen["snapshot_seq"], accepted, now
        )
        assert selected["selected"] is True
        formal_hashes = _sqlite_hashes(content_path)

        with (source_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# candidate fixture revision\n")
        candidate = await manager.prepare_candidate("content_clock_source")

        assert candidate is not None
        assert candidate.runtime_snapshot is not None
        candidate_content = candidate.runtime_snapshot.generations["content"]
        candidate_path = candidate_content.data_dir / "content.sqlite3"
        candidate_store = ContentStore(candidate_path)
        candidate_store.initialize()
        recovered = candidate_store.selection(accepted)
        assert recovered is not None
        assert recovered["selection_token"] == selected["selection_token"]
        assert "settlement_ref" not in recovered
        assert candidate_store.state_counts() == {"selected": 1}
        assert sum(len(timer.handles) for timer in timers) == 1
        assert source_store.state(now) == before
        assert _sqlite_hashes(content_path) == formal_hashes
        await manager.discard_prepared("content_clock_source")
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_promotion_drains_old_wait_and_only_new_root_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    timers: list[_Timer] = []

    def timer_factory() -> _Timer:
        timer = _Timer(now)
        timers.append(timer)
        return timer

    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", timer_factory)
    content_dir, source_dir = _copy_plugins(tmp_path)
    workspace = tmp_path / "workspace"
    source_store = FixtureSourceStore(
        workspace / "plugin-data" / "content_clock_source-builtin" / "source.sqlite3"
    )
    source_store.seed(({"kind": "future"},), now + timedelta(hours=1))
    manager = PluginManager(
        plugin_dirs=[content_dir, source_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: sum(len(timer.handles) for timer in timers) == 1)
        old_handle = next(timer.handles[0] for timer in timers if timer.handles)

        with (source_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# promoted fixture revision\n")
        assert await manager.prepare_candidate("content_clock_source") is not None
        result = await manager.publish_prepared("content_clock_source")

        assert result["publication_state"] == "committed"
        await _eventually(
            lambda: sum(
                not handle.future.done() for timer in timers for handle in timer.handles
            )
            == 1
        )
        assert old_handle.future.result().status is TimerStatus.CANCELLED
        assert source_store.state(now)["poll_count"] == 0
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_clone_stays_readable_during_concurrent_submit(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    content_dir, source_dir = _copy_plugins(tmp_path)
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[content_dir, source_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    source = snapshot.composition_root.context.require(CONTENT_SOURCE).bind(
        "clone-stress"
    )
    started = threading.Event()
    stop = threading.Event()
    written = 0

    def submit_until_stopped() -> None:
        nonlocal written
        while not stop.is_set() and written < 2_000:
            sequence = written + 1
            _ = source.submit(
                f"poll:{sequence}",
                (
                    {
                        "item_id": f"item-{sequence}",
                        "revision": "1",
                        "payload": {"sequence": sequence},
                        "not_before": now,
                    },
                ),
            )
            written = sequence
            started.set()
            time.sleep(0.0005)

    writer = asyncio.create_task(asyncio.to_thread(submit_until_stopped))
    candidate = None
    try:
        await asyncio.to_thread(started.wait)
        with (source_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# concurrent clone fixture revision\n")
        candidate = await manager.prepare_candidate("content_clock_source")
        assert candidate is not None and candidate.runtime_snapshot is not None
    finally:
        stop.set()
        await writer

    try:
        candidate_path = (
            candidate.runtime_snapshot.generations["content"].data_dir
            / "content.sqlite3"
        )
        candidate_store = ContentStore(candidate_path)
        candidate_store.initialize()
        candidate_count = sum(candidate_store.state_counts().values())
        formal_store = ContentStore(
            workspace / "plugin-data" / "content-builtin" / "content.sqlite3"
        )

        assert 1 <= candidate_count <= written
        assert sum(formal_store.state_counts().values()) == written
    finally:
        if candidate is not None:
            await manager.discard_prepared("content_clock_source")
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_readiness_rejects_unknown_content_schema(
    tmp_path: Path,
) -> None:
    content_dir, source_dir = _copy_plugins(tmp_path)
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[content_dir, source_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    content_path = workspace / "plugin-data" / "content-builtin" / "content.sqlite3"
    connection = sqlite3.connect(content_path)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    try:
        with (source_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# invalid clone fixture revision\n")

        assert await manager.prepare_candidate("content_clock_source") is None
        assert manager.current_snapshot is not None
    finally:
        connection = sqlite3.connect(content_path)
        connection.execute("PRAGMA user_version = 1")
        connection.close()
        await manager.terminate_all()


def test_fixture_declares_its_own_structural_content_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "tests" / "fixtures" / "content_clock_source" / "plugin.py"
    ).read_text(encoding="utf-8")

    assert "from plugins.content" not in source
    assert 'ServiceKey[ContentSourceServices]("content.source.v1")' in source
    assert "SCOPED_TURNS" not in source
    assert "DELIVERIES" not in source
    assert "MCP_SERVERS" not in source
