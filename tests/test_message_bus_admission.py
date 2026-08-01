from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from bus.events import InboundMessage
from bus.queue import MessageBus
from session.manager import SessionManager
from session.store import SessionStore


@pytest.mark.asyncio
async def test_cleanup_retry_keeps_owner_after_worker_drops_item(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:cleanup"
    manager.save(manager.get_or_create(session_key))
    _, admission_id = manager.admit_existing(session_key)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "mobile",
        "device:1",
        "cleanup",
        "hello",
        metadata={"session_key_override": session_key, "client_message_id": "client-1"},
        session_admission_id=admission_id,
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    assert consumed is item
    item_ref = weakref.ref(item)
    original_complete = store.complete_inbound_handoff
    attempts = 0

    def fail_once(handoff_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary delete failure")
        original_complete(handoff_id)

    store.complete_inbound_handoff = fail_once  # type: ignore[method-assign]

    class _Runtime:
        async def wait_thread_available(self, _session_key: str) -> None:
            return None

        async def start_turn(self, _request: object) -> object:
            raise RuntimeError("worker stopped before turn submission")

    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        _Runtime(),  # type: ignore[arg-type]
        SimpleNamespace(session_manager=manager),  # type: ignore[arg-type]
    )
    lane = asyncio.Queue()
    lane.put_nowait(consumed)
    worker._lane_queues[session_key] = lane
    lane_task = asyncio.create_task(worker._run_lane(session_key, lane))
    await lane_task
    assert manager.control_store.list_turns(session_key) == []
    owner_key = id(item)
    assert owner_key in bus._inbound_accepted
    del consumed
    del item
    gc.collect()
    assert item_ref() is not None

    async def cleanup_finished() -> None:
        while store.list_inbound_handoffs():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(cleanup_finished(), timeout=1)
    await asyncio.sleep(0)
    gc.collect()
    assert attempts >= 2
    assert bus._inbound_accepted == {}
    assert bus._inbound_cleanup_tasks == {}
    assert bus.chat_lane._states == {}
    replacement = InboundMessage(
        "mobile",
        "device:1",
        "cleanup",
        "replacement",
        metadata={"session_key_override": session_key, "client_message_id": "client-2"},
        session_admission_id=admission_id,
    )
    await bus.publish_inbound(replacement)
    assert bus._inbound_accepted[id(replacement)].item is replacement
    await bus.complete_inbound(replacement)
    await bus.aclose()
    manager.close()
    store.close()


@pytest.mark.asyncio
async def test_persistent_cleanup_failure_is_bounded_and_shutdown_cancels_retry(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "mobile",
        "device:1",
        "persistent",
        "hello",
        metadata={"client_message_id": "client-persistent"},
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    attempts = 0

    def always_fail(_handoff_id: str) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("persistent delete failure")

    store.complete_inbound_handoff = always_fail  # type: ignore[method-assign]
    with pytest.raises(OSError, match="persistent delete failure"):
        await bus.complete_inbound(consumed)
    await asyncio.sleep(0.35)
    assert 2 <= attempts <= 4
    assert len(bus._inbound_accepted) == 1
    assert len(bus._inbound_cleanup_tasks) == 1
    await bus.aclose()
    assert bus._inbound_cleanup_tasks == {}
    assert store._conn.execute(
        "SELECT 1 FROM inbound_handoffs WHERE handoff_id = ?",
        (consumed.handoff_id,),
    ).fetchone() is not None
    store.close()


@pytest.mark.asyncio
async def test_cleanup_finalize_failure_is_fatal_and_observable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "mobile",
        "device:1",
        "fatal",
        "hello",
        metadata={"client_message_id": "client-fatal"},
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    original_complete = store.complete_inbound_handoff
    failed = True

    def fail_once(handoff_id: str) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise OSError("temporary delete failure")
        original_complete(handoff_id)

    store.complete_inbound_handoff = fail_once  # type: ignore[method-assign]

    async def fatal_finalize(_owner_key: int, _owner: object) -> None:
        raise RuntimeError("owner mismatch")

    bus._finalize_inbound_owner = fatal_finalize  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError, match="temporary delete failure"):
            await bus.complete_inbound(consumed)
        await asyncio.sleep(0.2)
    assert bus._inbound_cleanup_error is not None
    assert bus._inbound_cleanup_tasks == {}
    assert "event=runtime_fatal" in caplog.text
    assert "owner=message_bus.inbound_cleanup" in caplog.text
    with pytest.raises(RuntimeError, match="cleanup owner failed"):
        await bus.aclose()
    store.close()
