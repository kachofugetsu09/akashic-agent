"""EventMail source ownership across local Fiber replacement."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest

from agent.plugin_composition import CompositionRoot, Context
from plugins.eventmail.plugin import (
    EVENTMAIL_CONTENT_SOURCE,
    _AlertSourceServices,
    _ContextSourceServices,
    _SourceServices,
)
from plugins.eventmail.store import EventMailStore


def test_source_ids_rebind_without_losing_mail_or_reviving_old_handles(tmp_path):
    """All three kinds retain mail while a closed source yields its ID."""
    store = EventMailStore(tmp_path / "eventmail.sqlite3")
    store.initialize()
    changed: list[str] = []
    content = _SourceServices(store, lambda: changed.append("content"))
    alerts = _AlertSourceServices(store, lambda: changed.append("alert"))
    contexts = _ContextSourceServices(store, lambda: changed.append("context"))
    now = datetime.now(UTC)

    old_content = content.bind("feed")
    old_alert = alerts.bind("calendar")
    old_context = contexts.bind("steam")
    with pytest.raises(RuntimeError, match="已有 owner"):
        content.bind("feed")
    with pytest.raises(RuntimeError, match="已有 owner"):
        alerts.bind("calendar")
    with pytest.raises(RuntimeError, match="已有 owner"):
        contexts.bind("steam")

    old_content.submit("first", [{
        "item_id": "one", "revision": "1", "not_before": now,
        "requires_ack": True, "payload": {"title": "old"},
    }])
    old_alert.report(event_id="one", payload={"title": "old"}, observed_at=now)
    old_context.report(event_id="one", payload={"title": "old"}, observed_at=now)
    for bound in (old_content, old_alert, old_context):
        bound.close()
        bound.close()

    with pytest.raises(RuntimeError, match="已关闭"):
        old_content.read_submission("first")
    with pytest.raises(RuntimeError, match="已关闭"):
        old_alert.status(event_id="one")
    with pytest.raises(RuntimeError, match="已关闭"):
        old_context.report(event_id="stale", payload={}, observed_at=now)

    new_content = content.bind("feed")
    new_alert = alerts.bind("calendar")
    new_context = contexts.bind("steam")
    assert new_content.read_submission("first") is not None
    assert new_alert.status(event_id="one") == "pending"
    new_content.submit("second", [{
        "item_id": "two", "revision": "1", "not_before": now,
        "requires_ack": True, "payload": {"title": "new"},
    }])
    new_alert.report(event_id="two", payload={"title": "new"}, observed_at=now)
    new_context.report(event_id="two", payload={"title": "new"}, observed_at=now)
    with closing(sqlite3.connect(tmp_path / "eventmail.sqlite3")) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute(
            "SELECT kind, COUNT(*) FROM mail_envelopes GROUP BY kind ORDER BY kind"
        ).fetchall() == [("alert", 2), ("content", 2), ("context", 2)]
    assert len(changed) == 6


@pytest.mark.asyncio
async def test_old_source_call_drains_before_its_binding_is_released(tmp_path):
    """A local source stop keeps the old ID until its admitted call ends."""
    root = CompositionRoot("eventmail-sources")
    root_token = root.instance_token
    store = EventMailStore(tmp_path / "eventmail.sqlite3")
    store.initialize()
    sources = _SourceServices(store, lambda: None)
    await root.context.provide(EVENTMAIL_CONTENT_SOURCE, sources)
    bound = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def apply_source(ctx: Context) -> None:
        source = ctx.require(EVENTMAIL_CONTENT_SOURCE).bind("feed")
        bound.append(source)
        await ctx.effect(lambda: source.close, label="feed-binding")

    async def apply_peer(_ctx: Context) -> None:
        return None

    fiber = await root.mount(
        apply_source, name="feed-source", inject=(EVENTMAIL_CONTENT_SOURCE,)
    )
    peer = await root.mount(apply_peer, name="unrelated")

    async def old_call() -> None:
        async with fiber.context.runtime_scope():
            entered.set()
            await release.wait()
            bound[0].submit("old", [{
                "item_id": "one", "revision": "1", "not_before": datetime.now(UTC),
                "requires_ack": True, "payload": {"title": "held"},
            }])

    call_task = asyncio.create_task(old_call())
    close_task = None
    try:
        await entered.wait()
        close_task = asyncio.create_task(fiber.dispose())
        await fiber._admission_closed.wait()  # The old activation has stopped new calls.
        with pytest.raises(RuntimeError, match="已有 owner"):
            sources.bind("feed")
        assert not close_task.done()
        assert root.instance_token is root_token
        assert peer.state.value == "active"

        release.set()
        await asyncio.gather(call_task, close_task)
        with pytest.raises(RuntimeError, match="已关闭"):
            bound[0].read_submission("old")
        current = sources.bind("feed")
        assert current.read_submission("old") is not None
        current.close()
        assert root.instance_token is root_token
        assert peer.state.value == "active"
    finally:
        release.set()
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await asyncio.gather(call_task, return_exceptions=True)
        await root.dispose()
