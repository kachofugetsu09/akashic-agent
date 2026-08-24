from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from agent.control.models import TurnItem, TurnItemKind, TurnStatus
from agent.control.scoped_turn import DurableTurnView, TurnAcceptedReceipt
from agent.plugin_composition import PluginScopedTurns, PluginTimers
from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from plugins.content.plugin import (
    ContentDeliveryServices,
    _DeliveryServices,
    _WakeServices as _ContentWakeServices,
)
from plugins.content.store import ContentStore
from plugins.drift.plugin import (
    DriftDeliveryServices,
    _DeliveryServices as _DriftDeliveryServices,
    _WakeServices as _DriftWakeServices,
)
from plugins.drift.store import DriftStore
from plugins.wake.plugin import DeliveryTarget, DriftWakeServices, WakeRuntime
from session.manager import SessionManager


class _NoTimers:
    def schedule(self, _deadline):
        raise AssertionError("ready delivery reconciliation must not schedule a timer")


class _Turns:
    def __init__(
        self, accepted: TurnAcceptedReceipt, response: str, *, legacy: bool = False
    ) -> None:
        self.accepted = accepted
        self.response = response
        self.legacy = legacy

    def read(self, accepted: TurnAcceptedReceipt) -> DurableTurnView:
        assert accepted == self.accepted
        return DurableTurnView(
            accepted.session_id,
            accepted.turn_id,
            TurnStatus.COMPLETED,
            self.response,
            None,
            None,
            None,
            (
                TurnItem(
                    TurnItemKind.TOOL_CALL,
                    "item:share",
                    {
                        "callId": "call:share",
                        "name": "share_content",
                        "status": "success",
                        "arguments": (
                            {"message": self.response}
                            if self.legacy
                            else {"message": self.response, "items": []}
                        ),
                        "resultPreview": '{"recorded":true}',
                    },
                ),
            ),
        )


class _NoDrift:
    def snapshot(self, _now):
        return {"next_due": None, "proposals": ()}

    def selected(self, _limit: int = 100):
        return ()


def _ready_content(
    path: Path,
    *,
    requires_ack: bool = False,
) -> tuple[ContentStore, TurnAcceptedReceipt, str]:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    accepted = TurnAcceptedReceipt("wake:default", "turn:delivery")
    store = ContentStore(path)
    _ = store.submit(
        "fitbit",
        "poll:delivery",
        (
            {
                "item_id": "sleep:delivery",
                "revision": "1",
                "payload": {"kind": "sleep"},
                "not_before": now,
                "requires_ack": requires_ack,
            },
        ),
    )
    snapshot = store.snapshot(now)
    selected = store.select(
        snapshot["items"][0]["ref"],
        snapshot["snapshot_seq"],
        {"session_id": accepted.session_id, "turn_id": accepted.turn_id},
        now,
    )
    token = selected["selection_token"]
    assert isinstance(token, str)
    assert store.transition(token, "ready_for_delivery")["changed"] is True
    return store, accepted, token


def _runtime(
    accepted: TurnAcceptedReceipt,
    content: ContentStore,
    deliveries: PluginDurableDeliveries,
    *,
    response: str = "Wake says hello",
    legacy: bool = False,
) -> WakeRuntime:
    return WakeRuntime(
        cast(PluginTimers, _NoTimers()),
        cast(PluginScopedTurns, _Turns(accepted, response, legacy=legacy)),
        _ContentWakeServices(content),
        cast(DriftWakeServices, _NoDrift()),
        deliveries=deliveries,
        content_delivery=cast(ContentDeliveryServices, _DeliveryServices(content)),
        drift_delivery=cast(
            DriftDeliveryServices,
            _DriftDeliveryServices(DriftStore(content.path.with_name("drift.sqlite3"))),
        ),
        target=DeliveryTarget(
            channel="recording",
            recipient="recipient:one",
            session_id="recipient-session",
        ),
    )


@pytest.mark.asyncio
async def test_drift_share_uses_same_provider_session_and_settlement_chain(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    accepted = TurnAcceptedReceipt("wake:default", "turn:drift-delivery")
    content = ContentStore(tmp_path / "content.sqlite3")
    content.initialize()
    drift = DriftStore(tmp_path / "drift.sqlite3")
    drift.initialize()
    drift.propose("reflection", "1", {"kind": "drift"}, now)
    proposal = drift.snapshot(now)["proposals"][0]
    selected = drift.select(
        proposal["ref"],
        {"session_id": accepted.session_id, "turn_id": accepted.turn_id},
        now,
    )
    token = selected["selection_token"]
    assert isinstance(token, str)
    assert drift.transition(token, "ready_for_delivery")["changed"] is True
    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    sessions = SessionManager(tmp_path / "workspace")
    provider_calls: list[str] = []

    async def sender(request, provider_started):
        provider_started(
            DurableBindingAttempt(
                request.logical_delivery_id,
                "snapshot:recording",
                "generation:recording",
                "binding:recording",
            )
        )
        provider_calls.append(request.body)
        return ChannelDeliveryReceipt(
            request.logical_delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider:recording",),
        )

    async def project(request) -> str:
        return await sessions.append_durable_delivery(
            session_key=request.projection_session_id,
            content=request.body,
            delivery_id=request.logical_delivery_id,
            control_turn_id=request.accepted_turn.turn_id,
        )

    deliveries = PluginDurableDeliveries(ledger, sender, project)
    runtime = WakeRuntime(
        cast(PluginTimers, _NoTimers()),
        cast(PluginScopedTurns, _Turns(accepted, "drift thought")),
        _ContentWakeServices(content),
        _DriftWakeServices(drift),
        deliveries=deliveries,
        content_delivery=cast(ContentDeliveryServices, _DeliveryServices(content)),
        drift_delivery=cast(
            DriftDeliveryServices,
            _DriftDeliveryServices(drift),
        ),
        target=DeliveryTarget(
            channel="recording",
            recipient="recipient:one",
            session_id="recipient-session",
        ),
    )

    await runtime.start()
    await runtime.close()

    assert provider_calls == ["drift thought"]
    messages = sessions.control_store.fetch_session_messages("recipient-session")
    assert [message["content"] for message in messages] == ["drift thought"]
    assert (
        drift.delivery(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
        )["status"]
        == "settled"
    )
    sessions.close()


@pytest.mark.asyncio
async def test_wake_composes_provider_session_content_and_core_settlement(
    tmp_path: Path,
) -> None:
    content, accepted, _token = _ready_content(tmp_path / "content.sqlite3")
    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    sessions = SessionManager(tmp_path / "workspace")
    provider_calls: list[str] = []

    async def sender(request, provider_started):
        provider_started(
            DurableBindingAttempt(
                request.logical_delivery_id,
                "snapshot:recording",
                "generation:recording",
                "binding:recording",
            )
        )
        provider_calls.append(request.body)
        return ChannelDeliveryReceipt(
            request.logical_delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider:recording",),
        )

    async def project(request) -> str:
        return await sessions.append_durable_delivery(
            session_key=request.projection_session_id,
            content=request.body,
            delivery_id=request.logical_delivery_id,
            control_turn_id=request.accepted_turn.turn_id,
        )

    deliveries = PluginDurableDeliveries(ledger, sender, project)
    runtime = _runtime(accepted, content, deliveries)
    await runtime.start()
    await runtime.close()

    delivery = deliveries.lookup(accepted)
    assert delivery is not None and delivery.state == "settled"
    assert delivery.provider_receipt == {
        "delivery_id": delivery.logical_delivery_id,
        "error": None,
        "provider_ids": ["provider:recording"],
        "status": "delivered",
    }
    assert provider_calls == ["Wake says hello"]
    messages = sessions.control_store.fetch_session_messages("recipient-session")
    assert len(messages) == 1
    assert messages[0]["content"] == "Wake says hello"
    assert messages[0]["control_turn_id"] == accepted.turn_id
    assert content.state_counts() == {"settled": 1}
    sessions.close()


@pytest.mark.asyncio
async def test_v1_ready_turn_with_message_only_decision_recovers_once(
    tmp_path: Path,
) -> None:
    content, accepted, _token = _ready_content(tmp_path / "content.sqlite3")
    with closing(sqlite3.connect(content.path)) as connection, connection:
        connection.executescript("""
            DROP INDEX content_selection_members_order_idx;
            DROP INDEX content_selection_status_idx;
            DROP TABLE content_selection_members;
            DROP TABLE content_selections;
            PRAGMA user_version = 1;
            """)
    content.initialize()
    recovered = content.selection(
        {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
    )
    assert recovered is not None and recovered["decision_format"] == "legacy_single"

    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    sessions = SessionManager(tmp_path / "workspace")
    provider_calls: list[str] = []

    async def sender(request, provider_started):
        provider_started(
            DurableBindingAttempt(
                request.logical_delivery_id,
                "snapshot:recording",
                "generation:recording",
                "binding:recording",
            )
        )
        provider_calls.append(request.body)
        return ChannelDeliveryReceipt(
            request.logical_delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider:recording",),
        )

    async def project(request) -> str:
        return await sessions.append_durable_delivery(
            session_key=request.projection_session_id,
            content=request.body,
            delivery_id=request.logical_delivery_id,
            control_turn_id=request.accepted_turn.turn_id,
        )

    deliveries = PluginDurableDeliveries(ledger, sender, project)
    runtime = _runtime(accepted, content, deliveries, legacy=True)
    await runtime.start()
    await runtime.close()

    assert provider_calls == ["Wake says hello"]
    assert content.state_counts() == {"settled": 1}
    messages = sessions.control_store.fetch_session_messages("recipient-session")
    assert [message["content"] for message in messages] == ["Wake says hello"]
    current = deliveries.lookup(accepted)
    assert current is not None and current.state == "settled"
    sessions.close()


@pytest.mark.asyncio
async def test_restart_after_content_settle_replays_receipt_into_core(
    tmp_path: Path,
) -> None:
    content, accepted, token = _ready_content(tmp_path / "content.sqlite3")
    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    ledger.initialize()
    logical_id = "wake:projected-crash"
    _prepare_delivery(ledger, accepted, logical_id)
    _ = ledger.mark_provider_started(
        logical_id,
        attempt_id=logical_id,
        snapshot_id="snapshot:one",
        generation_id="generation:one",
        binding_token="binding:one",
    )
    _ = ledger.mark_provider_result(
        logical_id,
        state="delivered",
        receipt={"status": "delivered"},
    )
    _ = ledger.mark_projected(logical_id, "message:one")
    first = _DeliveryServices(content).settle(token, logical_id)

    async def no_sender(_request, _provider_started):
        raise AssertionError("projected recovery must not resend")

    async def no_projector(_request):
        raise AssertionError("projected recovery must not append Session again")

    deliveries = PluginDurableDeliveries(ledger, no_sender, no_projector)
    runtime = _runtime(accepted, content, deliveries)
    await runtime.start()
    await runtime.close()

    settled = deliveries.lookup(accepted)
    assert settled is not None and settled.state == "settled"
    assert settled.domain_receipt == first["receipt"]
    assert content.state_counts() == {"settled": 1}


@pytest.mark.parametrize("terminal", ("rejected", "uncertain"))
@pytest.mark.asyncio
async def test_terminal_provider_result_is_observable_and_never_resent(
    tmp_path: Path,
    terminal: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content, accepted, _token = _ready_content(tmp_path / "content.sqlite3")
    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    ledger.initialize()
    logical_id = f"wake:{terminal}"
    _prepare_delivery(ledger, accepted, logical_id)
    _ = ledger.mark_provider_started(
        logical_id,
        attempt_id=logical_id,
        snapshot_id="snapshot:one",
        generation_id="generation:one",
        binding_token="binding:one",
    )
    _ = ledger.mark_provider_result(
        logical_id,
        state=terminal,
        receipt={"status": terminal, "error": "recording terminal"},
    )
    provider_calls = 0

    async def no_sender(_request, _provider_started):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("terminal delivery must not resend")

    async def no_projector(_request):
        raise AssertionError("terminal delivery must not project")

    deliveries = PluginDurableDeliveries(ledger, no_sender, no_projector)
    runtime = _runtime(accepted, content, deliveries)
    with caplog.at_level(logging.WARNING, logger="plugins.wake.plugin"):
        await runtime.start()
        await runtime.close()

    assert provider_calls == 0
    assert f"state={terminal}" in caplog.text
    assert "terminal without resend" in caplog.text
    assert content.state_counts() == {"ready_for_delivery": 1}
    current = deliveries.lookup(accepted)
    assert current is not None and current.state == terminal


@pytest.mark.asyncio
async def test_uncertain_batch_locks_only_cited_member_from_next_selection(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = ContentStore(tmp_path / "content.sqlite3")
    _ = content.submit(
        "feed",
        "poll:first",
        (
            {
                "item_id": "uncited",
                "revision": "1",
                "payload": {"title": "uncited"},
                "not_before": now,
                "requires_ack": False,
            },
            {
                "item_id": "cited",
                "revision": "1",
                "payload": {"title": "cited"},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )
    snapshot = content.snapshot(now)
    accepted = TurnAcceptedReceipt("wake:default", "turn:uncertain")
    selected = content.select_batch(
        tuple(item["ref"] for item in snapshot["items"]),
        snapshot["snapshot_seq"],
        {"session_id": accepted.session_id, "turn_id": accepted.turn_id},
        now,
    )
    token = selected["selection_token"]
    assert isinstance(token, str)
    cited = snapshot["items"][1]["ref"]
    _ = content.transition(token, "ready_for_delivery", selected_refs=(cited,))
    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    ledger.initialize()
    _prepare_delivery(ledger, accepted, "wake:uncertain-batch")
    _ = ledger.mark_provider_started(
        "wake:uncertain-batch",
        attempt_id="wake:uncertain-batch",
        snapshot_id="snapshot:one",
        generation_id="generation:one",
        binding_token="binding:one",
    )
    _ = ledger.mark_provider_result(
        "wake:uncertain-batch",
        state="uncertain",
        receipt={"status": "uncertain"},
    )

    async def no_sender(_request, _provider_started):
        raise AssertionError("uncertain delivery must not resend")

    async def no_projector(_request):
        raise AssertionError("uncertain delivery must not project")

    deliveries = PluginDurableDeliveries(ledger, no_sender, no_projector)
    runtime = _runtime(accepted, content, deliveries)
    await runtime.start()
    await runtime.close()
    _ = content.submit(
        "feed",
        "poll:second",
        (
            {
                "item_id": "new",
                "revision": "1",
                "payload": {"title": "new"},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )
    available = content.snapshot(now)
    available_ids = {str(item["ref"]["item_id"]) for item in available["items"]}

    assert available_ids == {"uncited", "new"}
    next_selected = content.select_batch(
        tuple(item["ref"] for item in available["items"]),
        available["snapshot_seq"],
        {"session_id": "wake:default", "turn_id": "turn:next"},
        now,
    )
    assert next_selected["selected"] is True
    assert deliveries.lookup(accepted).state == "uncertain"


def _prepare_delivery(
    ledger: DurableDeliveryStore,
    accepted: TurnAcceptedReceipt,
    logical_id: str,
) -> None:
    _ = ledger.prepare(
        {
            "logical_delivery_id": logical_id,
            "accepted_session_id": accepted.session_id,
            "accepted_turn_id": accepted.turn_id,
            "target_service": "content.delivery.v1",
            "channel": "recording",
            "recipient": "recipient:one",
            "projection_session_id": "recipient-session",
            "body": "Wake says hello",
            "metadata": {"proactive": True},
        }
    )
