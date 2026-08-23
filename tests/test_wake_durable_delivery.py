from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from agent.control.models import TurnStatus
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
    _WakeServices,
)
from plugins.content.store import ContentStore
from plugins.wake.plugin import DeliveryTarget, DriftWakeServices, WakeRuntime
from session.manager import SessionManager


class _NoTimers:
    def schedule(self, _deadline):
        raise AssertionError("ready delivery reconciliation must not schedule a timer")


class _Turns:
    def __init__(self, accepted: TurnAcceptedReceipt, response: str) -> None:
        self.accepted = accepted
        self.response = response

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
) -> WakeRuntime:
    return WakeRuntime(
        cast(PluginTimers, _NoTimers()),
        cast(PluginScopedTurns, _Turns(accepted, response)),
        _WakeServices(content),
        cast(DriftWakeServices, _NoDrift()),
        deliveries=deliveries,
        content_delivery=cast(ContentDeliveryServices, _DeliveryServices(content)),
        target=DeliveryTarget(
            channel="recording",
            recipient="recipient:one",
            session_id="recipient-session",
        ),
    )


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
