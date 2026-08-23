from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from agent.control.scoped_turn import TurnAcceptedReceipt
from agent.plugin_composition import TopologyView
from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot
from bus.event_bus import EventBus
from session.manager import SessionManager


def _request(logical_id: str = "delivery:one") -> DurableDeliveryRequest:
    return DurableDeliveryRequest(
        logical_delivery_id=logical_id,
        accepted_turn=TurnAcceptedReceipt("wake:default", "turn:one"),
        target_service="content.delivery.v1",
        channel="recording",
        recipient="recipient:one",
        projection_session_id="recipient-session",
        body="hello from Wake",
        metadata={"proactive": True},
    )


@pytest.mark.asyncio
async def test_provider_receipt_precedes_one_append_only_session_projection(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    sessions = SessionManager(tmp_path / "workspace")
    provider_states: list[str] = []

    async def sender(request, provider_started):
        provider_started(
            DurableBindingAttempt(
                request.logical_delivery_id,
                "snapshot:one",
                "generation:one",
                "binding:one",
            )
        )
        row = store.lookup("wake:default", "turn:one")
        assert row is not None
        provider_states.append(str(row["state"]))
        return ChannelDeliveryReceipt(
            request.logical_delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider:one",),
        )

    async def project(request) -> str:
        row = store.lookup("wake:default", "turn:one")
        assert row is not None and row["state"] == "delivered"
        return await sessions.append_durable_delivery(
            session_key=request.projection_session_id,
            content=request.body,
            delivery_id=request.logical_delivery_id,
            control_turn_id=request.accepted_turn.turn_id,
        )

    service = PluginDurableDeliveries(store, sender, project)
    projected = await service.submit(_request())
    duplicate = await service.submit(_request())

    assert provider_states == ["provider_started"]
    assert projected.state == duplicate.state == "projected"
    messages = sessions.control_store.fetch_session_messages("recipient-session")
    assert len(messages) == 1
    assert messages[0]["content"] == "hello from Wake"
    assert messages[0]["delivery_id"] == "delivery:one"
    assert store.confirm_settled("delivery:one", "domain:one")["state"] == "settled"
    assert store.confirm_settled("delivery:one", "domain:one")["state"] == "settled"
    sessions.close()


@pytest.mark.asyncio
async def test_delivered_restart_only_projects_without_provider_resend(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    store.initialize()
    request = _request()
    _ = store.prepare(
        {
            "logical_delivery_id": request.logical_delivery_id,
            "accepted_session_id": request.accepted_turn.session_id,
            "accepted_turn_id": request.accepted_turn.turn_id,
            "target_service": request.target_service,
            "channel": request.channel,
            "recipient": request.recipient,
            "projection_session_id": request.projection_session_id,
            "body": request.body,
            "metadata": dict(request.metadata),
        }
    )
    _ = store.mark_provider_started(
        request.logical_delivery_id,
        attempt_id=request.logical_delivery_id,
        snapshot_id="snapshot:one",
        generation_id="generation:one",
        binding_token="binding:one",
    )
    _ = store.mark_provider_result(
        request.logical_delivery_id,
        state="delivered",
        receipt={"delivery_id": request.logical_delivery_id, "status": "delivered"},
    )
    provider_calls = 0
    projections = 0

    async def sender(_request, _provider_started):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("delivered recovery must not call provider")

    async def projector(_request) -> str:
        nonlocal projections
        projections += 1
        return "recipient-session:1"

    service = PluginDurableDeliveries(store, sender, projector)
    result = await service.resume(request.accepted_turn)

    assert result.state == "projected"
    assert provider_calls == 0
    assert projections == 1


def test_provider_started_sigkill_recovers_uncertain_without_resend(
    tmp_path: Path,
) -> None:
    runner = Path(__file__).parent / "fixtures" / "durable_delivery_crash" / "runner.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, str(runner), str(tmp_path)],
        env=env,
        check=False,
    )
    assert completed.returncode == -9
    assert (tmp_path / "provider-edge").read_text(encoding="utf-8").splitlines() == [
        "provider_started"
    ]
    assert not (tmp_path / "provider-calls").exists()

    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    service = PluginDurableDeliveries(store, None, None)
    recovered = service.lookup(TurnAcceptedReceipt("session:crash", "turn:crash"))
    assert recovered is not None and recovered.state == "uncertain"
    assert service.recoverable() == ()


def test_exact_schema_rejects_same_version_missing_index_and_extra_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settlements.sqlite"
    store = DurableDeliveryStore(path)
    store.initialize()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP INDEX idx_deliveries_recoverable")
        connection.commit()
    with pytest.raises(RuntimeError, match="index identity"):
        store.initialize()

    path.unlink()
    store.initialize()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE extra_state(value TEXT)")
        connection.commit()
    with pytest.raises(RuntimeError, match="table identity"):
        store.initialize()


def test_terminal_transitions_reject_backward_or_conflicting_receipts(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    store.initialize()
    request = _request()
    envelope = {
        "logical_delivery_id": request.logical_delivery_id,
        "accepted_session_id": request.accepted_turn.session_id,
        "accepted_turn_id": request.accepted_turn.turn_id,
        "target_service": request.target_service,
        "channel": request.channel,
        "recipient": request.recipient,
        "projection_session_id": request.projection_session_id,
        "body": request.body,
        "metadata": dict(request.metadata),
    }
    _ = store.prepare(envelope)
    _ = store.mark_provider_started(
        request.logical_delivery_id,
        attempt_id=request.logical_delivery_id,
        snapshot_id="snapshot:one",
        generation_id="generation:one",
        binding_token="binding:one",
    )
    _ = store.mark_provider_result(
        request.logical_delivery_id,
        state="rejected",
        receipt={"status": "rejected"},
    )
    with pytest.raises(RuntimeError, match="provider result transition"):
        store.mark_provider_result(
            request.logical_delivery_id,
            state="delivered",
            receipt={"status": "delivered"},
        )
    with pytest.raises(RuntimeError, match="projected transition"):
        store.mark_projected(request.logical_delivery_id, "message:one")

    with pytest.raises(RuntimeError, match="immutable envelope conflict"):
        store.prepare({**envelope, "body": "different"})


def test_delivery_body_preserves_surrounding_newlines(tmp_path: Path) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    store.initialize()
    request = _request()
    body = "\n  model response  \n"
    row = store.prepare(
        {
            "logical_delivery_id": request.logical_delivery_id,
            "accepted_session_id": request.accepted_turn.session_id,
            "accepted_turn_id": request.accepted_turn.turn_id,
            "target_service": request.target_service,
            "channel": request.channel,
            "recipient": request.recipient,
            "projection_session_id": request.projection_session_id,
            "body": body,
            "metadata": {},
        }
    )
    assert row["body"] == body
    assert DurableDeliveryRequest(
        logical_delivery_id="delivery:body",
        accepted_turn=TurnAcceptedReceipt("session:body", "turn:body"),
        target_service="content.delivery.v1",
        channel="recording",
        recipient="recipient:body",
        projection_session_id="projection:body",
        body=body,
    ).body == body


def test_candidate_fence_reads_only_prepared_target_service_identity(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    manager = PluginManager([], event_bus=EventBus(), workspace=workspace)
    store = DurableDeliveryStore(
        workspace / "runtime" / "deliveries" / "settlements.sqlite"
    )
    store.initialize()
    request = _request()
    _ = store.prepare(
        {
            "logical_delivery_id": request.logical_delivery_id,
            "accepted_session_id": request.accepted_turn.session_id,
            "accepted_turn_id": request.accepted_turn.turn_id,
            "target_service": request.target_service,
            "channel": request.channel,
            "recipient": request.recipient,
            "projection_session_id": request.projection_session_id,
            "body": request.body,
            "metadata": dict(request.metadata),
        }
    )

    def candidate(services: tuple[str, ...]) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            "candidate",
            {},
            None,
            composition_topology=TopologyView(
                generation_id="root:candidate",
                identity="topology:candidate",
                composition_revision=1,
                fibers=(),
                services=services,
                effects=(),
                listeners=(),
            ),
        )

    manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
        candidate(("content.delivery.v1",))
    )
    with pytest.raises(RuntimeError, match="target service 不可解析"):
        manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
            candidate(())
        )

    _ = store.mark_provider_started(
        request.logical_delivery_id,
        attempt_id="attempt:changed-binding",
        snapshot_id="snapshot:changed",
        generation_id="generation:changed",
        binding_token="binding:changed",
    )
    _ = store.mark_provider_result(
        request.logical_delivery_id,
        state="rejected",
        receipt={"status": "rejected"},
    )
    manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
        candidate(())
    )
