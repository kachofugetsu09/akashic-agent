from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from agent.control.scoped_turn import TurnAcceptedReceipt
from agent.plugin_composition import TopologyView
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus,
    OutboundEnvelope,
)
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot
from bus.event_bus import EventBus
from bus.queue import MessageBus
from session.manager import SessionManager


def _request(logical_id: str = "delivery:one") -> DurableDeliveryRequest:
    return DurableDeliveryRequest(
        logical_delivery_id=logical_id,
        accepted_turn=TurnAcceptedReceipt("wake:default", "turn:one"),
        target_service="eventmail.delivery.v1",
        channel="recording",
        recipient="recipient:one",
        projection_session_id="recipient-session",
        body="hello from Wake",
        metadata={"proactive": True},
    )


def _envelope_for_test(request: DurableDeliveryRequest) -> dict[str, object]:
    return {
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


class _RecordingBinding:
    snapshot_id = "snapshot:recording"
    generation_id = "generation:recording"
    binding_token = "binding:recording"
    channel_name = "recording"
    active = True


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


def test_prepared_delivery_can_be_closed_before_provider_io(tmp_path: Path) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    request = _request()
    _ = store.prepare(_envelope_for_test(request))
    service = PluginDurableDeliveries(store, None, None, recover_started=False)

    cancelled = service.cancel_prepared(
        request.accepted_turn,
        reason="source fact expired before provider I/O",
    )

    assert cancelled.state == "rejected"
    assert cancelled.provider_receipt == {
        "status": "rejected",
        "error": "source fact expired before provider I/O",
        "provider_started": False,
    }
    assert service.recoverable() == ()


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


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_receipt_and_projection(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    sessions = SessionManager(tmp_path / "workspace")
    bus = MessageBus()
    provider_entered = asyncio.Event()
    provider_release = asyncio.Event()

    async def provider(
        envelope: OutboundEnvelope, _binding: object
    ) -> ChannelDeliveryReceipt:
        provider_entered.set()
        await provider_release.wait()
        return ChannelDeliveryReceipt(
            envelope.delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider:after-cancel",),
        )

    async def sender(request, provider_started):
        binding = _RecordingBinding()
        envelope = OutboundEnvelope(
            logical_delivery_id=request.logical_delivery_id,
            delivery_id=request.logical_delivery_id,
            attempt_sequence=1,
            snapshot_id=binding.snapshot_id,
            generation_id=binding.generation_id,
            binding_token=binding.binding_token,
            channel=binding.channel_name,
            recipient=request.recipient,
            body=request.body,
            metadata={},
        )
        attempt = DurableBindingAttempt(
            request.logical_delivery_id,
            binding.snapshot_id,
            binding.generation_id,
            binding.binding_token,
        )
        return await bus.publish_channel_outbound_awaited(
            envelope,
            binding,
            passive=False,
            before_provider=lambda: provider_started(attempt),
        )

    async def project(request) -> str:
        return await sessions.append_durable_delivery(
            session_key=request.projection_session_id,
            content=request.body,
            delivery_id=request.logical_delivery_id,
            control_turn_id=request.accepted_turn.turn_id,
        )

    bus.bind_channel_outbound_dispatcher(provider)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    service = PluginDurableDeliveries(store, sender, project)
    caller = asyncio.create_task(service.submit(_request()))
    await provider_entered.wait()
    caller.cancel()
    provider_release.set()

    with pytest.raises(asyncio.CancelledError):
        await caller
    current = service.lookup(TurnAcceptedReceipt("wake:default", "turn:one"))
    assert current is not None and current.state == "projected"
    assert current.provider_receipt is not None
    assert current.provider_receipt["status"] == "delivered"
    assert len(sessions.control_store.fetch_session_messages("recipient-session")) == 1
    bus.stop()
    await dispatch
    sessions.close()


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


@pytest.mark.asyncio
async def test_akashic_provider_started_restart_recovers_from_session_without_resend(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    store.initialize()
    request = DurableDeliveryRequest(
        logical_delivery_id="delivery:akashic-crash",
        accepted_turn=TurnAcceptedReceipt("wake:default", "turn:akashic-crash"),
        target_service="eventmail.delivery.v1",
        channel="akashic",
        recipient="chat:one",
        projection_session_id="akashic:chat:one",
        body="Session already committed this body",
    )
    _ = store.prepare(_envelope_for_test(request))
    _ = store.mark_provider_started(
        request.logical_delivery_id,
        attempt_id=request.logical_delivery_id,
        snapshot_id="snapshot:one",
        generation_id="generation:one",
        binding_token="binding:one",
    )
    sessions = SessionManager(tmp_path / "workspace")
    committed_id = await sessions.append_durable_delivery(
        session_key=request.projection_session_id,
        content=request.body,
        delivery_id=request.logical_delivery_id,
        control_turn_id=request.accepted_turn.turn_id,
    )

    async def no_sender(_request, _provider_started):
        raise AssertionError("Akashic crash recovery must not notify again")

    async def project(existing: DurableDeliveryRequest) -> str:
        return await sessions.append_durable_delivery(
            session_key=existing.projection_session_id,
            content=existing.body,
            delivery_id=existing.logical_delivery_id,
            control_turn_id=existing.accepted_turn.turn_id,
        )

    service = PluginDurableDeliveries(store, no_sender, project)
    assert [item.state for item in service.recoverable()] == ["provider_started"]
    recovered = await service.resume(request.accepted_turn)

    assert recovered.state == "projected"
    assert recovered.projection_message_id == committed_id
    assert recovered.provider_receipt == {
        "delivery_id": request.logical_delivery_id,
        "recovered_from": "session",
        "status": "delivered",
    }
    assert len(sessions.control_store.fetch_session_messages(request.projection_session_id)) == 1
    sessions.close()


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


def test_version_zero_unknown_schema_failure_leaves_database_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settlements.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE legacy_delivery(value TEXT)")
        connection.commit()

    def schema() -> tuple[int, str, tuple[tuple[str, str], ...]]:
        with closing(sqlite3.connect(path)) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
            objects = tuple(
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    "SELECT type, name FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                )
            )
            return version, journal_mode, objects

    before = schema()
    with pytest.raises(RuntimeError, match="version 0 schema must be empty"):
        DurableDeliveryStore(path).initialize()
    assert schema() == before == (
        0,
        "delete",
        (("table", "legacy_delivery"),),
    )


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
        target_service="eventmail.delivery.v1",
        channel="recording",
        recipient="recipient:body",
        projection_session_id="projection:body",
        body=body,
    ).body == body


def test_candidate_fence_keeps_akashic_crash_recovery_target(
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
            "channel": "akashic",
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
        candidate(("eventmail.delivery.v1",))
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
    with pytest.raises(RuntimeError, match="target service 不可解析"):
        manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
            candidate(())
        )
    _ = store.mark_provider_result(
        request.logical_delivery_id,
        state="delivered",
        receipt={"status": "delivered"},
    )
    _ = store.mark_projected(request.logical_delivery_id, "message:one")
    with pytest.raises(RuntimeError, match="target service 不可解析"):
        manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
            candidate(())
        )
    _ = store.confirm_settled(request.logical_delivery_id, "content:receipt")
    manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
        candidate(())
    )


@pytest.mark.parametrize("terminal", ("rejected", "uncertain"))
def test_candidate_fence_ignores_nonrecoverable_terminal_rows(
    tmp_path: Path,
    terminal: str,
) -> None:
    workspace = tmp_path / "workspace"
    manager = PluginManager([], event_bus=EventBus(), workspace=workspace)
    store = DurableDeliveryStore(
        workspace / "runtime" / "deliveries" / "settlements.sqlite"
    )
    store.initialize()
    request = _request(f"delivery:{terminal}")
    envelope = {
        "logical_delivery_id": request.logical_delivery_id,
        "accepted_session_id": request.accepted_turn.session_id,
        "accepted_turn_id": request.accepted_turn.turn_id,
        "target_service": request.target_service,
        "channel": request.channel,
        "recipient": request.recipient,
        "projection_session_id": request.projection_session_id,
        "body": request.body,
        "metadata": {},
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
        state=terminal,
        receipt={"status": terminal},
    )
    snapshot = RuntimeSnapshot(
        "candidate",
        {},
        None,
        composition_topology=TopologyView(
            "root:candidate", "topology:candidate", 1, (), (), (), ()
        ),
    )
    manager._preflight_durable_delivery_targets(  # pyright: ignore[reportPrivateUsage]
        snapshot
    )


def test_oversized_logical_id_is_rejected_before_any_write_or_provider(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    store.initialize()
    provider_calls = 0

    async def sender(_request, _provider_started):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("invalid delivery id must not reach provider")

    _ = PluginDurableDeliveries(store, sender, None)
    with pytest.raises(ValueError, match=r"1\.\.128"):
        DurableDeliveryRequest(
            logical_delivery_id="d" * 129,
            accepted_turn=TurnAcceptedReceipt("session:long", "turn:long"),
            target_service="eventmail.delivery.v1",
            channel="recording",
            recipient="recipient:long",
            projection_session_id="projection:long",
            body="payload",
        )

    assert provider_calls == 0
    assert store.recoverable() == ()
