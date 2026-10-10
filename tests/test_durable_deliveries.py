from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import pytest
from agent.plugin_composition.durable_deliveries import TurnAcceptedReceipt
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus,
)
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Output

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

@pytest.mark.asyncio
async def test_provider_receipt_precedes_one_append_only_session_projection(
    tmp_path: Path,
) -> None:
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    log = MessageLog(tmp_path / "sessions.db")
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
        writer = log.writer(
            request.projection_session_id,
            author="assistant", source="delivery", body_types=(Output,),
            content={"text": lambda _: ContentReferences()},
            message_metadata_keys=frozenset({"delivery"}),
        )
        return writer.append(
            request.logical_delivery_id,
            Output((ContentPart("text", request.body),), "complete"),
            metadata={"delivery": {"id": request.logical_delivery_id}},
        ).message_id

    service = PluginDurableDeliveries(store, sender, project)
    projected = await service.submit(_request())
    duplicate = await service.submit(_request())

    assert provider_states == ["provider_started"]
    assert projected.state == duplicate.state == "projected"
    messages = log.reader("recipient-session").snapshot()
    assert len(messages) == 1
    assert messages[0].body == Output((ContentPart("text", "hello from Wake"),), "complete")
    assert messages[0].metadata["delivery"] == {"id": "delivery:one"}
    assert store.confirm_settled("delivery:one", "domain:one")["state"] == "settled"
    assert store.confirm_settled("delivery:one", "domain:one")["state"] == "settled"
    log.close()

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
    assert recovered is not None and recovered.state == "failed"
    assert recovered.provider_receipt == {"status": "failed", "error": "provider call interrupted; delivery may have occurred"}
    assert service.recoverable() == ()
