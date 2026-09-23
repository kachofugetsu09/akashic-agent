from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from agent.control.scoped_turn import TurnAcceptedReceipt
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


class _Binding:
    snapshot_id = "snapshot:fixture"
    generation_id = "generation:fixture"
    binding_token = "binding:fixture"


async def main(root: Path) -> None:
    """Crash after durable provider_started and provider effect, before receipt."""

    # 1. Prepare the immutable Core delivery before the sender callback.
    logical_id = "delivery:crash"
    store = DurableDeliveryStore(root / "settlements.sqlite")
    request = DurableDeliveryRequest(
        logical_delivery_id=logical_id,
        accepted_turn=TurnAcceptedReceipt("session:crash", "turn:crash"),
        target_service="eventmail.delivery.v1",
        channel="recording",
        recipient="recipient:crash",
        projection_session_id="projection:crash",
        body="crash payload",
    )

    # 2. Commit the durable edge, crash before the provider call can be recorded.
    async def sender(
        current: DurableDeliveryRequest,
        provider_started,
    ) -> ChannelDeliveryReceipt:
        provider_started(
            DurableBindingAttempt(
                current.logical_delivery_id,
                _Binding.snapshot_id,
                _Binding.generation_id,
                _Binding.binding_token,
            )
        )
        with (root / "provider-edge").open("w", encoding="utf-8") as handle:
            _ = handle.write("provider_started\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
        with (root / "provider-calls").open("a", encoding="utf-8") as handle:
            _ = handle.write(f"{current.logical_delivery_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return ChannelDeliveryReceipt(
            current.logical_delivery_id,
            DeliveryStatus.DELIVERED,
        )

    async def projector(_request: DurableDeliveryRequest) -> str:
        raise AssertionError("provider crash must happen before projection")

    service = PluginDurableDeliveries(
        store,
        sender,
        projector,
        recover_started=False,
    )
    await service.submit(request)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
