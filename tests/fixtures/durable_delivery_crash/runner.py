from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus,
    OutboundEnvelope,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from bus.queue import MessageBus


class _Binding:
    snapshot_id = "snapshot:fixture"
    generation_id = "generation:fixture"
    binding_token = "binding:fixture"
    channel_name = "recording"
    active = True


async def main(root: Path) -> None:
    """Crash at the real MessageBus provider edge after the durable commit."""

    # 1. Prepare the immutable Core delivery before it enters the bus.
    logical_id = "delivery:crash"
    store = DurableDeliveryStore(root / "settlements.sqlite")
    store.initialize()
    _ = store.prepare(
        {
            "logical_delivery_id": logical_id,
            "accepted_session_id": "session:crash",
            "accepted_turn_id": "turn:crash",
            "target_service": "eventmail.delivery.v1",
            "channel": "recording",
            "recipient": "recipient:crash",
            "projection_session_id": "projection:crash",
            "body": "crash payload",
            "metadata": {},
        }
    )
    envelope = OutboundEnvelope(
        logical_delivery_id=logical_id,
        delivery_id=logical_id,
        attempt_sequence=1,
        snapshot_id=_Binding.snapshot_id,
        generation_id=_Binding.generation_id,
        binding_token=_Binding.binding_token,
        channel=_Binding.channel_name,
        recipient="recipient:crash",
        body="crash payload",
        metadata={},
    )

    # 2. Prove SIGKILL lands after the durable callback and before provider effect.
    async def provider(
        _envelope: OutboundEnvelope, _binding: object
    ) -> ChannelDeliveryReceipt:
        with (root / "provider-calls").open("a", encoding="utf-8") as handle:
            _ = handle.write(logical_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return ChannelDeliveryReceipt(logical_id, DeliveryStatus.DELIVERED)

    def before_provider() -> None:
        _ = store.mark_provider_started(
            logical_id,
            attempt_id=logical_id,
            snapshot_id=_Binding.snapshot_id,
            generation_id=_Binding.generation_id,
            binding_token=_Binding.binding_token,
        )
        with (root / "provider-edge").open("w", encoding="utf-8") as handle:
            _ = handle.write("provider_started\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.kill(os.getpid(), signal.SIGKILL)

    bus = MessageBus()
    bus.bind_channel_outbound_dispatcher(provider)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_channel_outbound_awaited(
        envelope,
        _Binding(),
        passive=False,
        before_provider=before_provider,
    )
    await dispatch


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
