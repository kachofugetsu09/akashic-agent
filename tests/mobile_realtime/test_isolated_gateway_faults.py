from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition.channels import ChannelInboundMessage, RawInbound
from session.manager import SessionManager
from tests_scenarios.mobile_isolated_gateway import FixedReplyBus, GatewayFaultController


def test_before_challenge_fault_waits_for_pairing_and_triggers_once() -> None:
    controller = GatewayFaultController("stall_before_challenge")

    assert not controller.claim_before_challenge(has_paired_device=False)
    assert controller.claim_before_challenge(has_paired_device=True)
    assert not controller.claim_before_challenge(has_paired_device=True)
    assert not controller.claim_after_auth()


def test_after_auth_fault_triggers_once() -> None:
    controller = GatewayFaultController("stall_after_auth")

    assert not controller.claim_before_challenge(has_paired_device=True)
    assert controller.claim_after_auth()
    assert not controller.claim_after_auth()


def test_fault_controller_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知隔离 Gateway 故障模式"):
        GatewayFaultController("drop_everything")


@pytest.mark.asyncio
async def test_fixed_reply_is_persisted_before_v3_admit_returns(tmp_path: Any) -> None:
    manager = SessionManager(tmp_path / "workspace")
    media = tmp_path / "fixed.gif"
    _ = media.write_bytes(b"fixture")
    started = asyncio.Event()

    class BlockingChannel:
        name = "mobile"

        async def _on_turn_started(self, event: object) -> None:
            started.set()
            await asyncio.Event().wait()

    bus = FixedReplyBus(manager, media)
    bus.bind(cast(Any, SimpleNamespace(channel=BlockingChannel())))
    raw = RawInbound(
        message_id="client-message",
        provider_identity="mobile:akashic:test",
        recipient="mobile:akashic:test",
        message=ChannelInboundMessage(
            channel="mobile",
            sender="device:test",
            chat_id="mobile:akashic:test",
            content="问题",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "client_message_id": "client-message",
                "session_key_override": "akashic:test",
            },
        ),
    )

    try:
        assert await bus.admit(raw)
        session = manager.get_existing("akashic:test")
        assert [message["role"] for message in session.messages] == ["user", "assistant"]
        assert session.messages[0]["client_message_id"] == "client-message"
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        await bus.aclose()
        manager.close()
