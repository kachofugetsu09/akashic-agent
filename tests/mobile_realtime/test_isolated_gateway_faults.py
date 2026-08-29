from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition.channels import ChannelInboundMessage, RawInbound
from session.manager import SessionManager
from tests_scenarios.mobile_isolated_gateway import (
    FixedReplyBus,
    GatewayFaultController,
    load_replay_turn,
)


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


def test_load_replay_turn_validates_and_preserves_real_stage_order(tmp_path: Any) -> None:
    replay_path = tmp_path / "turn.json"
    _ = replay_path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "问题"},
                {
                    "role": "assistant",
                    "content": "最终回答",
                    "tool_chain": json.dumps(
                        [
                            {
                                "reasoning_content": "先思考",
                                "text": "中间说明",
                                "calls": [
                                    {
                                        "call_id": "call-1",
                                        "name": "inspect",
                                        "status": "success",
                                        "arguments": {"step": 1},
                                        "final_arguments": {"step": 1},
                                        "result": "完成",
                                    }
                                ],
                            }
                        ]
                    ),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    replay = load_replay_turn(replay_path)

    assert replay.content == "最终回答"
    assert replay.reasoning == "先思考"
    assert replay.call_count == 1
    assert replay.stages[0].text == "中间说明"
    assert replay.stages[0].calls[0].name == "inspect"


def test_performance_fixture_can_emit_one_character_provider_deltas(tmp_path: Any) -> None:
    manager = SessionManager(tmp_path / "workspace")
    media = tmp_path / "fixed.gif"
    _ = media.write_bytes(b"fixture")
    bus = FixedReplyBus(
        manager,
        media,
        tokens_per_second=100,
        stream_tokens=12,
        stream_chunk_chars=1,
    )

    try:
        thinking, _, answer, thinking_delay, answer_delay = bus._stream_payloads()  # pyright: ignore[reportPrivateUsage]
        assert all(len(delta) == 1 for delta in (*thinking, *answer))
        assert len(thinking) + len(answer) == 12
        assert thinking_delay == pytest.approx(0.01)
        assert answer_delay == pytest.approx(0.01)
    finally:
        manager.close()


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
