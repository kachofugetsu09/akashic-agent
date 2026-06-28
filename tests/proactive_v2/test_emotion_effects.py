from __future__ import annotations

from proactive_v2.gateway import GatewayDeps
from tests.proactive_v2.conftest import FakeLLM, make_proactive_pipeline, run_proactive_pipeline
from unittest.mock import AsyncMock


async def test_proactive_prompt_slot_injects_prompt_and_tick_log():
    llm = FakeLLM([("finish_turn", {"decision": "skip", "reason": "no_content"})])
    state_store = None
    tick = make_proactive_pipeline(
        llm_fn=llm,
        state_store=state_store,
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(
                return_value=[
                    {
                        "event_id": "c1",
                        "ack_server": "feed",
                        "title": "测试内容",
                        "source": "feed",
                    }
                ]
            ),
            context_fn=AsyncMock(return_value=[]),
        ),
    )
    slots = {
        "proactive:prompt:system_bottom:emotion": "当前 VAD: valence=0.10, arousal=0.20, dominance=-0.30。",
        "proactive:effect:emotion": {
            "provider_name": "emotion",
            "threshold_delta": 0.04,
            "metadata": {
                "final_threshold": 0.64,
                "expected_effect": "raise_send_bar",
            },
        },
    }

    await run_proactive_pipeline(tick, slots=slots)

    assert "【主动插件状态】" in llm.calls[0][0]["content"]
    assert "当前 VAD" in llm.calls[0][0]["content"]
    effects = tick._state_store.tick_log_finishes[0]["proactive_effects"]
    assert effects[0]["provider_name"] == "emotion"
    assert effects[0]["metadata"]["expected_effect"] == "raise_send_bar"
