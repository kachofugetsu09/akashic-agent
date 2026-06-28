from __future__ import annotations
from typing import Any, cast

from types import SimpleNamespace
from unittest.mock import MagicMock

from bootstrap.proactive import _build_proactive_provider, build_proactive_runtime
from agent.core.proactive_turn import ProactiveTurnPipeline, ProactiveTurnPipelineDeps
from core.memory.markdown import MemoryProfileApi
from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.gateway import GatewayDeps, GatewayResult
from proactive_v2.tools import ToolDeps


def test_build_proactive_runtime_accepts_facade_memory(tmp_path):
    proactive_cfg = ProactiveConfig()
    proactive_cfg.enabled = True
    proactive_cfg.default_channel = "telegram"
    proactive_cfg.default_chat_id = "1"
    cfg = SimpleNamespace(
        proactive=proactive_cfg,
        memory_optimizer_enabled=False,
        memory_optimizer_interval_seconds=3600,
        model="m",
        max_tokens=128,
    )
    facade = MagicMock()

    tasks, loop = build_proactive_runtime(
        cast(Any, cfg),
        tmp_path,
        session_manager=cast(Any, SimpleNamespace(workspace=tmp_path)),
        provider=cast(Any, SimpleNamespace()),
        push_tool=cast(Any, SimpleNamespace()),
        memory_store=facade,
        presence=cast(Any, SimpleNamespace()),
        agent_loop=cast(Any, SimpleNamespace(processing_state=None)),
    )

    assert loop is not None
    assert loop._memory is facade
    for task in tasks:
        close = getattr(task, "close", None)
        if callable(close):
            close()


def test_build_proactive_provider_strips_enable_thinking():
    provider = MagicMock()
    cfg = SimpleNamespace(
        api_key="k",
        base_url="https://example.com/v1",
        system_prompt="sys",
        extra_body={"enable_thinking": True, "foo": "bar"},
    )

    proactive_provider = _build_proactive_provider(cast(Any, cfg), provider)

    assert proactive_provider is not provider
    assert proactive_provider._extra_body == {"foo": "bar"}
    assert proactive_provider._force_disable_thinking is True


def test_agent_tick_prompt_keeps_self_block_with_facade():
    tick = ProactiveTurnPipeline(
        ProactiveTurnPipelineDeps(
            cfg=ProactiveConfig(),
            session_key="test",
            state_store=MagicMock(),
            any_action_gate=MagicMock(),
            last_user_at_fn=lambda: None,
            passive_busy_fn=None,
            turn_orchestrator=None,
            deduper=MagicMock(),
            tool_deps=ToolDeps(
                memory=cast(MemoryProfileApi, SimpleNamespace(
                    read_long_term_context=lambda: "MEMORY",
                    read_self=lambda: "SELF",
                )),
                recent_chat_fn=None,
            ),
            gateway_deps=GatewayDeps(
                alert_fn=MagicMock(),
                feed_fn=MagicMock(),
                context_fn=MagicMock(),
            ),
            workspace_context_fn=None,
            llm_fn=None,
            rng=None,
            recent_proactive_fn=None,
            drift_pipeline=None,
        ),
    )

    runtime_context = tick._prompt_builder.build_runtime_context_message(
        AgentTickContext(session_key="test"),
        GatewayResult(),
    )
    content = str(runtime_context["content"])

    assert "self_model" in content
    assert "SELF" in content
