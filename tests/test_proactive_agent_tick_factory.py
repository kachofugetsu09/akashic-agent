from __future__ import annotations
from typing import Any, cast

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.default_proactive.factory import AgentTickFactory
from plugins.default_proactive.runtime import build_default_proactive_modules
from plugins.drift_flow.modules import build_drift_flow_modules
from plugins.proactive_flow.modules import build_proactive_flow_modules
from proactive_v2.config import ProactiveConfig
from plugins.default_proactive.context import AgentTickContext
from proactive_v2.mcp_sources import SharedMcpGateway
from proactive_v2.lifecycle import ProactiveLifecycleBuilder, ProactiveLifecycleSpec
from proactive_v2.runtime_scope import ProactiveRuntimeScope
from agent.tools.registry import ToolRegistry
from bootstrap.proactive import build_proactive_runtime


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(tool_calls=[], content="")


def _build_deps():
    cfg = ProactiveConfig(
        default_chat_id="cid",
        message_dedupe_recent_n=3,
    )
    sense = SimpleNamespace(
        target_session_key=lambda: "telegram:1",
        collect_recent=lambda: [],
        collect_recent_proactive=lambda n: [],
    )
    return ProactiveRuntimeScope(
        cfg=cfg,
        sense=sense,
        presence=SimpleNamespace(get_last_user_at=lambda _: None),
        provider=_FakeProvider(),
        model="m",
        max_tokens=128,
        memory=None,
        state_store=SimpleNamespace(),
        any_action_gate=SimpleNamespace(),
        passive_busy_fn=None,
        deduper=None,
        rng=SimpleNamespace(),
        workspace_context_fn=lambda: "",
        mcp_gateway=SharedMcpGateway(Path("."), ToolRegistry()),
    )


def test_agent_tick_factory_build_returns_tick():
    deps = _build_deps()
    tick = AgentTickFactory(deps).build_runtime()
    assert tick is not None


def test_plugin_flow_modules_close_default_lifecycle():
    runtime = AgentTickFactory(_build_deps()).build_runtime()
    modules = [
        *build_default_proactive_modules(runtime),
        *build_proactive_flow_modules(runtime),
        *build_drift_flow_modules(runtime),
    ]

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(
            id="default",
            terminal_slots=("run:next_wakeup",),
        ),
        modules,
    )

    assert lifecycle.modules[-1].slot == "proactive.schedule"


def test_agent_tick_factory_builds_drift_pipeline_when_enabled(tmp_path):
    deps = _build_deps()
    deps.cfg = ProactiveConfig(
        default_chat_id="cid",
        drift_enabled=True,
    )
    deps.state_store = SimpleNamespace(workspace_dir=tmp_path)
    deps.any_action_gate = SimpleNamespace()
    tick = AgentTickFactory(deps).build_runtime()
    assert tick._drift_pipeline is not None
    assert tick._drift_pipeline._store.drift_dir == tmp_path / "drift"
    assert tick._drift_pipeline._store.include_builtin_skills is False
    assert tick._drift_pipeline._store.builtin_skills_dir is None


def test_agent_tick_factory_binds_drift_step_recorder_to_tick_store(tmp_path):
    deps = _build_deps()
    deps.cfg = ProactiveConfig(
        default_chat_id="cid",
        drift_enabled=True,
    )
    state_store = SimpleNamespace(
        workspace_dir=tmp_path,
        record_tick_step_log=MagicMock(),
    )
    deps.state_store = state_store

    tick = AgentTickFactory(deps).build_runtime()
    assert tick._drift_pipeline is not None
    assert tick._drift_pipeline.step_recorder is not None

    ctx = AgentTickContext(tick_id="tick1", session_key="telegram:1")
    ctx.steps_taken = 3
    tick._drift_pipeline.step_recorder(
        ctx,
        "drift",
        "read_file",
        "call1",
        {"path": "x"},
        "ok",
    )

    state_store.record_tick_step_log.assert_called_once()
    kwargs = state_store.record_tick_step_log.call_args.kwargs
    assert kwargs["tick_id"] == "tick1"
    assert kwargs["step_index"] == 3
    assert kwargs["phase"] == "drift"
    assert kwargs["tool_name"] == "read_file"


def test_build_proactive_runtime_accepts_light_agent_loop_stub(tmp_path):
    cfg = SimpleNamespace(
        proactive=SimpleNamespace(
            enabled=False,
        ),
        memory_optimizer_enabled=False,
        memory_optimizer_interval_seconds=3600,
        model="m",
        max_tokens=128,
    )
    tasks, loop = build_proactive_runtime(
        cast(Any, cfg),
        tmp_path,
        session_manager=cast(Any, SimpleNamespace()),
        provider=cast(Any, SimpleNamespace()),
        push_tool=cast(Any, SimpleNamespace()),
        memory_store=None,
        presence=cast(Any, SimpleNamespace()),
        agent_loop=cast(Any, SimpleNamespace(processing_state=None)),
    )
    assert tasks == []
    assert loop is None


async def test_agent_tick_factory_llm_fn_forces_disable_thinking():
    deps = _build_deps()
    provider = deps.provider
    tick = AgentTickFactory(deps).build_runtime()

    await tick._llm_fn(
        messages=[{"role": "user", "content": "hi"}],
        schemas=[{"type": "function"}],
        tool_choice="required",
        disable_thinking=True,
    )

    assert provider.calls[-1]["disable_thinking"] is True


async def test_agent_tick_factory_llm_fn_honors_disable_thinking_without_schemas():
    deps = _build_deps()
    provider = deps.provider
    tick = AgentTickFactory(deps).build_runtime()

    await tick._llm_fn(
        messages=[{"role": "user", "content": "hi"}],
        schemas=[],
        disable_thinking=True,
    )

    assert provider.calls[-1]["disable_thinking"] is True
