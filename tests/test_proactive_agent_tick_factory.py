from __future__ import annotations

from types import SimpleNamespace

from proactive.agent_tick_factory import AgentTickDeps, AgentTickFactory
from bootstrap.proactive import build_proactive_runtime


class _FakeProvider:
    async def chat(self, **kwargs):
        return SimpleNamespace(tool_calls=[])


def _build_deps(use_agent_tick: bool):
    cfg = SimpleNamespace(
        use_agent_tick=use_agent_tick,
        default_chat_id="cid",
        agent_tick_model="",
        agent_tick_web_fetch_max_chars=4000,
        message_dedupe_recent_n=3,
    )
    sense = SimpleNamespace(
        target_session_key=lambda: "telegram:1",
        collect_recent=lambda: [],
        collect_recent_proactive=lambda n: [],
    )
    return AgentTickDeps(
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
        sender=SimpleNamespace(),
        deduper=None,
        rng=SimpleNamespace(),
        workspace_context_fn=lambda: "",
        observe_writer=None,
    )


def test_agent_tick_factory_build_returns_none_when_disabled():
    deps = _build_deps(use_agent_tick=False)
    assert AgentTickFactory(deps).build() is None


def test_agent_tick_factory_build_returns_tick_when_enabled():
    deps = _build_deps(use_agent_tick=True)
    tick = AgentTickFactory(deps).build()
    assert tick is not None


def test_build_proactive_runtime_accepts_light_agent_loop_stub(tmp_path):
    cfg = SimpleNamespace(
        proactive=SimpleNamespace(
            enabled=False,
            skill_actions_enabled=False,
            skill_actions_path="",
            fitbit_enabled=False,
            fitbit_monitor_path="",
        ),
        memory_optimizer_enabled=False,
        memory_optimizer_interval_seconds=3600,
        model="m",
        max_tokens=128,
        light_model="lm",
    )
    tasks, loop = build_proactive_runtime(
        cfg,
        tmp_path,
        session_manager=SimpleNamespace(),
        provider=SimpleNamespace(),
        light_provider=None,
        push_tool=SimpleNamespace(),
        memory_store=None,
        presence=SimpleNamespace(),
        agent_loop=SimpleNamespace(processing_state=None),
    )
    assert tasks == []
    assert loop is None
