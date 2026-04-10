"""
TDD — Phase 4: proactive_v2/agent_tick.py — Pre-gate

测试覆盖：
  - passive_busy_fn 硬 veto
  - delivery_cooldown 阻断
  - AnyAction gate 阻断
  - context gate 概率 + 配额双重检查
  - 所有 gate 通过后进入 loop（不返回 None）
  - pre-gate 失败时不 ACK 任何内容
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.proactive_v2.conftest import (
    FakeRng,
    FakeStateStore,
    cfg_with,
    make_agent_tick,
)


# ── passive_busy_fn ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_passive_busy_returns_none():
    tick = make_agent_tick(passive_busy_fn=lambda sk: True)
    result = await tick.tick()
    assert result is None


@pytest.mark.asyncio
async def test_passive_busy_false_does_not_block():
    tick = make_agent_tick(passive_busy_fn=lambda sk: False)
    result = await tick.tick()
    assert result is not None  # 进入 loop（loop 返回 stub，不是 None）


@pytest.mark.asyncio
async def test_passive_busy_none_fn_does_not_block():
    tick = make_agent_tick(passive_busy_fn=None)
    result = await tick.tick()
    assert result is not None


@pytest.mark.asyncio
async def test_passive_busy_receives_session_key():
    received = []
    tick = make_agent_tick(
        session_key="my_session",
        passive_busy_fn=lambda sk: received.append(sk) or False,
    )
    await tick.tick()
    assert received == ["my_session"]


# ── delivery_cooldown ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delivery_cooldown_blocks_when_count_gt_zero():
    state = FakeStateStore()
    state.set_delivery_count(1)
    tick = make_agent_tick(state_store=state)
    result = await tick.tick()
    assert result is None


@pytest.mark.asyncio
async def test_delivery_cooldown_passes_when_count_zero():
    state = FakeStateStore()
    state.set_delivery_count(0)
    tick = make_agent_tick(state_store=state)
    result = await tick.tick()
    assert result is not None


@pytest.mark.asyncio
async def test_delivery_cooldown_uses_configured_window():
    """state_store 收到正确的 window_hours"""
    received_window = []
    state = FakeStateStore()

    orig = state.count_deliveries_in_window
    def patched(sk, window_hours):
        received_window.append(window_hours)
        return orig(sk, window_hours)
    state.count_deliveries_in_window = patched

    tick = make_agent_tick(
        state_store=state,
        cfg=cfg_with(agent_tick_delivery_cooldown_hours=3),
    )
    await tick.tick()
    assert received_window[0] == 3


# ── AnyAction gate ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anyaction_gate_blocks_when_should_act_false():
    gate = MagicMock()
    gate.should_act.return_value = (False, {"reason": "quota_exhausted"})
    tick = make_agent_tick(any_action_gate=gate)
    result = await tick.tick()
    assert result is None


@pytest.mark.asyncio
async def test_anyaction_gate_passes_when_should_act_true():
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    tick = make_agent_tick(any_action_gate=gate)
    result = await tick.tick()
    assert result is not None


@pytest.mark.asyncio
async def test_anyaction_gate_none_skips_check():
    """gate=None 时跳过检查，不阻断"""
    tick = make_agent_tick(any_action_gate=None)
    result = await tick.tick()
    assert result is not None


@pytest.mark.asyncio
async def test_anyaction_gate_called_with_now_utc():
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    tick = make_agent_tick(any_action_gate=gate)
    await tick.tick()
    call_kwargs = gate.should_act.call_args[1]
    assert "now_utc" in call_kwargs
    assert isinstance(call_kwargs["now_utc"], datetime)


@pytest.mark.asyncio
async def test_anyaction_gate_called_with_last_user_at():
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    last_user_at = datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc)
    tick = make_agent_tick(
        any_action_gate=gate,
        last_user_at_fn=lambda: last_user_at,
    )
    await tick.tick()
    call_kwargs = gate.should_act.call_args[1]
    assert call_kwargs["last_user_at"] == last_user_at


# ── Context gate — 概率 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_gate_prob_zero_always_closed():
    state = FakeStateStore()
    state.set_context_only_count(0)
    state.set_last_context_only_at(None)
    tick = make_agent_tick(
        cfg=cfg_with(agent_tick_context_prob=0.0),
        state_store=state,
        rng=FakeRng(value=0.0),
    )
    result = await tick.tick()
    assert result is not None  # 不阻断，只是 context_as_fallback_open=False
    # 验证 ctx.context_as_fallback_open — 通过 loop_ctx 回调暴露
    ctx = tick.last_ctx
    assert ctx.context_as_fallback_open is False


@pytest.mark.asyncio
async def test_context_gate_prob_one_opens_when_quota_ok():
    state = FakeStateStore()
    state.set_context_only_count(0)
    state.set_last_context_only_at(None)
    tick = make_agent_tick(
        cfg=cfg_with(
            agent_tick_context_prob=1.0,
            context_only_daily_max=3,
            context_only_min_interval_hours=12,
        ),
        state_store=state,
        rng=FakeRng(value=0.0),  # 0.0 < 1.0 → 开
    )
    await tick.tick()
    assert tick.last_ctx.context_as_fallback_open is True


# ── Context gate — 配额 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_gate_quota_exceeded_closes():
    state = FakeStateStore()
    state.set_context_only_count(1)  # 已达 daily_max=1
    state.set_last_context_only_at(None)
    tick = make_agent_tick(
        cfg=cfg_with(
            agent_tick_context_prob=1.0,
            context_only_daily_max=1,
        ),
        state_store=state,
        rng=FakeRng(value=0.0),
    )
    await tick.tick()
    assert tick.last_ctx.context_as_fallback_open is False


@pytest.mark.asyncio
async def test_context_gate_min_interval_closes():
    state = FakeStateStore()
    state.set_context_only_count(0)
    # 2h 前发过，min_interval=12h → 未满足
    state.set_last_context_only_at(
        datetime.now(timezone.utc) - timedelta(hours=2)
    )
    tick = make_agent_tick(
        cfg=cfg_with(
            agent_tick_context_prob=1.0,
            context_only_min_interval_hours=12,
            context_only_daily_max=3,
        ),
        state_store=state,
        rng=FakeRng(value=0.0),
    )
    await tick.tick()
    assert tick.last_ctx.context_as_fallback_open is False


@pytest.mark.asyncio
async def test_context_gate_interval_satisfied_opens():
    state = FakeStateStore()
    state.set_context_only_count(0)
    # 13h 前发过，min_interval=12h → 已满足
    state.set_last_context_only_at(
        datetime.now(timezone.utc) - timedelta(hours=13)
    )
    tick = make_agent_tick(
        cfg=cfg_with(
            agent_tick_context_prob=1.0,
            context_only_min_interval_hours=12,
            context_only_daily_max=3,
        ),
        state_store=state,
        rng=FakeRng(value=0.0),
    )
    await tick.tick()
    assert tick.last_ctx.context_as_fallback_open is True


@pytest.mark.asyncio
async def test_drift_interval_blocks_recent_drift():
    state = FakeStateStore()
    state.set_last_drift_at(datetime.now(timezone.utc) - timedelta(hours=1))
    drift_runner = MagicMock()
    tick = make_agent_tick(
        cfg=cfg_with(drift_enabled=True, drift_min_interval_hours=3, agent_tick_context_prob=0.0),
        state_store=state,
        rng=FakeRng(value=1.0),
        llm_fn=AsyncMock(return_value=None),
        drift_runner=drift_runner,
    )
    await tick.tick()
    assert tick.last_ctx.drift_entered is False
    assert tick.last_ctx.skip_reason == "no_content"
    drift_runner.run.assert_not_called()


@pytest.mark.asyncio
async def test_drift_interval_allows_after_window():
    from proactive_v2.drift_runner import DriftRunner
    from proactive_v2.drift_state import DriftStateStore
    from proactive_v2.drift_tools import DriftToolDeps
    from tests.proactive_v2.conftest import FakeLLM
    from pathlib import Path

    state = FakeStateStore()
    state.set_last_drift_at(datetime.now(timezone.utc) - timedelta(hours=4))
    llm = FakeLLM([
        ("finish_drift", {"skill_used": "explore-curiosity", "one_line": "x", "next": "y"}),
    ])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        skill_dir = tmp_path / "skills" / "explore-curiosity"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: explore-curiosity\ndescription: x\n---\n",
            encoding="utf-8",
        )
        tick = make_agent_tick(
            cfg=cfg_with(drift_enabled=True, drift_min_interval_hours=3, drift_dir=str(tmp_path), agent_tick_context_prob=0.0),
            state_store=state,
            llm_fn=llm,
            rng=FakeRng(value=1.0),
            drift_runner=DriftRunner(
                store=DriftStateStore(tmp_path),
                tool_deps=DriftToolDeps(drift_dir=tmp_path, store=DriftStateStore(tmp_path)),
                max_steps=5,
            ),
        )
        await tick.tick()
        assert tick.last_ctx.drift_entered is True
        assert state.drift_run_marked is True


# ── Gate 顺序：passive_busy 最先 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_passive_busy_checked_before_delivery_cooldown():
    """passive_busy=True 时，state_store 不应被调用"""
    state = FakeStateStore()
    called = []
    orig = state.count_deliveries_in_window
    state.count_deliveries_in_window = lambda sk, wh: called.append("delivery") or orig(sk, wh)

    tick = make_agent_tick(
        passive_busy_fn=lambda sk: True,
        state_store=state,
    )
    await tick.tick()
    assert "delivery" not in called


# ── Pre-gate 失败不进 loop ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pregate_fail_does_not_call_alert_fn():
    alert_fn = AsyncMock(return_value=[])
    from proactive_v2.tools import ToolDeps
    from proactive_v2.gateway import GatewayDeps
    deps = ToolDeps(
    )
    tick = make_agent_tick(
        passive_busy_fn=lambda sk: True,
        tool_deps=deps,
        gateway_deps=GatewayDeps(
            alert_fn=alert_fn,
            feed_fn=AsyncMock(return_value=[]),
        ),
    )
    await tick.tick()
    alert_fn.assert_not_called()


# ── 所有 gate 通过 ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_gates_pass_returns_non_none():
    state = FakeStateStore()
    state.set_delivery_count(0)
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    tick = make_agent_tick(
        passive_busy_fn=lambda sk: False,
        state_store=state,
        any_action_gate=gate,
    )
    result = await tick.tick()
    assert result is not None
