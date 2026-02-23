"""
TDD for proactive/energy.py

测试覆盖：
  - compute_energy: 多时间尺度衰减
  - urge_base: 二次冲动函数
  - time_weight: 昼夜节律
  - next_tick_interval: 自适应心跳间隔
"""
from datetime import datetime, timezone, timedelta
import pytest
from proactive.energy import compute_energy, urge_base, time_weight, next_tick_interval


def _ago(minutes: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


# ── compute_energy ────────────────────────────────────────────────

def test_energy_is_one_at_t_zero():
    now = datetime.now(timezone.utc)
    e = compute_energy(now, now)
    assert abs(e - 1.0) < 1e-9


def test_energy_is_zero_when_never_messaged():
    assert compute_energy(None) == 0.0


def test_energy_decays_below_half_after_one_hour():
    last = _ago(60)
    e = compute_energy(last)
    assert e < 0.5


def test_energy_below_cool_threshold_after_24h():
    """24h 后电量应低于默认 cool_threshold=0.20，触发冲动区。"""
    last = _ago(60 * 24)
    e = compute_energy(last)
    assert e < 0.20


def test_energy_below_crisis_threshold_after_72h():
    """72h 后电量应低于默认 crisis_threshold=0.05，触发危机模式。"""
    last = _ago(60 * 72)
    e = compute_energy(last)
    assert e < 0.05


def test_energy_is_strictly_decreasing():
    times = [_ago(m) for m in [0, 30, 120, 480, 1440, 4320]]
    energies = [compute_energy(t) for t in times]
    for a, b in zip(energies, energies[1:]):
        assert a > b, f"energy should decrease: {a} > {b}"


def test_energy_stays_positive_after_long_time():
    """衰减应趋近 0 但不变成负数。"""
    last = _ago(60 * 24 * 30)
    e = compute_energy(last)
    assert 0.0 <= e < 0.001


def test_energy_accepts_custom_decay_params():
    """快速衰减：tau1=1min，30 分钟后接近 0。"""
    now = datetime.now(timezone.utc)
    last = now - timedelta(minutes=30)
    e = compute_energy(last, now, tau1_min=1.0, tau2_min=2.0, tau3_min=5.0)
    assert e < 0.01


# ── urge_base ────────────────────────────────────────────────────

def test_urge_is_zero_when_energy_above_threshold():
    assert urge_base(0.20) == 0.0
    assert urge_base(0.50) == 0.0
    assert urge_base(1.00) == 0.0


def test_urge_is_zero_exactly_at_threshold():
    assert urge_base(0.20, cool_threshold=0.20) == 0.0


def test_urge_is_one_when_energy_is_zero():
    assert urge_base(0.0) == 1.0


def test_urge_is_quarter_at_half_threshold():
    """E = θ/2 → urge = 0.25（二次曲线中点）。"""
    theta = 0.20
    e = theta / 2
    u = urge_base(e, cool_threshold=theta)
    assert abs(u - 0.25) < 1e-9


def test_urge_rises_as_energy_drops():
    energies = [0.19, 0.15, 0.10, 0.05, 0.01, 0.0]
    urges = [urge_base(e) for e in energies]
    for a, b in zip(urges, urges[1:]):
        assert a < b, f"urge should increase as energy drops: {a} < {b}"


def test_urge_clamps_at_one_for_negative_energy():
    """防御：负电量也不超过 1.0。"""
    assert urge_base(-0.1) == 1.0


# ── time_weight ───────────────────────────────────────────────────

def test_time_weight_is_one_during_daytime(
    # 默认 quiet 23:00-08:00，9~22 全段白天
):
    for hour in range(9, 23):
        w = time_weight(hour)
        assert w == 1.0, f"hour={hour} should be 1.0, got {w}"


def test_time_weight_is_zero_in_deep_quiet_hours():
    for hour in [0, 1, 2, 3, 4, 5, 6, 7]:
        w = time_weight(hour)
        assert w == 0.0, f"hour={hour} should be 0.0, got {w}"


def test_time_weight_is_zero_at_quiet_start():
    assert time_weight(23) == 0.0


def test_time_weight_custom_quiet_window():
    # 仅 2:00-4:00 静音，其他时间全开
    assert time_weight(3, quiet_start=2, quiet_end=4) == 0.0
    assert time_weight(5, quiet_start=2, quiet_end=4) == 1.0
    assert time_weight(1, quiet_start=2, quiet_end=4) == 1.0


# ── next_tick_interval ────────────────────────────────────────────

def test_tick_interval_high_when_energy_above_half():
    interval = next_tick_interval(0.60)
    assert interval == 7200


def test_tick_interval_normal_when_energy_between_cool_and_half():
    interval = next_tick_interval(0.30)
    assert interval == 1800


def test_tick_interval_low_when_energy_below_cool_threshold():
    interval = next_tick_interval(0.15)
    assert interval == 900


def test_tick_interval_crisis_when_energy_below_crisis_threshold():
    interval = next_tick_interval(0.03)
    assert interval == 600


def test_tick_interval_crisis_at_zero_energy():
    interval = next_tick_interval(0.0)
    assert interval == 600


def test_tick_interval_decreases_as_energy_drops():
    intervals = [
        next_tick_interval(e)
        for e in [0.80, 0.40, 0.15, 0.02]
    ]
    for a, b in zip(intervals, intervals[1:]):
        assert a >= b, f"interval should not increase as energy drops: {a} >= {b}"


# ── random_weight ─────────────────────────────────────────────────

from proactive.energy import random_weight
import random as _random_module


def test_random_weight_is_in_valid_range():
    for _ in range(200):
        w = random_weight()
        assert 0.5 <= w <= 1.5, f"random_weight out of range: {w}"


def test_random_weight_is_deterministic_with_seed():
    rng = _random_module.Random(42)
    w1 = random_weight(rng=rng)
    rng2 = _random_module.Random(42)
    w2 = random_weight(rng=rng2)
    assert w1 == w2


def test_random_weight_varies_across_calls():
    weights = [random_weight() for _ in range(50)]
    assert len(set(weights)) > 1, "random_weight should not be constant"


def test_random_weight_roughly_centered():
    """均值应在 0.9 ~ 1.1 之间（Beta(2,2) 中心为 0.5，映射后中心为 1.0）。"""
    weights = [random_weight() for _ in range(2000)]
    mean = sum(weights) / len(weights)
    assert 0.9 <= mean <= 1.1, f"mean={mean:.3f} unexpectedly off-center"


# ── content_weight ────────────────────────────────────────────────

from proactive.energy import content_weight


def test_content_weight_zero_when_no_content_no_crisis():
    assert content_weight(new_items=0, has_memory=False, is_crisis=False) == 0.0


def test_content_weight_positive_with_new_items():
    w = content_weight(new_items=3, has_memory=False, is_crisis=False)
    assert w > 0.0


def test_content_weight_positive_with_memory_only():
    w = content_weight(new_items=0, has_memory=True, is_crisis=False)
    assert w > 0.0


def test_content_weight_at_least_one_in_crisis():
    w = content_weight(new_items=0, has_memory=False, is_crisis=True)
    assert w >= 1.0


def test_content_weight_crisis_boosts_even_with_content():
    w_normal = content_weight(new_items=2, has_memory=True, is_crisis=False)
    w_crisis = content_weight(new_items=2, has_memory=True, is_crisis=True)
    assert w_crisis >= w_normal


def test_content_weight_more_items_more_weight():
    w_few = content_weight(new_items=1, has_memory=False, is_crisis=False)
    w_many = content_weight(new_items=5, has_memory=False, is_crisis=False)
    assert w_many >= w_few
