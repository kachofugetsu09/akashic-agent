"""
TDD for proactive/energy.py

测试覆盖：
  - compute_energy: 多时间尺度衰减
  - random_weight: 随机扰动
"""

from datetime import datetime, timezone, timedelta
from proactive_v2.energy import compute_energy


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


# ── random_weight ─────────────────────────────────────────────────

from proactive_v2.energy import random_weight
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
