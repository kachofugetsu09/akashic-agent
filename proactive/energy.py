"""
proactive/energy.py — 动态电量衰减与主动冲动计算。

核心思路（多时间尺度指数衰减）：
  E(t) = α·exp(-t/τ₁) + β·exp(-t/τ₂) + γ·exp(-t/τ₃)

  τ₁=30min  短时：对话余温
  τ₂=240min 中时：同一天语境
  τ₃=2880min 长时：关系连续性（48h）

电量 → 冲动：
  E 高 (≥ cool_threshold)  → 冲动 = 0，不打扰
  E 低                     → 冲动二次上升
  E < crisis_threshold     → 危机模式，必须找话说
"""
from __future__ import annotations

import math
import random as _random
from datetime import datetime, timezone


def compute_energy(
    last_user_at: datetime | None,
    now: datetime | None = None,
    *,
    alpha: float = 0.50,
    beta: float = 0.35,
    gamma: float = 0.15,
    tau1_min: float = 30.0,
    tau2_min: float = 240.0,
    tau3_min: float = 2880.0,
) -> float:
    """返回 [0, 1] 的当前电量。从未收到消息则返回 0.0。"""
    if last_user_at is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    t = max(0.0, (now - last_user_at).total_seconds() / 60.0)
    return (
        alpha * math.exp(-t / tau1_min)
        + beta  * math.exp(-t / tau2_min)
        + gamma * math.exp(-t / tau3_min)
    )


def urge_base(energy: float, cool_threshold: float = 0.20) -> float:
    """电量低于 cool_threshold 时冲动二次上升，[0, 1]。"""
    energy = max(0.0, energy)
    if energy >= cool_threshold:
        return 0.0
    return min(1.0, ((cool_threshold - energy) / cool_threshold) ** 2)


def time_weight(
    hour: int,
    quiet_start: int = 23,
    quiet_end: int = 8,
    quiet_weight: float = 0.0,
) -> float:
    """基于当地小时的昼夜节律权重。

    quiet zone 返回 quiet_weight（默认 0.0 = 完全静默），其余时段返回 1.0。
    quiet_weight > 0 时静默期仍有小概率触发（如 0.1 = 10% 概率量级）。
    """
    if quiet_start > quiet_end:
        in_quiet = hour >= quiet_start or hour < quiet_end
    else:
        in_quiet = quiet_start <= hour < quiet_end
    return quiet_weight if in_quiet else 1.0


def random_weight(rng: _random.Random | None = None) -> float:
    """随机扰动系数，防止行为过于规律可预测。

    从 Beta(2, 2) 采样（偏中间，极端少），线性映射到 [0.5, 1.5]。
    均值 ≈ 1.0，标准差适中。
    """
    r = rng or _random
    sample = r.betavariate(2, 2)   # [0, 1]，均值 0.5
    return 0.5 + sample            # [0.5, 1.5]


def content_weight(
    new_items: int,
    has_memory: bool,
    is_crisis: bool,
) -> float:
    """轻量预判：不调 LLM，评估"现在有没有话说"。

    new_items  有新 feed 条目
    has_memory 本轮抽到了随机记忆片段
    is_crisis  电量极低，危机模式
    """
    if is_crisis:
        # 危机时哪怕没内容也要找话说，托底 1.0
        base = min(1.0 + new_items * 0.2 + (0.3 if has_memory else 0.0), 2.0)
        return max(base, 1.0)

    w = 0.0
    if new_items > 0:
        w += min(new_items * 0.3, 1.0)   # 每条 +0.3，最多 +1.0
    if has_memory:
        w += 0.3
    return w


def next_tick_interval(
    energy: float,
    *,
    cool_threshold: float = 0.20,
    crisis_threshold: float = 0.05,
    tick_high: int = 7200,
    tick_normal: int = 1800,
    tick_low: int = 900,
    tick_crisis: int = 600,
    tick_jitter: float = 0.3,
    rng: _random.Random | None = None,
) -> int:
    """根据当前电量返回下一次心跳的等待秒数（自适应 + 随机抖动）。

    电量高 → 拉长间隔（刚聊完，别烦）
    电量低 → 缩短间隔（快去找话说）
    tick_jitter 控制随机浮动幅度：0.3 表示 ±30%，让行为不那么机械。
    """
    if energy > 0.50:
        base = tick_high
    elif energy > cool_threshold:
        base = tick_normal
    elif energy > crisis_threshold:
        base = tick_low
    else:
        base = tick_crisis

    if tick_jitter <= 0:
        return base
    r = (rng or _random).uniform(1 - tick_jitter, 1 + tick_jitter)
    return max(1, int(base * r))
