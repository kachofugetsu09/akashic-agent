"""
proactive/energy.py — 动态电量衰减与多维主动冲动计算。

核心思路（多时间尺度指数衰减）：
  E(t) = α·exp(-t/τ₁) + β·exp(-t/τ₂) + γ·exp(-t/τ₃)

  τ₁=30min  短时：对话余温
  τ₂=240min 中时：同一天语境
  τ₃=2880min 长时：关系连续性（48h）

多维打分模型（新）：
  base_score = w_e·D_energy + w_c·D_content + w_r·D_recent

  D_energy  = 1 - energy            互动饥渴度（越久没说话越高）
  D_content = 1 - exp(-n/halfsat)   信息流新鲜度（新条目越多越高）
  D_recent  = log(1+k)/log(1+scale) 对话语境丰富度（近期消息越多越高）

  draw_score = base_score × random_weight()
  draw_score > llm_threshold → 调 LLM 反思
"""

from __future__ import annotations

import math
import random as _random
from datetime import datetime, timezone

# ── 电量计算（保留，loop.py 仍依赖）────────────────────────────────


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
        + beta * math.exp(-t / tau2_min)
        + gamma * math.exp(-t / tau3_min)
    )


# ── 三维贡献函数 ───────────────────────────────────────────────────


def d_energy(energy: float) -> float:
    """互动饥渴度：energy 越低（越久没互动）→ D_energy 越高。

    线性映射：D_energy = 1 - energy，范围 [0, 1]。
    高电量（刚聊完）→ 贡献小但非零，不再作为硬闸。
    """
    return 1.0 - max(0.0, min(1.0, energy))


def d_content(new_items: int, halfsat: float = 3.0) -> float:
    """信息流新鲜度：新条目越多 → D_content 越高。

    指数饱和曲线：D_content = 1 - exp(-new_items / halfsat)
    halfsat=3 时：0条→0.00  1条→0.28  3条→0.63  5条→0.81  10条→0.96
    """
    if new_items <= 0:
        return 0.0
    return 1.0 - math.exp(-max(0, new_items) / max(halfsat, 0.1))


def d_recent(msg_count: int, scale: float = 10.0) -> float:
    """对话语境丰富度：近期消息越多 → D_recent 越高。

    对数归一化：D_recent = log(1+k) / log(1+scale)，上限 1.0。
    scale=10 时：0条→0.00  5条→0.59  10条→0.76  20条→0.92
    """
    if msg_count <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0, msg_count)) / math.log1p(max(scale, 1.0)))


def composite_score(
    de: float,
    dc: float,
    dr: float,
    w_e: float = 0.40,
    w_c: float = 0.40,
    w_r: float = 0.20,
) -> float:
    """三维加权合成，结果裁剪至 [0, 1]。"""
    raw = w_e * de + w_c * dc + w_r * dr
    return max(0.0, min(1.0, raw))


# ── tick 间隔（由 base_score 驱动，替代旧的 energy 驱动）───────────


def next_tick_from_score(
    base_score: float,
    *,
    tick_s3: int = 420,  # base_score > 0.70 → ~7 min
    tick_s2: int = 1080,  # base_score > 0.40 → ~18 min
    tick_s1: int = 2400,  # base_score > 0.20 → ~40 min
    tick_s0: int = 4800,  # base_score ≤ 0.20 → ~80 min
    tick_jitter: float = 0.3,
    rng: _random.Random | None = None,
) -> int:
    """根据 base_score 返回下一次 tick 的等待秒数（含随机抖动）。

    base_score 越高 → 间隔越短 → 单位时间内抽签次数越多 → 越快触发。
    """
    if base_score > 0.70:
        base = tick_s3
    elif base_score > 0.40:
        base = tick_s2
    elif base_score > 0.20:
        base = tick_s1
    else:
        base = tick_s0

    if tick_jitter <= 0:
        return base
    r = (rng or _random).uniform(1.0 - tick_jitter, 1.0 + tick_jitter)
    return max(1, int(base * r))


def random_weight(rng: _random.Random | None = None) -> float:
    """随机扰动系数，防止行为过于规律可预测。

    从 Beta(2, 2) 采样（偏中间，极端少），线性映射到 [0.5, 1.5]。
    均值 ≈ 1.0，标准差适中。
    """
    r = rng or _random
    sample = r.betavariate(2, 2)  # [0, 1]，均值 0.5
    return 0.5 + sample  # [0.5, 1.5]
