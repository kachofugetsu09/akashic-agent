"""Proactive 预设配置定义"""

from __future__ import annotations

from typing import TypedDict


class TriggerPreset(TypedDict):
    tick_interval_s0: int
    tick_interval_s1: int
    tick_interval_s2: int
    tick_interval_s3: int
    tick_jitter: float


class GatePreset(TypedDict):
    score_llm_threshold: float
    judge_send_threshold: float
    judge_balance_daily_max: int


class AnyActionPreset(TypedDict):
    anyaction_enabled: bool
    anyaction_daily_max_actions: int
    anyaction_min_interval_seconds: int
    anyaction_probability_min: float
    anyaction_probability_max: float
    anyaction_idle_scale_minutes: float
    anyaction_reset_hour_local: int
    anyaction_timezone: str


class SafetyPreset(TypedDict):
    delivery_dedupe_hours: int
    message_dedupe_recent_n: int


class ContextPreset(TypedDict):
    context_only_daily_max: int
    context_only_min_interval_hours: int
    context_only_judge_threshold: float
    context_only_judge_threshold_with_evidence: float


class PresetConfig(TypedDict):
    trigger: TriggerPreset
    gate: GatePreset
    anyaction: AnyActionPreset
    safety: SafetyPreset
    context: ContextPreset


# 预设定义
PRESETS: dict[str, PresetConfig] = {
    "daily": {
        # 基于你当前实际使用的配置
        "trigger": {
            "tick_interval_s0": 480,  # 8分钟
            "tick_interval_s1": 240,  # 4分钟
            "tick_interval_s2": 120,  # 2分钟
            "tick_interval_s3": 60,   # 1分钟
            "tick_jitter": 0.2,
        },
        "gate": {
            "score_llm_threshold": 0.14,
            "judge_send_threshold": 0.60,
            "judge_balance_daily_max": 48,
        },
        "anyaction": {
            "anyaction_enabled": True,
            "anyaction_daily_max_actions": 48,
            "anyaction_min_interval_seconds": 180,
            "anyaction_probability_min": 0.2,
            "anyaction_probability_max": 0.82,
            "anyaction_idle_scale_minutes": 30.0,
            "anyaction_reset_hour_local": 12,
            "anyaction_timezone": "Asia/Shanghai",
        },
        "safety": {
            "delivery_dedupe_hours": 10,
            "message_dedupe_recent_n": 5,
        },
        "context": {
            "context_only_daily_max": 1,
            "context_only_min_interval_hours": 12,
            "context_only_judge_threshold": 0.72,
            "context_only_judge_threshold_with_evidence": 0.68,
        },
    },
    "dev_verify": {
        # 改完代码后 2-5 分钟内可见效果
        "trigger": {
            "tick_interval_s0": 60,   # 1分钟
            "tick_interval_s1": 30,   # 30秒
            "tick_interval_s2": 15,   # 15秒
            "tick_interval_s3": 10,   # 10秒
            "tick_jitter": 0.0,       # 无抖动，精确触发
        },
        "gate": {
            "score_llm_threshold": 0.08,  # 极低门槛
            "judge_send_threshold": 0.28,
            "judge_balance_daily_max": 48,
        },
        "anyaction": {
            "anyaction_enabled": True,
            "anyaction_daily_max_actions": 999,
            "anyaction_min_interval_seconds": 20,
            "anyaction_probability_min": 0.75,
            "anyaction_probability_max": 0.98,
            "anyaction_idle_scale_minutes": 15.0,  # 15分钟就算空闲
            "anyaction_reset_hour_local": 12,
            "anyaction_timezone": "Asia/Shanghai",
        },
        "safety": {
            "delivery_dedupe_hours": 1,
            "message_dedupe_recent_n": 5,
        },
        "context": {
            "context_only_daily_max": 20,
            "context_only_min_interval_hours": 1,
            "context_only_judge_threshold": 0.30,
            "context_only_judge_threshold_with_evidence": 0.25,
        },
    },
    "quiet": {
        # 低打扰模式，比 daily 慢 3-4 倍
        "trigger": {
            "tick_interval_s0": 1800,  # 30分钟
            "tick_interval_s1": 900,   # 15分钟
            "tick_interval_s2": 480,   # 8分钟
            "tick_interval_s3": 240,   # 4分钟
            "tick_jitter": 0.3,
        },
        "gate": {
            "score_llm_threshold": 0.35,
            "judge_send_threshold": 0.75,
            "judge_balance_daily_max": 12,
        },
        "anyaction": {
            "anyaction_enabled": True,
            "anyaction_daily_max_actions": 12,
            "anyaction_min_interval_seconds": 600,  # 10分钟
            "anyaction_probability_min": 0.05,
            "anyaction_probability_max": 0.30,
            "anyaction_idle_scale_minutes": 120.0,  # 2小时
            "anyaction_reset_hour_local": 12,
            "anyaction_timezone": "Asia/Shanghai",
        },
        "safety": {
            "delivery_dedupe_hours": 24,
            "message_dedupe_recent_n": 8,
        },
        "context": {
            "context_only_daily_max": 1,
            "context_only_min_interval_hours": 24,
            "context_only_judge_threshold": 0.80,
            "context_only_judge_threshold_with_evidence": 0.75,
        },
    },
}


# 策略内置参数（不对外暴露）
STRATEGY_PARAMS = {
    # 评分权重（使用旧配置的实际值）
    "score_weight_energy": 0.35,
    "score_recent_scale": 8.0,
    # 打断权重（使用旧配置的实际值）
    "interrupt_weight_reply": 0.35,
    "interrupt_weight_activity": 0.25,
    "interrupt_weight_fatigue": 0.20,
    "interrupt_activity_decay_minutes": 180.0,
    "interrupt_reply_decay_minutes": 120.0,
    "interrupt_no_reply_decay_minutes": 180.0,
    "interrupt_fatigue_window_hours": 24,
    "interrupt_fatigue_soft_cap": 4.0,
    "interrupt_random_strength": 0.16,
    "interrupt_min_floor": 0.06,
    # Judge 权重
    "judge_weight_urgency": 0.15,
    "judge_weight_balance": 0.10,
    "judge_weight_dynamics": 0.10,
    "judge_weight_information_gap": 0.25,
    "judge_weight_relevance": 0.20,
    "judge_weight_expected_impact": 0.20,
    "judge_urgency_horizon_hours": 36.0,
    # judge_balance_daily_max 已移到预设配置
    "judge_veto_balance_min": 0.1,
    "judge_veto_llm_dim_min": 2,
    # Memory retrieval 细节
    "memory_history_gate_enabled": True,
    # 去重细节
    "message_dedupe_enabled": True,
    # 其他
    "threshold": 0.70,
    "recent_chat_messages": 20,
    "interval_seconds": 1800,
    "sleep_modifier_sleeping": 0.15,
}


# Overrides 白名单
ALLOWED_OVERRIDE_KEYS = {
    "trigger": {
        "tick_interval_s0",
        "tick_interval_s1",
        "tick_interval_s2",
        "tick_interval_s3",
        "tick_jitter",
    },
    "gate": {
        "score_llm_threshold",
        "judge_send_threshold",
        "judge_balance_daily_max",
    },
    "anyaction": {
        "anyaction_enabled",
        "anyaction_daily_max_actions",
        "anyaction_min_interval_seconds",
        "anyaction_probability_min",
        "anyaction_probability_max",
        "anyaction_idle_scale_minutes",
        "anyaction_reset_hour_local",
        "anyaction_timezone",
    },
    "safety": {
        "delivery_dedupe_hours",
        "message_dedupe_recent_n",
    },
    "context": {
        "context_only_daily_max",
        "context_only_min_interval_hours",
        "context_only_judge_threshold",
        "context_only_judge_threshold_with_evidence",
    },
}
