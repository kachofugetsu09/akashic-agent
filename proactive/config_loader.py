"""Proactive 配置加载和验证"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from proactive.config import ProactiveConfig
from proactive.presets import ALLOWED_OVERRIDE_KEYS, PRESETS, STRATEGY_PARAMS


class ProactiveConfigError(Exception):
    """Proactive 配置错误"""
    pass


def _validate_preset_name(preset: str) -> None:
    """验证预设名称"""
    if preset not in PRESETS:
        raise ProactiveConfigError(
            f"无效的 preset: '{preset}'。"
            f"只允许: {', '.join(PRESETS.keys())}"
        )


def _validate_overrides(overrides: dict[str, Any]) -> None:
    """验证 overrides 只包含白名单键"""
    for category, values in overrides.items():
        if category not in ALLOWED_OVERRIDE_KEYS:
            raise ProactiveConfigError(
                f"overrides 中的非法类别: '{category}'。"
                f"只允许: {', '.join(ALLOWED_OVERRIDE_KEYS.keys())}"
            )

        if not isinstance(values, dict):
            raise ProactiveConfigError(
                f"overrides.{category} 必须是字典，当前类型: {type(values).__name__}"
            )

        allowed = ALLOWED_OVERRIDE_KEYS[category]
        for key in values.keys():
            if key not in allowed:
                raise ProactiveConfigError(
                    f"overrides.{category}.{key} 不在白名单中。"
                    f"允许的键: {', '.join(sorted(allowed))}"
                )


def _validate_ranges(config: dict[str, Any]) -> None:
    """验证参数范围"""
    # 阈值类必须 0~1
    threshold_keys = [
        "score_llm_threshold",
        "score_pre_threshold",
        "judge_send_threshold",
        "context_only_judge_threshold",
        "context_only_judge_threshold_with_evidence",
        "anyaction_probability_min",
        "anyaction_probability_max",
    ]
    for key in threshold_keys:
        if key in config:
            val = config[key]
            if not (0 <= val <= 1):
                raise ProactiveConfigError(
                    f"{key} 必须在 [0, 1] 范围内，当前值: {val}"
                )

    # probability_min <= probability_max
    if "anyaction_probability_min" in config and "anyaction_probability_max" in config:
        pmin = config["anyaction_probability_min"]
        pmax = config["anyaction_probability_max"]
        if pmin > pmax:
            raise ProactiveConfigError(
                f"anyaction_probability_min ({pmin}) 不能大于 "
                f"anyaction_probability_max ({pmax})"
            )

    # tick_interval_s0 >= s1 >= s2 >= s3 >= 1
    intervals = [
        config.get("tick_interval_s0"),
        config.get("tick_interval_s1"),
        config.get("tick_interval_s2"),
        config.get("tick_interval_s3"),
    ]
    if all(x is not None for x in intervals):
        for i in range(len(intervals) - 1):
            if intervals[i] < intervals[i + 1]:
                raise ProactiveConfigError(
                    f"tick_interval 必须递减: s{i} ({intervals[i]}) < s{i+1} ({intervals[i+1]})"
                )
        if intervals[-1] < 1:
            raise ProactiveConfigError(
                f"tick_interval_s3 必须 >= 1，当前值: {intervals[-1]}"
            )

    # context_only_judge_threshold_with_evidence <= context_only_judge_threshold
    if "context_only_judge_threshold" in config and "context_only_judge_threshold_with_evidence" in config:
        with_ev = config["context_only_judge_threshold_with_evidence"]
        without_ev = config["context_only_judge_threshold"]
        if with_ev > without_ev:
            raise ProactiveConfigError(
                f"context_only_judge_threshold_with_evidence ({with_ev}) "
                f"不能大于 context_only_judge_threshold ({without_ev})"
            )


def _check_forbidden_keys(p: dict[str, Any]) -> None:
    """检查是否有旧的平铺键直接出现在 proactive 根下"""
    # 允许的根级键
    allowed_root_keys = {
        "enabled",
        "default_channel",
        "default_chat_id",
        "model",
        "preset",
        "features",
        "overrides",
        # Feed poller（独立子系统）
        "feed_poller_enabled",
        "feed_poller_interval_seconds",
        # Interest filter（独立子系统）
        "interest_filter",
        # Fitbit（独立子系统）
        "fitbit_url",
        "fitbit_poll_seconds",
        "fitbit_monitor_path",
        # v2 Agent Tick（独立子系统）
        "agent_tick",
    }

    forbidden = set(p.keys()) - allowed_root_keys
    if forbidden:
        raise ProactiveConfigError(
            f"proactive 配置中出现非法的根级键: {', '.join(sorted(forbidden))}。\n"
            f"请使用 preset + overrides 方式配置。\n"
            f"允许的根级键: {', '.join(sorted(allowed_root_keys))}"
        )


def _validate_feature_keys(features: dict[str, Any]) -> None:
    allowed = {
        "memory_retrieval_enabled",
        "preference_retrieval_enabled",
        "research_enabled",
        "fitbit_enabled",
    }
    forbidden = set(features.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.features 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def _validate_agent_tick_keys(agent_tick: dict[str, Any]) -> None:
    allowed = {
        "model",
        "max_steps",
        "content_limit",
        "web_fetch_max_chars",
        "context_prob",
        "delivery_cooldown_hours",
    }
    forbidden = set(agent_tick.keys()) - allowed
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.agent_tick 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(allowed))}"
        )


def load_proactive_config(p: dict[str, Any]) -> ProactiveConfig:
    """从配置字典加载 ProactiveConfig

    Args:
        p: proactive 配置字典

    Returns:
        ProactiveConfig 实例

    Raises:
        ProactiveConfigError: 配置错误时抛出
    """
    # 检查是否有非法的根级键
    _check_forbidden_keys(p)

    # 必填字段
    enabled = p.get("enabled", False)
    default_channel = p.get("default_channel", "telegram")
    default_chat_id = str(p.get("default_chat_id", ""))
    model = p.get("model", "")

    # 预设名称（必填）
    preset_name = p.get("preset")
    if not preset_name:
        raise ProactiveConfigError(
            "proactive.preset 是必填字段。"
            f"可选值: {', '.join(PRESETS.keys())}"
        )

    _validate_preset_name(preset_name)
    preset = PRESETS[preset_name]

    # 功能开关
    features = p.get("features", {})
    _validate_feature_keys(features)
    memory_retrieval_enabled = features.get("memory_retrieval_enabled", True)
    preference_retrieval_enabled = features.get("preference_retrieval_enabled", True)
    research_enabled = features.get("research_enabled", True)
    fitbit_enabled = features.get("fitbit_enabled", False)
    # Fitbit 配置
    fitbit_url = p.get("fitbit_url", "http://127.0.0.1:18765")
    fitbit_poll_seconds = p.get("fitbit_poll_seconds", 300)
    fitbit_monitor_path = p.get("fitbit_monitor_path", "")

    # Feed Poller 配置
    feed_poller_enabled = p.get("feed_poller_enabled", True)
    feed_poller_interval_seconds = p.get("feed_poller_interval_seconds", 150)

    # 合并预设和覆盖
    overrides = p.get("overrides", {})
    _validate_overrides(overrides)

    # 构建最终配置
    final_config = {}

    # 1. 应用预设
    for category, values in preset.items():
        final_config.update(values)

    # 2. 应用策略内置参数
    final_config.update(STRATEGY_PARAMS)

    # 3. 应用覆盖
    for category, values in overrides.items():
        final_config.update(values)

    # 验证范围
    _validate_ranges(final_config)

    # 移除已经显式设置的键，避免冲突
    explicit_keys = {
        "enabled",
        "default_channel",
        "default_chat_id",
        "model",
        "memory_retrieval_enabled",
        "preference_retrieval_enabled",
        "research_enabled",
        "fitbit_enabled",
        "fitbit_url",
        "fitbit_poll_seconds",
        "fitbit_monitor_path",
        "feed_poller_enabled",
        "feed_poller_interval_seconds",
    }
    for key in explicit_keys:
        final_config.pop(key, None)

    # 构建 ProactiveConfig
    config = ProactiveConfig(
        enabled=enabled,
        default_channel=default_channel,
        default_chat_id=default_chat_id,
        model=model,
        memory_retrieval_enabled=memory_retrieval_enabled,
        preference_retrieval_enabled=preference_retrieval_enabled,
        research_enabled=research_enabled,
        fitbit_enabled=fitbit_enabled,
        fitbit_url=fitbit_url,
        fitbit_poll_seconds=fitbit_poll_seconds,
        fitbit_monitor_path=fitbit_monitor_path,
        feed_poller_enabled=feed_poller_enabled,
        feed_poller_interval_seconds=feed_poller_interval_seconds,
        **final_config,
    )

    # Interest Filter 配置（独立子系统）
    interest_filter = p.get("interest_filter") or {}
    config.interest_filter = SimpleNamespace(
        enabled=bool(interest_filter.get("enabled", False)),
        memory_max_chars=max(1, int(interest_filter.get("memory_max_chars", 4000))),
        keyword_max_count=max(1, int(interest_filter.get("keyword_max_count", 80))),
        min_token_len=max(1, int(interest_filter.get("min_token_len", 2))),
        min_score=float(interest_filter.get("min_score", 0.14)),
        top_k=max(1, int(interest_filter.get("top_k", 10))),
        exploration_ratio=float(interest_filter.get("exploration_ratio", 0.2)),
    )

    # v2 Agent Tick 配置（独立子系统）
    at = p.get("agent_tick") or {}
    _validate_agent_tick_keys(at)
    if at.get("model"):
        config.agent_tick_model = str(at["model"])
    if "max_steps" in at:
        config.agent_tick_max_steps = max(1, int(at["max_steps"]))
    if "content_limit" in at:
        config.agent_tick_content_limit = max(1, int(at["content_limit"]))
    if "web_fetch_max_chars" in at:
        config.agent_tick_web_fetch_max_chars = max(1000, int(at["web_fetch_max_chars"]))
    if "context_prob" in at:
        config.agent_tick_context_prob = max(0.0, min(1.0, float(at["context_prob"])))
    if "delivery_cooldown_hours" in at:
        config.agent_tick_delivery_cooldown_hours = max(0, int(at["delivery_cooldown_hours"]))

    return config
