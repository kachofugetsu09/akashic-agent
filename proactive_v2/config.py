from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProactiveConfig:
    """Proactive 配置

    使用预设 + 覆盖的方式配置，大部分算法参数内置在策略中。
    """
    # 必填运行信息
    enabled: bool = False
    default_channel: str = "telegram"
    default_chat_id: str = ""
    model: str = ""

    # Feed Poller 配置
    feed_poller_interval_seconds: int = 150

    # === 以下参数由预设 + 覆盖控制 ===

    # Trigger 配置
    tick_interval_s0: int = 4800
    tick_interval_s1: int = 2400
    tick_jitter: float = 0.3

    # Gate 配置
    judge_send_threshold: float = 0.60

    # AnyAction 配置
    anyaction_enabled: bool = False
    anyaction_daily_max_actions: int = 24
    anyaction_min_interval_seconds: int = 300
    anyaction_probability_min: float = 0.03
    anyaction_probability_max: float = 0.45
    anyaction_idle_scale_minutes: float = 240.0
    anyaction_reset_hour_local: int = 12
    anyaction_timezone: str = "Asia/Shanghai"

    # Safety 配置
    delivery_dedupe_hours: int = 24
    message_dedupe_recent_n: int = 5

    # Context 配置
    context_only_daily_max: int = 1
    context_only_min_interval_hours: int = 12

    # === 策略内置参数（不对外暴露，由 presets.STRATEGY_PARAMS 提供） ===

    # 评分权重
    score_weight_energy: float = 0.40

    # 去重细节
    message_dedupe_enabled: bool = True

    # 其他
    recent_chat_messages: int = 20
    interval_seconds: int = 1800

    # === v2 Agent Tick（唯一实现） ===
    agent_tick_max_steps: int = 35
    agent_tick_model: str = ""
    agent_tick_content_limit: int = 5
    agent_tick_web_fetch_max_chars: int = 8_000
    agent_tick_context_prob: float = 0.03
    agent_tick_delivery_cooldown_hours: int = 1
    drift_enabled: bool = False
    drift_max_steps: int = 20
    drift_min_interval_hours: int = 3
