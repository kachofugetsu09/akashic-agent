from __future__ import annotations

from dataclasses import dataclass, field


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

    # 功能开关
    memory_retrieval_enabled: bool = True
    preference_retrieval_enabled: bool = True
    research_enabled: bool = True
    fitbit_enabled: bool = False

    # Fitbit 配置
    fitbit_url: str = "http://127.0.0.1:18765"
    fitbit_poll_seconds: int = 300
    fitbit_monitor_path: str = ""

    # Feed Poller 配置（保留，因为是独立子系统）
    feed_poller_enabled: bool = True
    feed_poller_interval_seconds: int = 150

    # Interest Filter 配置（保留，因为是独立子系统）
    interest_filter: object = None  # SimpleNamespace

    # === 以下参数由预设 + 覆盖控制 ===

    # Trigger 配置
    tick_interval_s0: int = 4800
    tick_interval_s1: int = 2400
    tick_interval_s2: int = 1080
    tick_interval_s3: int = 420
    tick_jitter: float = 0.3

    # Gate 配置
    score_llm_threshold: float = 0.40
    score_pre_threshold: float = 0.05
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
    llm_reject_cooldown_hours: int = 12
    message_dedupe_recent_n: int = 5

    # Context 配置
    context_only_enabled: bool = True
    context_only_daily_max: int = 1
    context_only_min_interval_hours: int = 12
    context_only_judge_threshold: float = 0.72
    context_only_judge_threshold_with_evidence: float = 0.68

    # === 策略内置参数（不对外暴露，由 presets.STRATEGY_PARAMS 提供） ===

    # 评分权重
    score_weight_energy: float = 0.40
    score_weight_content: float = 0.40
    score_weight_recent: float = 0.20
    score_content_halfsat: float = 3.0
    score_recent_scale: float = 10.0
    decision_score_random_strength: float = 0.0

    # 打断权重
    interrupt_weight_reply: float = 0.35
    interrupt_weight_activity: float = 0.25
    interrupt_weight_fatigue: float = 0.15
    interrupt_activity_decay_minutes: float = 180.0
    interrupt_reply_decay_minutes: float = 120.0
    interrupt_no_reply_decay_minutes: float = 360.0
    interrupt_fatigue_window_hours: int = 24
    interrupt_fatigue_soft_cap: float = 6.0
    interrupt_random_strength: float = 0.12
    interrupt_min_floor: float = 0.08

    # Judge 权重
    judge_weight_urgency: float = 0.15
    judge_weight_balance: float = 0.10
    judge_weight_dynamics: float = 0.10
    judge_weight_information_gap: float = 0.25
    judge_weight_relevance: float = 0.20
    judge_weight_expected_impact: float = 0.20
    judge_urgency_horizon_hours: float = 12.0
    judge_balance_daily_max: int = 8
    judge_veto_balance_min: float = 0.1
    judge_veto_llm_dim_min: int = 2

    # Memory retrieval 细节
    memory_top_k_procedure: int = 4
    memory_top_k_history: int = 6
    memory_query_max_recent_messages: int = 3
    memory_query_max_items: int = 3
    memory_history_gate_enabled: bool = True
    memory_scope_fallback_to_global: bool = False
    memory_trace_enabled: bool = True
    preference_per_source_top_k: int = 2
    preference_max_sources: int = 5
    preference_hyde_enabled: bool = False
    preference_hyde_timeout_ms: int = 2000

    # Research 细节
    research_max_iterations: int = 10
    research_tools: list[str] = field(default_factory=lambda: ["web_search", "web_fetch", "read_file"])
    research_min_body_chars: int = 500
    research_timeout_seconds: int = 30
    research_apply_on_context_only: bool = True
    research_include_all_mcp_tools: bool = True
    research_fail_policy: str = "drop"
    research_transparent_message: str = ""
    research_skip_alert: bool = True

    # 去重细节
    dedupe_seen_ttl_hours: int = 24 * 14
    semantic_dedupe_window_hours: int = 72
    semantic_dedupe_text_max_chars: int = 240
    message_dedupe_enabled: bool = True

    # 其他
    threshold: float = 0.70
    recent_chat_messages: int = 20
    interval_seconds: int = 1800
    use_global_memory: bool = True
    compose_no_content_token: str = "<no_content/>"
    bg_context_main_topic_min_interval_hours: int = 6
    context_as_assist_enabled: bool = True
    sleep_modifier_sleeping: float = 0.15

    # === v2 Agent Tick（唯一实现） ===
    agent_tick_max_steps: int = 20
    agent_tick_model: str = ""
    agent_tick_content_limit: int = 5
    agent_tick_web_fetch_max_chars: int = 8_000
    agent_tick_context_prob: float = 0.03
    agent_tick_delivery_cooldown_hours: int = 1
    drift_enabled: bool = False
    drift_max_steps: int = 20
    drift_dir: str = ""
    drift_min_interval_hours: int = 3
