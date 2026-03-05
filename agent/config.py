"""
配置加载模块
从 config.json 读取配置，支持 ${ENV_VAR} 格式的环境变量插值。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from proactive.loop import ProactiveConfig
from proactive.interest import InterestFilterConfig

_PRESETS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}

# CLI channel 默认 Unix socket 路径
DEFAULT_SOCKET = "/tmp/akasic.sock"


def _validated_timezone(tz_name: str, *, enabled: bool) -> str:
    """仅当 anyaction_enabled=True 时校验时区合法性，无效则启动时 fail-fast。"""
    if not enabled:
        return tz_name
    try:
        ZoneInfo(tz_name)
        return tz_name
    except Exception:
        raise ValueError(
            f"proactive.anyaction_timezone 无效: {tz_name!r}，"
            "请使用 IANA 格式，如 'Asia/Shanghai'"
        )


@dataclass
class TelegramChannelConfig:
    token: str
    allow_from: list[str] = field(default_factory=list)  # 空 = 允许所有人


@dataclass
class QQGroupConfig:
    group_id: str
    allow_from: list[str] = field(default_factory=list)  # 空 = 群内所有人
    require_at: bool = True  # 仅响应 @ 消息


@dataclass
class QQChannelConfig:
    bot_uin: str  # Bot 的 QQ 号
    allow_from: list[str] = field(default_factory=list)  # 私聊白名单，空 = 允许所有人
    groups: list[QQGroupConfig] = field(default_factory=list)  # 群组配置


@dataclass
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
    qq: QQChannelConfig | None = None
    socket: str = DEFAULT_SOCKET  # IPC server 监听路径


@dataclass
class MemoryV2Config:
    enabled: bool = False
    db_path: str = ""  # 空则用 workspace/memory/memory2.db
    embed_model: str = "text-embedding-v3"
    retrieve_top_k: int = 8
    score_threshold: float = 0.45
    disable_full_memory: bool = False  # True 时不再注入全量 MEMORY.md


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    base_url: str | None = None
    extra_body: dict = field(default_factory=dict)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    memory_optimizer_enabled: bool = True
    memory_optimizer_interval_seconds: int = 3600  # 默认每小时
    # 轻量模型：用于 self-check pass 和 eval judge，默认退化到主模型
    # 可配置独立的 provider（不同 base_url / api_key）
    light_model: str = ""
    light_api_key: str = ""
    light_base_url: str = ""
    memory_v2: MemoryV2Config = field(default_factory=MemoryV2Config)

    @classmethod
    def load(cls, path: str | Path = "config.json") -> Config:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        provider = data["provider"]
        channels_data = data.get("channels", {})

        telegram = None
        if tg := channels_data.get("telegram"):
            telegram = TelegramChannelConfig(
                token=_resolve(tg["token"]),
                allow_from=[str(u) for u in tg.get("allowFrom", [])],
            )

        qq = None
        if qq_data := channels_data.get("qq"):
            groups = [
                QQGroupConfig(
                    group_id=str(g["groupId"]),
                    allow_from=[str(u) for u in g.get("allowFrom", [])],
                    require_at=g.get("requireAt", True),
                )
                for g in qq_data.get("groups", [])
            ]
            qq = QQChannelConfig(
                bot_uin=str(qq_data["bot_uin"]),
                allow_from=[str(u) for u in qq_data.get("allowFrom", [])],
                groups=groups,
            )

        channels = ChannelsConfig(
            telegram=telegram,
            qq=qq,
            socket=channels_data.get("cli", {}).get("socket", DEFAULT_SOCKET),
        )

        # ── proactive ──
        proactive = ProactiveConfig()
        if p := data.get("proactive"):
            if_cfg = p.get("interest_filter", {}) or {}
            proactive = ProactiveConfig(
                enabled=p.get("enabled", False),
                interval_seconds=p.get("interval_seconds", 1800),
                threshold=p.get("threshold", 0.70),
                items_per_source=p.get("items_per_source", 3),
                recent_chat_messages=p.get("recent_chat_messages", 20),
                model=p.get("model", ""),
                default_channel=p.get("default_channel", "telegram"),
                default_chat_id=str(p.get("default_chat_id", "")),
                dedupe_seen_ttl_hours=int(p.get("dedupe_seen_ttl_hours", 24 * 14)),
                delivery_dedupe_hours=int(p.get("delivery_dedupe_hours", 24)),
                only_new_items_trigger=bool(p.get("only_new_items_trigger", True)),
                semantic_dedupe_enabled=bool(p.get("semantic_dedupe_enabled", True)),
                semantic_dedupe_threshold=float(
                    p.get("semantic_dedupe_threshold", 0.90)
                ),
                semantic_dedupe_window_hours=int(
                    p.get("semantic_dedupe_window_hours", 72)
                ),
                semantic_dedupe_max_candidates=int(
                    p.get("semantic_dedupe_max_candidates", 200)
                ),
                semantic_dedupe_ngram=int(p.get("semantic_dedupe_ngram", 3)),
                semantic_dedupe_text_max_chars=int(
                    p.get("semantic_dedupe_text_max_chars", 240)
                ),
                use_global_memory=bool(p.get("use_global_memory", True)),
                global_memory_max_chars=int(p.get("global_memory_max_chars", 3000)),
                interest_filter=InterestFilterConfig(
                    enabled=bool(if_cfg.get("enabled", False)),
                    memory_max_chars=int(if_cfg.get("memory_max_chars", 4000)),
                    keyword_max_count=int(if_cfg.get("keyword_max_count", 80)),
                    min_token_len=int(if_cfg.get("min_token_len", 2)),
                    min_score=float(if_cfg.get("min_score", 0.14)),
                    top_k=int(if_cfg.get("top_k", 10)),
                    exploration_ratio=float(if_cfg.get("exploration_ratio", 0.20)),
                ),
                # ── 多维打分 ──
                score_weight_energy=float(p.get("score_weight_energy", 0.40)),
                score_weight_content=float(p.get("score_weight_content", 0.40)),
                score_weight_recent=float(p.get("score_weight_recent", 0.20)),
                score_content_halfsat=float(p.get("score_content_halfsat", 3.0)),
                score_recent_scale=float(p.get("score_recent_scale", 10.0)),
                score_llm_threshold=float(p.get("score_llm_threshold", 0.40)),
                score_pre_threshold=float(p.get("score_pre_threshold", 0.05)),
                decision_score_random_strength=float(
                    p.get("decision_score_random_strength", 0.0)
                ),
                interrupt_weight_reply=float(p.get("interrupt_weight_reply", 0.35)),
                interrupt_weight_activity=float(
                    p.get("interrupt_weight_activity", 0.25)
                ),
                interrupt_weight_fatigue=float(p.get("interrupt_weight_fatigue", 0.15)),
                interrupt_activity_decay_minutes=float(
                    p.get("interrupt_activity_decay_minutes", 180.0)
                ),
                interrupt_reply_decay_minutes=float(
                    p.get("interrupt_reply_decay_minutes", 120.0)
                ),
                interrupt_no_reply_decay_minutes=float(
                    p.get("interrupt_no_reply_decay_minutes", 360.0)
                ),
                interrupt_fatigue_window_hours=int(
                    p.get("interrupt_fatigue_window_hours", 24)
                ),
                interrupt_fatigue_soft_cap=float(
                    p.get("interrupt_fatigue_soft_cap", 6.0)
                ),
                interrupt_random_strength=float(
                    p.get("interrupt_random_strength", 0.12)
                ),
                interrupt_min_floor=float(p.get("interrupt_min_floor", 0.08)),
                # ── 静默时间窗口（本地时间）──
                # ── tick 间隔（秒，由 base_score 驱动）──
                tick_interval_s0=int(p.get("tick_interval_s0", 4800)),
                tick_interval_s1=int(p.get("tick_interval_s1", 2400)),
                tick_interval_s2=int(p.get("tick_interval_s2", 1080)),
                tick_interval_s3=int(p.get("tick_interval_s3", 420)),
                tick_jitter=float(p.get("tick_jitter", 0.3)),
                anyaction_enabled=bool(p.get("anyaction_enabled", False)),
                anyaction_daily_max_actions=int(
                    p.get("anyaction_daily_max_actions", 24)
                ),
                anyaction_min_interval_seconds=int(
                    p.get("anyaction_min_interval_seconds", 300)
                ),
                anyaction_reset_hour_local=int(p.get("anyaction_reset_hour_local", 12)),
                anyaction_timezone=_validated_timezone(
                    str(p.get("anyaction_timezone", "Asia/Shanghai")),
                    enabled=bool(p.get("anyaction_enabled", False)),
                ),
                anyaction_probability_min=float(
                    p.get("anyaction_probability_min", 0.03)
                ),
                anyaction_probability_max=float(
                    p.get("anyaction_probability_max", 0.45)
                ),
                anyaction_idle_scale_minutes=float(
                    p.get("anyaction_idle_scale_minutes", 240.0)
                ),
                feature_scoring_enabled=bool(p.get("feature_scoring_enabled", False)),
                feature_send_threshold=float(p.get("feature_send_threshold", 0.52)),
                feature_weight_topic_continuity=float(
                    p.get("feature_weight_topic_continuity", 0.24)
                ),
                feature_weight_interest_match=float(
                    p.get("feature_weight_interest_match", 0.24)
                ),
                feature_weight_content_novelty=float(
                    p.get("feature_weight_content_novelty", 0.20)
                ),
                feature_weight_reconnect_value=float(
                    p.get("feature_weight_reconnect_value", 0.16)
                ),
                feature_weight_message_readiness=float(
                    p.get("feature_weight_message_readiness", 0.16)
                ),
                feature_weight_disturb_risk=float(
                    p.get("feature_weight_disturb_risk", 0.70)
                ),
                feature_weight_interrupt_penalty=float(
                    p.get("feature_weight_interrupt_penalty", 0.30)
                ),
                feature_weight_d_recent_bonus=float(
                    p.get("feature_weight_d_recent_bonus", 0.10)
                ),
                feature_weight_d_content_bonus=float(
                    p.get("feature_weight_d_content_bonus", 0.10)
                ),
                feature_weight_d_energy_bonus=float(
                    p.get("feature_weight_d_energy_bonus", 0.08)
                ),
                message_dedupe_enabled=bool(p.get("message_dedupe_enabled", True)),
                message_dedupe_recent_n=int(p.get("message_dedupe_recent_n", 5)),
                # ── LLM 拒绝冷却 ──
                llm_reject_cooldown_hours=max(
                    0, int(p.get("llm_reject_cooldown_hours", 12))
                ),
                # ── Feed 轮询器 ──
                feed_poller_enabled=bool(p.get("feed_poller_enabled", False)),
                feed_poller_interval_seconds=max(
                    5, int(p.get("feed_poller_interval_seconds", 300))
                ),
                feed_poller_fetch_limit=max(
                    1, int(p.get("feed_poller_fetch_limit", 20))
                ),
                feed_poller_buffer_ttl_hours=max(
                    1, int(p.get("feed_poller_buffer_ttl_hours", 48))
                ),
                feed_poller_buffer_max_per_source=max(
                    1, int(p.get("feed_poller_buffer_max_per_source", 100))
                ),
                feed_poller_read_limit=max(0, int(p.get("feed_poller_read_limit", 50))),
                # ── Skill Action ──
                skill_actions_enabled=bool(p.get("skill_actions_enabled", False)),
                skill_actions_path=str(p.get("skill_actions_path", "")),
                # ── Fitbit 睡眠感知 ──
                fitbit_enabled=bool(p.get("fitbit_enabled", False)),
                fitbit_url=str(p.get("fitbit_url", "http://127.0.0.1:18765")),
                fitbit_poll_seconds=max(1, int(p.get("fitbit_poll_seconds", 300))),
                fitbit_monitor_path=str(p.get("fitbit_monitor_path", "")),
                # ── SourceScorer ──
                source_scorer_enabled=bool(p.get("source_scorer_enabled", False)),
                source_scorer_total_budget=max(
                    1, int(p.get("source_scorer_total_budget", 60))
                ),
                source_scorer_min_per_source=max(
                    0, int(p.get("source_scorer_min_per_source", 2))
                ),
                source_scorer_max_per_source=max(
                    1, int(p.get("source_scorer_max_per_source", 20))
                ),
                source_scorer_cache_path=str(p.get("source_scorer_cache_path", "")),
            )

        mv2 = data.get("memory_v2", {})
        memory_v2 = MemoryV2Config(
            enabled=bool(mv2.get("enabled", False)),
            db_path=mv2.get("db_path", ""),
            embed_model=mv2.get("embed_model", "text-embedding-v3"),
            retrieve_top_k=int(mv2.get("retrieve_top_k", 8)),
            score_threshold=float(mv2.get("score_threshold", 0.45)),
            disable_full_memory=bool(mv2.get("disable_full_memory", False)),
        )

        return cls(
            provider=provider,
            model=data["model"],
            api_key=_resolve(data.get("api_key", "")),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            max_tokens=data.get("max_tokens", 8192),
            max_iterations=data.get("max_iterations", 10),
            base_url=data.get("base_url") or _PRESETS.get(provider),
            extra_body=data.get("extra_body", {}),
            channels=channels,
            proactive=proactive,
            memory_optimizer_enabled=bool(data.get("memory_optimizer_enabled", True)),
            memory_optimizer_interval_seconds=int(
                data.get("memory_optimizer_interval_seconds", 3600)
            ),
            light_model=data.get("light_model", ""),
            light_api_key=_resolve(data.get("light_api_key", "")),
            light_base_url=data.get("light_base_url", ""),
            memory_v2=memory_v2,
        )


def _resolve(value: str) -> str:
    resolved = re.sub(
        r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value
    )
    # 若仍是未展开的占位符，尝试从 workspace/memory/<VAR_NAME> 文件读取
    m = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if m:
        key_file = Path.home() / ".akasic" / "workspace" / "memory" / m.group(1)
        if key_file.exists():
            resolved = key_file.read_text(encoding="utf-8").strip()
    return resolved
