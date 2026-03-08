from __future__ import annotations

from dataclasses import dataclass, field

from proactive.config import ProactiveConfig


@dataclass
class TelegramChannelConfig:
    token: str
    allow_from: list[str] = field(default_factory=list)


@dataclass
class QQGroupConfig:
    group_id: str
    allow_from: list[str] = field(default_factory=list)
    require_at: bool = True


@dataclass
class QQChannelConfig:
    bot_uin: str
    allow_from: list[str] = field(default_factory=list)
    groups: list[QQGroupConfig] = field(default_factory=list)


@dataclass
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
    qq: QQChannelConfig | None = None
    socket: str = "/tmp/akasic.sock"


@dataclass
class MemoryV2Config:
    enabled: bool = False
    db_path: str = ""
    embed_model: str = "text-embedding-v3"
    retrieve_top_k: int = 8
    top_k_history: int = 8
    top_k_procedure: int = 4
    score_threshold: float = 0.45
    score_threshold_procedure: float = 0.60
    score_threshold_preference: float = 0.60
    score_threshold_event: float = 0.68
    score_threshold_profile: float = 0.68
    relative_delta: float = 0.06
    inject_max_chars: int = 1200
    inject_max_forced: int = 3
    inject_max_procedure_preference: int = 4
    inject_max_event_profile: int = 2
    route_intention_enabled: bool = False
    sop_guard_enabled: bool = True
    gate_llm_timeout_ms: int = 800
    gate_max_tokens: int = 96


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
    memory_optimizer_interval_seconds: int = 3600
    light_model: str = ""
    light_api_key: str = ""
    light_base_url: str = ""
    memory_v2: MemoryV2Config = field(default_factory=MemoryV2Config)
    tool_search_enabled: bool = False


__all__ = [
    "ChannelsConfig",
    "Config",
    "MemoryV2Config",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
]
