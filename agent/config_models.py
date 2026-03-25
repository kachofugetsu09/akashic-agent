from __future__ import annotations

from dataclasses import dataclass, field

from proactive_v2.config import ProactiveConfig


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
    retrieve_top_k: int = 4
    top_k_history: int = 4
    top_k_procedure: int = 4
    score_threshold: float = 0.45
    score_threshold_procedure: float = 0.60
    score_threshold_preference: float = 0.60
    score_threshold_event: float = 0.62
    score_threshold_profile: float = 0.62
    relative_delta: float = 0.06
    inject_max_chars: int = 1200
    inject_max_forced: int = 3
    inject_max_procedure_preference: int = 4
    inject_max_event_profile: int = 2
    inject_line_max: int = 180
    route_intention_enabled: bool = False
    sop_guard_enabled: bool = True
    gate_llm_timeout_ms: int = 800
    gate_max_tokens: int = 96
    hyde_enabled: bool = False
    hyde_timeout_ms: int = 2000
    sufficiency_check_enabled: bool = True
    profile_extraction_enabled: bool = True
    profile_supersede_enabled: bool = True
    dedup_enabled: bool = False
    dedup_similarity_threshold: float = 0.45   # 库内预筛阈值（宽松，找候选送 LLM）
    batch_dedup_threshold: float = 0.90        # 批内去重阈值（严格，同轮同义检测）
    hotness_alpha: float = 0.0                 # 热度融合权重（0=纯语义）
    hotness_half_life_days: float = 14.0       # 热度半衰期（天）


@dataclass
class PeerAgentConfig:
    name: str
    base_url: str
    launcher: list[str]          # 拉起命令，如 ["uv", "run", "python", "-m", "app.a2a_server"]
    cwd: str | None = None       # 子进程工作目录，None 表示继承父进程
    description: str = ""        # 工具描述，用于 LLM 路由；服务器在线时会被 AgentCard 覆盖
    health_path: str = "/health"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 10


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    memory_window: int = 40
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
    spawn_enabled: bool = True
    peer_agents: list[PeerAgentConfig] = field(default_factory=list)


__all__ = [
    "ChannelsConfig",
    "Config",
    "MemoryV2Config",
    "PeerAgentConfig",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
]
