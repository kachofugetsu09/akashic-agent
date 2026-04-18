from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    socket: str = "/tmp/akashic.sock"


@dataclass
class MemoryV2Config:
    enabled: bool = False
    db_path: str = ""
    embed_model: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = ""
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
    procedure_guard_enabled: bool = True
    gate_llm_timeout_ms: int = 800
    gate_max_tokens: int = 96
    hyde_enabled: bool = False
    hyde_timeout_ms: int = 2000


@dataclass
class FitbitIntegrationConfig:
    enabled: bool = False


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
class WiringConfig:
    context: str = "default"
    memory: str = "default"
    memory_engine: str = "default"
    toolsets: list[str] = field(
        default_factory=lambda: [
            "meta_common",
            "fitbit",
            "spawn",
            "schedule",
            "mcp",
        ]
    )


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    memory_window: int = 24
    base_url: str | None = None
    extra_body: dict = field(default_factory=dict)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    memory_optimizer_enabled: bool = True
    memory_optimizer_interval_seconds: int = 3600
    light_model: str = ""
    light_api_key: str = ""
    light_base_url: str = ""
    agent_model: str = ""
    agent_api_key: str = ""
    agent_base_url: str = ""
    memory_v2: MemoryV2Config = field(default_factory=MemoryV2Config)
    fitbit: FitbitIntegrationConfig = field(default_factory=FitbitIntegrationConfig)
    tool_search_enabled: bool = False
    spawn_enabled: bool = True
    peer_agents: list[PeerAgentConfig] = field(default_factory=list)
    wiring: WiringConfig = field(default_factory=WiringConfig)

    @classmethod
    def load(cls, path: str | Path = "config.toml") -> Config:
        from importlib import import_module

        return import_module("agent.config").load_config(path)


__all__ = [
    "ChannelsConfig",
    "Config",
    "FitbitIntegrationConfig",
    "MemoryV2Config",
    "PeerAgentConfig",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "WiringConfig",
]
