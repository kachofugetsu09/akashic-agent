from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

@dataclass
class TelegramChannelConfig:
    token: str
    allow_from: list[str] = field(default_factory=list)
    channel_name: str = "telegram"


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
    websocket_open_timeout_seconds: float = 5.0


@dataclass
class WebChatConfig:
    enabled: bool = True
    channel_name: str = "web"


@dataclass
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
    qq: QQChannelConfig | None = None
    chat: WebChatConfig = field(default_factory=WebChatConfig)


@dataclass
class AppServerConfig:
    enabled: bool = True
    listen: str = ""
    max_connections: int = 32
    ingress_queue_size: int = 128
    outbound_queue_size: int = 512
    max_message_bytes: int = 2 * 1024 * 1024


@dataclass(frozen=True)
class MobileKeyEncryptionConfig:
    provider: str = "secret_service"
    master_key_namespace: str = "akasic/mobile-realtime"
    master_key_file: Path = Path("data/mobile/master-keys.json")
    keyset_manifest: Path = Path("data/mobile/keys/current.json")


@dataclass(frozen=True)
class MobileRealtimeConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 6323
    database: Path = Path("data/mobile_realtime.db")
    lan_hostname: str = "akashic.local"
    public_url: str = ""
    max_attachment_mb: int = 50
    inbox_retention_days: int = 7
    key_encryption: MobileKeyEncryptionConfig = field(
        default_factory=MobileKeyEncryptionConfig
    )

    @property
    def inbox_retention(self) -> timedelta:
        return timedelta(days=self.inbox_retention_days)


@dataclass
class MemoryEmbeddingConfig:
    model_ref: str = ""
    model: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = ""
    output_dimensionality: int | None = None
    auth: str = ""


@dataclass
class MemoryConfig:
    enabled: bool = False
    engine: str = ""
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)


@dataclass(frozen=True)
class ContextCompactionConfig:
    """Session compaction policy independent from any model runtime."""

    keep_recent_tokens: int = 20_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.keep_recent_tokens, int)
            or isinstance(self.keep_recent_tokens, bool)
            or self.keep_recent_tokens <= 0
        ):
            raise ValueError(
                "agent.context.compaction.keep_recent_tokens 必须是正整数"
            )


@dataclass
class WiringConfig:
    context: str = "default"
    memory: str = "default"
    toolsets: list[str] = field(
        default_factory=lambda: [
            "meta_common",
        ]
    )


@dataclass(frozen=True)
class ModelRuntimeConfig:
    runtime_id: str
    provider: str
    model: str
    source_id: str = ""
    source_name: str = ""
    catalog_provider_id: str = ""
    auth: str = ""
    api_key: str = ""
    base_url: str = ""
    reasoning_effort: str = ""
    supported_reasoning_efforts: tuple[str, ...] = ()
    context_window: int = 0
    # 0 表示不向 provider 发送输出上限，由模型服务自身边界负责。
    max_output_tokens: int = 0
    input_modalities: tuple[str, ...] = ("text",)
    capability_source: str = "unknown"
    context_window_source: str = "unknown"
    max_output_tokens_source: str = "unknown"
    input_modalities_source: str = "unknown"
    use_responses_lite: bool = False
    supports_parallel_tool_calls: bool = True
    reasoning_summary: str = "none"

    def __post_init__(self) -> None:
        from agent.model_runtime.provider_profiles import validate_profile_runtime

        if not self.provider or not self.model:
            raise ValueError(f"runtime {self.runtime_id} 必须配置 provider 和 model")
        if self.provider == "codex" and not self.auth:
            raise ValueError(f"Codex runtime {self.runtime_id} 必须配置 auth")
        if self.context_window < 0:
            raise ValueError(f"runtime {self.runtime_id} 的 context_window 不能小于 0")
        if self.max_output_tokens < 0:
            raise ValueError(
                f"runtime {self.runtime_id} 的 max_output_tokens 不能小于 0"
            )
        if self.context_window > 0 and self.max_output_tokens >= self.context_window:
            raise ValueError(
                f"runtime {self.runtime_id} 的 max_output_tokens 必须小于 context_window"
            )
        if "text" not in self.input_modalities:
            raise ValueError(
                f"runtime {self.runtime_id} 的 input_modalities 必须包含 text"
            )
        allowed_sources = {"explicit", "provider_catalog", "litellm", "unknown"}
        for field_name, source in (
            ("context_window_source", self.context_window_source),
            ("max_output_tokens_source", self.max_output_tokens_source),
            ("input_modalities_source", self.input_modalities_source),
        ):
            if source not in allowed_sources:
                raise ValueError(f"runtime {self.runtime_id} 的 {field_name} 无效")
        if self.capability_source not in allowed_sources | {"mixed"}:
            raise ValueError(f"runtime {self.runtime_id} 的 capability_source 无效")
        validate_profile_runtime(
            provider=self.provider,
            model=self.model,
            input_modalities=self.input_modalities,
        )


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 0
    max_iterations: int = 10
    context_compaction: ContextCompactionConfig = field(
        default_factory=ContextCompactionConfig
    )
    base_url: str | None = None
    extra_body: dict[str, object] = field(default_factory=dict)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    app_server: AppServerConfig = field(default_factory=AppServerConfig)
    mobile_realtime: MobileRealtimeConfig = field(default_factory=MobileRealtimeConfig)
    memory_optimizer_enabled: bool = True
    memory_optimizer_interval_seconds: int = 64800
    light_model: str = ""
    light_api_key: str = ""
    light_base_url: str = ""
    agent_model: str = ""
    agent_api_key: str = ""
    agent_base_url: str = ""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    multimodal: bool = True
    vl_model: str = ""
    vl_api_key: str = ""
    vl_base_url: str = ""
    tool_search_enabled: bool = False
    disabled_builtin_plugins: frozenset[str] = frozenset()
    dev_mode: bool = False
    wiring: WiringConfig = field(default_factory=WiringConfig)
    runtime_id: str = "main"
    auth_id: str = ""
    context_window: int = 0
    reasoning_effort: str = ""
    input_modalities: tuple[str, ...] = ("text",)
    use_responses_lite: bool = False
    supports_parallel_tool_calls: bool = True
    reasoning_summary: str = "none"
    model_runtimes: dict[str, ModelRuntimeConfig] = field(default_factory=dict)
    fast_runtime_id: str = ""
    agent_runtime_id: str = ""
    vl_runtime_id: str = ""
    model_registry_revision: int = 0
    config_path: Path = Path("config.toml")
    workspace_path: Path = Path(".")

    @classmethod
    def load(
        cls,
        path: str | Path = "config.toml",
        *,
        workspace: str | Path,
        credential_store: object | None = None,
    ) -> Config:
        from importlib import import_module

        return import_module("agent.config").load_config(
            path,
            workspace=workspace,
            credential_store=credential_store,
        )


__all__ = [
    "AppServerConfig",
    "ChannelsConfig",
    "Config",
    "ContextCompactionConfig",
    "MemoryConfig",
    "MemoryEmbeddingConfig",
    "MobileKeyEncryptionConfig",
    "MobileRealtimeConfig",
    "ModelRuntimeConfig",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "WebChatConfig",
    "WiringConfig",
]
