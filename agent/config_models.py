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
class WiringConfig:
    context: str = "default"
    toolsets: list[str] = field(
        default_factory=lambda: [
            "meta_common",
        ]
    )


@dataclass
class Config:
    system_prompt: str
    max_iterations: int = 10
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    app_server: AppServerConfig = field(default_factory=AppServerConfig)
    mobile_realtime: MobileRealtimeConfig = field(default_factory=MobileRealtimeConfig)
    tool_search_enabled: bool = False
    disabled_builtin_plugins: frozenset[str] = frozenset()
    dev_mode: bool = False
    wiring: WiringConfig = field(default_factory=WiringConfig)
    config_path: Path = Path("config.toml")
    workspace_path: Path = Path(".")

    @classmethod
    def load(
        cls,
        path: str | Path = "config.toml",
        *,
        workspace: str | Path,
    ) -> Config:
        from agent.config import load_config

        return load_config(path, workspace=workspace)


__all__ = [
    "AppServerConfig",
    "ChannelsConfig",
    "Config",
    "MobileKeyEncryptionConfig",
    "MobileRealtimeConfig",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "WebChatConfig",
    "WiringConfig",
]
