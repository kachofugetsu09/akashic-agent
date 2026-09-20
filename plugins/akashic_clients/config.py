"""Akashic Web/Mobile client plugin configuration."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WebClientConfig(BaseModel):
    """Web Chat switch and host-owned socket projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    socket_path: str = ""

    @field_validator("socket_path")
    @classmethod
    def validate_socket_path(cls, value: str) -> str:
        value = value.strip()
        if "\x00" in value:
            raise ValueError("web.socket_path 不能包含 NUL")
        return value


class MobileKeyEncryptionConfig(BaseModel):
    """Mobile key protection references; the plugin never moves key files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = "secret_service"
    master_key_namespace: str = "akasic/mobile-realtime"
    master_key_file: Path = Path("data/mobile/master-keys.json")
    keyset_manifest: Path = Path("data/mobile/keys/current.json")

    @model_validator(mode="after")
    def validate_provider(self) -> MobileKeyEncryptionConfig:
        if self.provider not in {"secret_service", "file"}:
            raise ValueError("mobile_realtime.key_encryption.provider 只支持 secret_service 或 file")
        if self.provider == "secret_service" and not self.master_key_namespace.strip():
            raise ValueError("mobile_realtime.key_encryption.master_key_namespace 不能为空")
        if self.keyset_manifest.name != "current.json":
            raise ValueError("mobile_realtime.key_encryption.keyset_manifest 必须指向 current.json")
        return self


class MobileRealtimeConfig(BaseModel):
    """Mobile WSS state references and transport limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=6323, ge=1, le=65535)
    database: Path = Path("data/mobile_realtime.db")
    lan_hostname: str = "akashic.local"
    public_url: str = ""
    max_attachment_mb: int = Field(default=50, gt=0)
    inbox_retention_days: int = Field(default=7, gt=0)
    key_encryption: MobileKeyEncryptionConfig = Field(default_factory=MobileKeyEncryptionConfig)

    @property
    def inbox_retention(self) -> timedelta:
        return timedelta(days=self.inbox_retention_days)

    @model_validator(mode="after")
    def validate_transport(self) -> MobileRealtimeConfig:
        if not self.host.strip():
            raise ValueError("mobile_realtime.host 不能为空")
        if not self.lan_hostname.strip() or any(token in self.lan_hostname for token in ("/", "\\\\", " ")):
            raise ValueError("mobile_realtime.lan_hostname 格式无效")
        if self.public_url:
            public = urlsplit(self.public_url)
            if (
                public.scheme != "wss"
                or not public.netloc
                or public.path != "/ws"
                or public.query
                or public.fragment
                or public.username
                or public.password
            ):
                raise ValueError("mobile_realtime.public_url 必须是无凭据的 wss://.../ws")
        return self


class AkashicClientsConfig(BaseModel):
    """The only config model consumed by the ordinary akashic_clients plugin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    web: WebClientConfig = Field(default_factory=WebClientConfig)
    mobile_realtime: MobileRealtimeConfig = Field(default_factory=MobileRealtimeConfig)

    @model_validator(mode="after")
    def require_enabled_client(self) -> AkashicClientsConfig:
        if self.enabled and not self.web.enabled and not self.mobile_realtime.enabled:
            raise ValueError("akashic_clients 启用时至少需要启用 Web 或 Mobile")
        return self


Config = AkashicClientsConfig

__all__ = [
    "AkashicClientsConfig",
    "Config",
    "MobileKeyEncryptionConfig",
    "MobileRealtimeConfig",
    "WebClientConfig",
]
