from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppServerConfig:
    enabled: bool = True
    listen: str = ""
    max_connections: int = 32
    ingress_queue_size: int = 128
    outbound_queue_size: int = 512
    max_message_bytes: int = 2 * 1024 * 1024


@dataclass
class Config:
    app_server: AppServerConfig = field(default_factory=AppServerConfig)
    disabled_builtin_plugins: frozenset[str] = frozenset()
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
    "Config",
]
