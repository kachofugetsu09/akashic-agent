from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
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
    "Config",
]
