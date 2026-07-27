"""Load the versioned Akasha V2 plugin configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .domain.model import MemoryConfig


@dataclass(frozen=True)
class AkashaConfig:
    """Configure storage, output budget, and graph-forming dynamics."""

    db_path: str = "memory/akasha.db"
    index_path: str = "memory/akasha-v2-index.db"
    inject_max_chars: int = 12_000
    context_recall_limit: int = 40
    restart: float = 0.25
    tolerance: float = 1e-7
    learning_rate: float = 0.5
    activation_power: float = 2.0
    recurrent_budget: float = 1.0
    reverse_temporal_ratio: float = 0.25
    forgetting_enabled: bool = True

    def validate(self) -> None:
        """Reject invalid adapter and dynamics values at config load."""

        if not self.db_path or not self.index_path:
            raise ValueError("Akasha storage paths cannot be empty")
        if self.inject_max_chars <= 0:
            raise ValueError("inject_max_chars must be positive")
        if not 1 <= self.context_recall_limit <= 40:
            raise ValueError("context_recall_limit must be in [1, 40]")
        self.memory_config()

    def memory_config(self) -> MemoryConfig:
        config = MemoryConfig(
            restart=self.restart,
            tolerance=self.tolerance,
            learning_rate=self.learning_rate,
            activation_power=self.activation_power,
            recurrent_budget=self.recurrent_budget,
            reverse_temporal_ratio=self.reverse_temporal_ratio,
            forgetting_enabled=self.forgetting_enabled,
        )
        config.validate()
        return config


def load_akasha_config(path: Path) -> AkashaConfig:
    """Read one strict TOML config or return documented defaults."""

    if not path.exists():
        return AkashaConfig()
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    allowed = set(AkashaConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown Akasha V2 config fields: {unknown}")
    config = AkashaConfig(**payload)
    config.validate()
    return config


def render_akasha_config(config: AkashaConfig = AkashaConfig()) -> str:
    """Render a complete deterministic local configuration."""

    return "\n".join(
        [
            f'db_path = "{config.db_path}"',
            f'index_path = "{config.index_path}"',
            f"inject_max_chars = {config.inject_max_chars}",
            f"context_recall_limit = {config.context_recall_limit}",
            f"restart = {config.restart}",
            f"tolerance = {config.tolerance}",
            f"learning_rate = {config.learning_rate}",
            f"activation_power = {config.activation_power}",
            f"recurrent_budget = {config.recurrent_budget}",
            (
                "reverse_temporal_ratio = "
                f"{config.reverse_temporal_ratio}"
            ),
            (
                "forgetting_enabled = "
                f"{str(config.forgetting_enabled).lower()}"
            ),
            "",
        ]
    )


def resolve_workspace_path(
    workspace: Path,
    configured: str,
) -> Path:
    """Resolve one configured path and keep it inside the workspace."""

    root = workspace.resolve(strict=False)
    path = (root / configured).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(
            f"Akasha V2 storage path must stay in workspace: {configured}"
        )
    return path
