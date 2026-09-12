from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from .akasha.domain.model import MemoryConfig

@dataclass(frozen=True)
class AkashaConfig:
    db_path: str = "memory/akasha.db"
    index_path: str = "memory/akasha-v2-index.db"
    inject_max_chars: int = 12000
    context_recall_limit: int = 40
    restart: float = 0.25
    tolerance: float = 1e-7
    learning_rate: float = 0.5
    activation_power: float = 2.0
    recurrent_budget: float = 1.0
    reverse_temporal_ratio: float = 0.25
    forgetting_enabled: bool = True
    def validate(self) -> None:
        if not self.db_path or not self.index_path: raise ValueError("Akasha paths empty")
        if self.inject_max_chars <= 0 or not 1 <= self.context_recall_limit <= 40: raise ValueError("invalid Akasha config")
        self.memory_config()
    def memory_config(self) -> MemoryConfig:
        value = MemoryConfig(self.restart,self.tolerance,self.learning_rate,self.activation_power,self.recurrent_budget,self.reverse_temporal_ratio,self.forgetting_enabled)
        value.validate(); return value

def load_akasha_config(path: Path) -> AkashaConfig:
    if not path.exists(): return AkashaConfig()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    allowed = set(AkashaConfig.__dataclass_fields__)
    unknown = sorted(set(raw)-allowed)
    if unknown: raise ValueError(f"unknown historical Akasha config fields: {unknown}")
    value = AkashaConfig(**raw); value.validate(); return value

def resolve_memory_path(memory_root: Path, configured: str) -> Path:
    if memory_root.is_symlink(): raise ValueError(f"Akasha memory root symlink: {memory_root}")
    raw = PurePosixPath(configured); parts = raw.parts
    if not parts or raw.is_absolute(): raise ValueError(f"invalid Akasha sidecar path: {configured}")
    if parts[0] == memory_root.name:
        if len(parts) != 2: raise ValueError(f"invalid Akasha sidecar path: {configured}")
        parts = parts[1:]
    elif len(parts) != 1: raise ValueError(f"invalid Akasha sidecar path: {configured}")
    if PurePosixPath(*parts).suffix.lower() == ".md": raise ValueError("Akasha sidecar cannot target markdown")
    root = memory_root.resolve(strict=False); path = root.joinpath(*parts).resolve(strict=False)
    if not path.is_relative_to(root): raise ValueError("Akasha sidecar escaped memory root")
    return path
