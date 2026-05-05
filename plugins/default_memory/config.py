from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalThresholdsConfig:
    procedure: float = 0.66
    preference: float = 0.5
    event: float = 0.5
    profile: float = 0.5


@dataclass(frozen=True)
class RetrievalInjectConfig:
    max_chars: int = 6000
    forced: int = 3
    procedure_preference: int = 4
    event_profile: int = 4
    line_max: int = 600


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_history: int = 8
    score_threshold: float = 0.45
    relative_delta: float = 0.2
    procedure_guard_enabled: bool = True
    thresholds: RetrievalThresholdsConfig = field(
        default_factory=RetrievalThresholdsConfig
    )
    inject: RetrievalInjectConfig = field(default_factory=RetrievalInjectConfig)


@dataclass(frozen=True)
class DefaultMemoryConfig:
    db_path: str = ""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def load_default_memory_config(
    *,
    plugin_dir: Path | None = None,
) -> DefaultMemoryConfig:
    root = plugin_dir or Path(__file__).resolve().parent
    payload = _read_toml(root / "config.local.toml")
    return _build_config(payload)


def render_default_memory_config(config: DefaultMemoryConfig | None = None) -> str:
    cfg = config or DefaultMemoryConfig()
    retrieval = cfg.retrieval
    return "\n".join([
        f'db_path = "{cfg.db_path}"',
        "",
        "[retrieval]",
        f"top_k_history = {retrieval.top_k_history}",
        f"score_threshold = {retrieval.score_threshold}",
        f"relative_delta = {retrieval.relative_delta}",
        f"procedure_guard_enabled = {str(retrieval.procedure_guard_enabled).lower()}",
        "",
        "[retrieval.thresholds]",
        f"procedure = {retrieval.thresholds.procedure}",
        f"preference = {retrieval.thresholds.preference}",
        f"event = {retrieval.thresholds.event}",
        f"profile = {retrieval.thresholds.profile}",
        "",
        "[retrieval.inject]",
        f"max_chars = {retrieval.inject.max_chars}",
        f"forced = {retrieval.inject.forced}",
        f"procedure_preference = {retrieval.inject.procedure_preference}",
        f"event_profile = {retrieval.inject.event_profile}",
        f"line_max = {retrieval.inject.line_max}",
        "",
    ])


def ensure_default_memory_config_file(*, plugin_dir: Path | None = None) -> Path:
    root = plugin_dir or Path(__file__).resolve().parent
    path = root / "config.local.toml"
    if not path.exists():
        path.write_text(render_default_memory_config(), encoding="utf-8")
    return path


def resolve_memory_db_path(
    *,
    workspace: Path,
    default_config: DefaultMemoryConfig,
) -> Path:
    if not default_config.db_path:
        return workspace / "memory" / "memory2.db"
    path = Path(default_config.db_path)
    return path if path.is_absolute() else workspace / path


def _build_config(payload: dict[str, Any]) -> DefaultMemoryConfig:
    retrieval = _as_dict(payload.get("retrieval"))
    thresholds = _as_dict(retrieval.get("thresholds"))
    inject = _as_dict(retrieval.get("inject"))
    return DefaultMemoryConfig(
        db_path=str(payload.get("db_path", "")),
        retrieval=RetrievalConfig(
            top_k_history=int(retrieval.get("top_k_history", 8)),
            score_threshold=float(retrieval.get("score_threshold", 0.45)),
            relative_delta=float(retrieval.get("relative_delta", 0.2)),
            procedure_guard_enabled=bool(
                retrieval.get("procedure_guard_enabled", True)
            ),
            thresholds=RetrievalThresholdsConfig(
                procedure=float(thresholds.get("procedure", 0.66)),
                preference=float(thresholds.get("preference", 0.5)),
                event=float(thresholds.get("event", 0.5)),
                profile=float(thresholds.get("profile", 0.5)),
            ),
            inject=RetrievalInjectConfig(
                max_chars=int(inject.get("max_chars", 6000)),
                forced=int(inject.get("forced", 3)),
                procedure_preference=int(inject.get("procedure_preference", 4)),
                event_profile=int(inject.get("event_profile", 4)),
                line_max=int(inject.get("line_max", 600)),
            ),
        ),
    )


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
