from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import cast

from agent.plugins.manifest import (
    builtin_plugin_data_dir,
    ensure_workspace_plugin_data_dir,
)
from infra.persistence.json_store import atomic_write_text


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
    workspace: Path | None = None,
    plugin_dir: Path | None = None,
) -> DefaultMemoryConfig:
    root = _config_root(workspace=workspace, plugin_dir=plugin_dir)
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


def ensure_default_memory_config_file(
    *,
    workspace: Path | None = None,
    plugin_dir: Path | None = None,
) -> Path:
    """迁移或创建 default memory 的用户配置。"""

    # 1. 已有用户配置直接复用
    root = _config_root(workspace=workspace, plugin_dir=plugin_dir)
    if plugin_dir is None:
        ensure_workspace_plugin_data_dir(root, cast(Path, workspace))
    path = root / "config.local.toml"
    if path.exists():
        return path

    # 2. 首次迁移保留旧目录配置，否则写入默认配置
    legacy_path = Path(__file__).resolve().parent / "config.local.toml"
    content = (
        legacy_path.read_text(encoding="utf-8")
        if legacy_path.exists()
        else render_default_memory_config()
    )
    atomic_write_text(path, content, domain="default_memory.config")
    return path


def _config_root(*, workspace: Path | None, plugin_dir: Path | None) -> Path:
    if plugin_dir is not None:
        return plugin_dir
    if workspace is None:
        raise RuntimeError("default_memory 配置缺少 workspace")
    return builtin_plugin_data_dir("default_memory", workspace)


def resolve_memory_db_path(
    *,
    workspace: Path,
    default_config: DefaultMemoryConfig,
) -> Path:
    root = workspace.resolve(strict=False)
    configured = default_config.db_path or "memory/memory2.db"
    path = (root / configured).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(f"default_memory.db_path 必须位于 workspace 内: {configured}")
    return path


def _build_config(payload: dict[str, object]) -> DefaultMemoryConfig:
    retrieval = _section(payload, "retrieval")
    thresholds = _section(retrieval, "retrieval.thresholds")
    inject = _section(retrieval, "retrieval.inject")
    return DefaultMemoryConfig(
        db_path=_string_value(payload, "db_path", ""),
        retrieval=RetrievalConfig(
            top_k_history=_int_value(retrieval, "retrieval.top_k_history", 8),
            score_threshold=_float_value(
                retrieval, "retrieval.score_threshold", 0.45
            ),
            relative_delta=_float_value(retrieval, "retrieval.relative_delta", 0.2),
            procedure_guard_enabled=_bool_value(
                retrieval, "retrieval.procedure_guard_enabled", True
            ),
            thresholds=RetrievalThresholdsConfig(
                procedure=_float_value(
                    thresholds, "retrieval.thresholds.procedure", 0.66
                ),
                preference=_float_value(
                    thresholds, "retrieval.thresholds.preference", 0.5
                ),
                event=_float_value(thresholds, "retrieval.thresholds.event", 0.5),
                profile=_float_value(
                    thresholds, "retrieval.thresholds.profile", 0.5
                ),
            ),
            inject=RetrievalInjectConfig(
                max_chars=_int_value(inject, "retrieval.inject.max_chars", 6000),
                forced=_int_value(inject, "retrieval.inject.forced", 3),
                procedure_preference=_int_value(
                    inject,
                    "retrieval.inject.procedure_preference",
                    4,
                ),
                event_profile=_int_value(inject, "retrieval.inject.event_profile", 4),
                line_max=_int_value(inject, "retrieval.inject.line_max", 600),
            ),
        ),
    )


def _read_toml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))


def _field_key(field: str) -> str:
    return field.rpartition(".")[2]


def _section(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    key = _field_key(field)
    if key not in payload:
        return {}
    value = payload[key]
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    raise ValueError(f"默认记忆配置 {field} 必须是 section，实际为 {value!r}")


def _string_value(
    payload: dict[str, object],
    field: str,
    default: str,
) -> str:
    key = _field_key(field)
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, str):
        return value
    raise ValueError(f"默认记忆配置 {field} 必须是字符串，实际为 {value!r}")


def _int_value(
    payload: dict[str, object],
    field: str,
    default: int,
) -> int:
    key = _field_key(field)
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
    raise ValueError(f"默认记忆配置 {field} 必须是整数，实际为 {value!r}")


def _float_value(
    payload: dict[str, object],
    field: str,
    default: float,
) -> float:
    key = _field_key(field)
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool) and isinstance(value, int | float | str):
        try:
            parsed = float(value)
        except ValueError:
            pass
        else:
            if isfinite(parsed):
                return parsed
    raise ValueError(f"默认记忆配置 {field} 必须是有限数字，实际为 {value!r}")


def _bool_value(
    payload: dict[str, object],
    field: str,
    default: bool,
) -> bool:
    key = _field_key(field)
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    raise ValueError(f"默认记忆配置 {field} 必须是布尔值，实际为 {value!r}")
