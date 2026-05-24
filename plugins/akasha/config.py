from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class AkashaConfig:
    db_path: str = ""
    dense_top_k: int = 10
    ripple_top_k: int = 10
    activate_limit: int = 8
    inject_max_chars: int = 6000
    assistant_preview_chars: int = 15
    dense_seed_threshold: float = 0.675
    nearby_time_seconds: int = 1800
    nearby_dense_threshold: float = 0.28
    activation_threshold: float = 0.22
    soft_recall_threshold: float = 0.165
    soft_recall_direct_floor: float = 0.45
    cross_boost: float = 36.0


# 读取 Akasha 插件配置文件。
def load_akasha_config(
    *,
    plugin_dir: Path | None = None,
) -> AkashaConfig:
    # 1. 读取插件目录下的本地配置。
    root = plugin_dir or Path(__file__).resolve().parent
    payload = _read_toml(root / "config.local.toml")

    # 2. 把 TOML 字段收敛成强类型配置。
    return AkashaConfig(
        db_path=str(payload.get("db_path") or ""),
        dense_top_k=_int_value(payload.get("dense_top_k"), 10),
        ripple_top_k=_int_value(payload.get("ripple_top_k"), 10),
        activate_limit=_int_value(payload.get("activate_limit"), 8),
        inject_max_chars=_int_value(payload.get("inject_max_chars"), 6000),
        assistant_preview_chars=_int_value(payload.get("assistant_preview_chars"), 15),
        dense_seed_threshold=_float_value(payload.get("dense_seed_threshold"), 0.675),
        nearby_time_seconds=_int_value(payload.get("nearby_time_seconds"), 1800),
        nearby_dense_threshold=_float_value(payload.get("nearby_dense_threshold"), 0.28),
        activation_threshold=_float_value(payload.get("activation_threshold"), 0.22),
        soft_recall_threshold=_float_value(payload.get("soft_recall_threshold"), 0.165),
        soft_recall_direct_floor=_float_value(payload.get("soft_recall_direct_floor"), 0.45),
        cross_boost=_float_value(payload.get("cross_boost"), 36.0),
    )


# 渲染默认 Akasha 配置。
def render_akasha_config(config: AkashaConfig | None = None) -> str:
    # 1. 使用传入配置或默认配置生成本地配置文本。
    cfg = config or AkashaConfig()
    return "\n".join([
        f'db_path = "{cfg.db_path}"',
        f"dense_top_k = {cfg.dense_top_k}",
        f"ripple_top_k = {cfg.ripple_top_k}",
        f"activate_limit = {cfg.activate_limit}",
        f"inject_max_chars = {cfg.inject_max_chars}",
        f"assistant_preview_chars = {cfg.assistant_preview_chars}",
        f"dense_seed_threshold = {cfg.dense_seed_threshold}",
        f"nearby_time_seconds = {cfg.nearby_time_seconds}",
        f"nearby_dense_threshold = {cfg.nearby_dense_threshold}",
        f"activation_threshold = {cfg.activation_threshold}",
        f"soft_recall_threshold = {cfg.soft_recall_threshold}",
        f"soft_recall_direct_floor = {cfg.soft_recall_direct_floor}",
        f"cross_boost = {cfg.cross_boost}",
        "",
    ])


# 确保 Akasha 本地配置文件存在。
def ensure_akasha_config_file(*, plugin_dir: Path | None = None) -> Path:
    # 1. 缺省时只写入默认配置，不覆盖用户已有配置。
    root = plugin_dir or Path(__file__).resolve().parent
    path = root / "config.local.toml"
    if not path.exists():
        _ = path.write_text(render_akasha_config(), encoding="utf-8")
    return path


# 解析 Akasha sidecar 数据库路径。
def resolve_akasha_db_path(
    *,
    workspace: Path,
    akasha_config: AkashaConfig,
) -> Path:
    # 1. 默认落在 workspace/memory/akasha.db。
    if not akasha_config.db_path:
        return workspace / "memory" / "akasha.db"

    # 2. 相对路径以 workspace 为根，绝对路径原样使用。
    path = Path(akasha_config.db_path)
    return path if path.is_absolute() else workspace / path


# 读取 TOML 文件为普通 dict。
def _read_toml(path: Path) -> dict[str, object]:
    # 1. 配置不存在时回到默认值。
    if not path.exists():
        return {}
    return cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))


# 把配置值转换成 int。
def _int_value(value: object, default: int) -> int:
    # 1. 支持 TOML 数字和字符串数字。
    if isinstance(value, int):
        return value
    if isinstance(value, str | float):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


# 把配置值转换成 float。
def _float_value(value: object, default: float) -> float:
    # 1. 支持 TOML 数字和字符串数字。
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
