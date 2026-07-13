from __future__ import annotations

import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from agent.plugins.manifest import builtin_plugin_data_dir
from infra.persistence.json_store import atomic_write_text


@dataclass(frozen=True)
class AkashaConfig:
    db_path: str = ""
    inject_max_chars: int = 6000
    assistant_preview_chars: int = 0
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
    root = plugin_dir or builtin_plugin_data_dir("akasha")
    payload = _read_toml(root / "config.local.toml")

    # 2. 把 TOML 字段收敛成强类型配置。
    return AkashaConfig(
        db_path=_string_value(payload, "db_path", ""),
        inject_max_chars=_int_value(payload, "inject_max_chars", 6000),
        assistant_preview_chars=_int_value(payload, "assistant_preview_chars", 0),
        dense_seed_threshold=_float_value(payload, "dense_seed_threshold", 0.675),
        nearby_time_seconds=_int_value(payload, "nearby_time_seconds", 1800),
        nearby_dense_threshold=_float_value(payload, "nearby_dense_threshold", 0.28),
        activation_threshold=_float_value(payload, "activation_threshold", 0.22),
        soft_recall_threshold=_float_value(payload, "soft_recall_threshold", 0.165),
        soft_recall_direct_floor=_float_value(payload, "soft_recall_direct_floor", 0.45),
        cross_boost=_float_value(payload, "cross_boost", 36.0),
    )


# 渲染默认 Akasha 配置。
def render_akasha_config(config: AkashaConfig | None = None) -> str:
    # 1. 使用传入配置或默认配置生成本地配置文本。
    cfg = config or AkashaConfig()
    return "\n".join([
        f'db_path = "{cfg.db_path}"',
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


def ensure_akasha_config_file(*, plugin_dir: Path | None = None) -> Path:
    """迁移或创建 Akasha 的用户配置。"""

    # 1. 已有用户配置直接复用
    root = plugin_dir or builtin_plugin_data_dir("akasha")
    path = root / "config.local.toml"
    if path.exists():
        return path

    # 2. 首次迁移保留旧目录配置，否则写入默认配置
    root.mkdir(parents=True, exist_ok=True)
    legacy_path = Path(__file__).resolve().parent / "config.local.toml"
    content = (
        legacy_path.read_text(encoding="utf-8")
        if legacy_path.exists()
        else render_akasha_config()
    )
    atomic_write_text(path, content, domain="akasha.config")
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


# 读取字符串配置，缺失字段使用默认值。
def _string_value(payload: dict[str, object], key: str, default: str) -> str:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, str):
        return value
    raise ValueError(f"Akasha 配置 {key} 必须是字符串，实际为 {value!r}")


# 读取整数配置，缺失字段使用默认值。
def _int_value(payload: dict[str, object], key: str, default: int) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        raise ValueError(f"Akasha 配置 {key} 必须是整数，实际为 {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if isfinite(value) and value.is_integer():
            return int(value)
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError(f"Akasha 配置 {key} 必须是整数，实际为 {value!r}")


# 读取浮点配置，缺失字段使用默认值。
def _float_value(payload: dict[str, object], key: str, default: float) -> float:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        raise ValueError(f"Akasha 配置 {key} 必须是数字，实际为 {value!r}")
    if isinstance(value, int | float | str):
        try:
            parsed = float(value)
        except ValueError:
            pass
        else:
            if isfinite(parsed):
                return parsed
    raise ValueError(f"Akasha 配置 {key} 必须是有限数字，实际为 {value!r}")
