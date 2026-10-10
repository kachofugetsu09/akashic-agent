"""从 TOML 读取 Core 自有的中性配置。"""

from __future__ import annotations

import tomllib
from pathlib import Path

from agent.config_models import Config


def load_config(
    path: str | Path = "config.toml",
    *,
    workspace: str | Path,
) -> Config:
    workspace_path = Path(workspace)
    config_path = Path(path)
    data = _load_config_data(config_path)
    _validate_core_config(data)

    return Config(
        config_path=config_path.expanduser().resolve(),
        workspace_path=workspace_path.expanduser().resolve(),
    )


def _validate_core_config(data: dict) -> None:
    """Validate only the neutral root config owned by Core.

    Plugin-owned settings live under each installed plugin's data directory.
    Rejecting unknown tables here is what prevents an old client table from
    being silently accepted when its migration bundle is absent.
    """

    _reject_unknown_keys(data, {"runtime"}, field="配置")
    runtime = _as_dict(data.get("runtime"), field="runtime")
    _reject_unknown_keys(runtime, {"workspace"}, field="runtime")
    if "workspace" in runtime and not isinstance(runtime["workspace"], str):
        raise ValueError("runtime.workspace 必须是字符串")


def _reject_unknown_keys(data: dict, allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        joined = ", ".join(f"{field}.{name}" for name in unknown)
        raise ValueError(f"Core 配置不支持字段: {joined}")


def _as_dict(value: object, *, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 TOML table")
    return value


def _load_config_data(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"主配置仅支持 TOML: {path.suffix}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "Config",
    "load_config",
]
