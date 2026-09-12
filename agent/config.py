"""
配置加载模块
从 config.toml 读取配置，支持 ${ENV_VAR} 格式的环境变量插值。
"""

from __future__ import annotations

import os
import tomllib
import zlib
from pathlib import Path

from agent.config_models import (
    AppServerConfig,
    Config,
)

# 空值表示由 workspace 派生 app-server 端点，避免多个实例争用全局路径。
DEFAULT_SOCKET = ""


def _normalize_app_server_endpoint(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_SOCKET
    if os.name != "nt":
        return text
    host, sep, port = text.rpartition(":")
    if sep and host:
        try:
            int(port)
            return text
        except ValueError:
            pass
    port_seed = zlib.crc32(text.encode("utf-8")) % 20000
    return f"127.0.0.1:{20000 + port_seed}"


def resolve_app_server_endpoint(value: str, workspace: Path) -> str:
    """解析当前 workspace 独占的 app-server 端点。"""

    # 1. 显式配置保持原样
    if value:
        return value

    # 2. 缺省配置按 workspace 稳定派生
    if os.name != "nt":
        return str(workspace / "akashic.sock")
    port_seed = zlib.crc32(str(workspace).encode("utf-8")) % 20000
    return f"127.0.0.1:{20000 + port_seed}"


def load_config(
    path: str | Path = "config.toml",
    *,
    workspace: str | Path,
) -> Config:
    workspace_path = Path(workspace)
    config_path = Path(path)
    data = _load_config_data(config_path)
    _validate_core_config(data)
    agent_cfg = _as_dict(data.get("agent"), field="agent")
    agent_plugins = _as_dict(agent_cfg.get("plugins"), field="agent.plugins")
    app_server = _load_app_server_config(data)

    return Config(
        app_server=app_server,
        disabled_builtin_plugins=_disabled_builtin_plugins(agent_plugins),
        config_path=config_path.expanduser().resolve(),
        workspace_path=workspace_path.expanduser().resolve(),
    )


def _validate_core_config(data: dict) -> None:
    """Validate only the neutral root config owned by Core.

    Plugin-owned settings live under each installed plugin's data directory.
    Rejecting unknown tables here is what prevents an old client table from
    being silently accepted when its migration bundle is absent.
    """

    _reject_unknown_keys(data, {"runtime", "app_server", "agent"}, field="配置")
    runtime = _as_dict(data.get("runtime"), field="runtime")
    _reject_unknown_keys(runtime, {"workspace"}, field="runtime")
    if "workspace" in runtime and not isinstance(runtime["workspace"], str):
        raise ValueError("runtime.workspace 必须是字符串")
    agent = _as_dict(data.get("agent"), field="agent")
    _reject_unknown_keys(agent, {"plugins"}, field="agent")
    plugins = _as_dict(agent.get("plugins"), field="agent.plugins")
    _reject_unknown_keys(plugins, {"disabled_builtin"}, field="agent.plugins")
    app_server = _as_dict(data.get("app_server"), field="app_server")
    _reject_unknown_keys(
        app_server,
        {
            "enabled",
            "listen",
            "max_connections",
            "ingress_queue_size",
            "outbound_queue_size",
            "max_message_bytes",
        },
        field="app_server",
    )


def _reject_unknown_keys(data: dict, allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        joined = ", ".join(f"{field}.{name}" for name in unknown)
        raise ValueError(f"Core 配置不支持字段: {joined}")


def _load_app_server_config(data: dict) -> AppServerConfig:
    """在配置边界校验本地控制面的资源上限。"""

    raw = _as_dict(data.get("app_server"), field="app_server")
    config = AppServerConfig(
        enabled=_as_bool(raw.get("enabled", True), field="app_server.enabled"),
        listen=_normalize_app_server_endpoint(str(raw.get("listen", ""))),
        max_connections=int(raw.get("max_connections", 32)),
        ingress_queue_size=int(raw.get("ingress_queue_size", 128)),
        outbound_queue_size=int(raw.get("outbound_queue_size", 512)),
        max_message_bytes=int(raw.get("max_message_bytes", 2 * 1024 * 1024)),
    )
    for name in (
        "max_connections",
        "ingress_queue_size",
        "outbound_queue_size",
        "max_message_bytes",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"app_server.{name} 必须大于 0")
    return config


def _as_dict(value: object, *, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 TOML table")
    return value


def _as_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} 必须是布尔值")
    return value


def _disabled_builtin_plugins(
    plugins: dict,
) -> frozenset[str]:
    """Validate the generic builtin plugin activation projection."""

    # 1. 新配置只描述插件身份，不把某个功能写进 bootstrap 控制流。
    raw = plugins.get("disabled_builtin", ())
    if not isinstance(raw, list | tuple):
        raise ValueError("agent.plugins.disabled_builtin 必须是字符串数组")
    disabled = {
        item for item in raw if isinstance(item, str) and item and item.strip() == item
    }
    if len(disabled) != len(raw):
        raise ValueError(
            "agent.plugins.disabled_builtin 必须只包含非空且无首尾空白的字符串"
        )

    return frozenset(disabled)


def _load_config_data(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"主配置仅支持 TOML: {path.suffix}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "Config",
    "DEFAULT_SOCKET",
    "resolve_app_server_endpoint",
    "load_config",
]
