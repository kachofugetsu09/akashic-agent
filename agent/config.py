"""
配置加载模块
从 config.toml 读取配置，支持 ${ENV_VAR} 格式的环境变量插值。
"""

from __future__ import annotations

import os
import re
import tomllib
import zlib
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from agent.config_models import (
    AppServerConfig,
    ChannelsConfig,
    Config,
    MobileKeyEncryptionConfig,
    MobileRealtimeConfig,
    QQChannelConfig,
    QQGroupConfig,
    TelegramChannelConfig,
    WebChatConfig,
    WiringConfig,
)

_DEFAULT_TOOLSETS = ("meta_common",)

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
    _reject_removed_proactive_configuration(data)
    _reject_removed_peer_configuration(data)
    _reject_retired_model_configuration(data)
    agent_cfg = _as_dict(data.get("agent"), field="agent")
    agent_context = _as_dict(agent_cfg.get("context"), field="agent.context")
    _reject_removed_context_configuration(data, agent_context)
    agent_tools = _as_dict(agent_cfg.get("tools"), field="agent.tools")
    agent_plugins = _as_dict(agent_cfg.get("plugins"), field="agent.plugins")
    if "spawn_enabled" in agent_tools or "spawn_enabled" in data:
        raise ValueError("spawn_enabled 已移除；请使用 agent.plugins.disabled_builtin")
    agent_maintenance = _as_dict(
        agent_cfg.get("maintenance"), field="agent.maintenance"
    )
    channels = _load_channels_config(data, workspace_path)
    app_server = _load_app_server_config(data)
    mobile_realtime = _load_mobile_realtime_config(data)
    if mobile_realtime.enabled and not channels.chat.enabled:
        raise ValueError("mobile_realtime 启用时必须启用 channels.chat 配对入口")
    wiring = _load_wiring_config(data)
    retired_optimizer_keys = {
        "memory_optimizer_enabled",
        "memory_optimizer_interval_seconds",
    }
    found_retired = sorted(
        retired_optimizer_keys.intersection(data)
        | retired_optimizer_keys.intersection(agent_maintenance)
    )
    if found_retired:
        raise ValueError(
            "PENDING/MemoryOptimizer 已移除，请删除配置: " + ", ".join(found_retired)
        )

    return Config(
        system_prompt=str(
            agent_cfg.get("system_prompt")
            or data.get("system_prompt", "You are a helpful assistant.")
        ),
        max_iterations=int(
            agent_cfg.get("max_iterations", data.get("max_iterations", 10))
        ),
        channels=channels,
        app_server=app_server,
        mobile_realtime=mobile_realtime,
        tool_search_enabled=_as_bool(
            agent_tools.get("search_enabled", data.get("tool_search_enabled", False)),
            field="agent.tools.search_enabled",
        ),
        disabled_builtin_plugins=_disabled_builtin_plugins(agent_plugins),
        dev_mode=_as_bool(
            agent_cfg.get(
                "dev_mode",
                agent_cfg.get(
                    "dev_model",
                    data.get("dev_mode", data.get("dev_model", False)),
                ),
            ),
            field="agent.dev_mode",
        ),
        wiring=wiring,
        config_path=config_path.expanduser().resolve(),
        workspace_path=workspace_path.expanduser().resolve(),
    )


def _load_channels_config(data: dict, workspace: Path) -> ChannelsConfig:
    channels_data = _as_dict(data.get("channels"), field="channels")

    telegram = None
    tg = _as_dict(channels_data.get("telegram"), field="channels.telegram")
    if tg:
        token = _normalize_optional_config_text(
            _resolve(str(tg.get("token", "")), workspace)
        )
        if (
            _as_bool(tg.get("enabled", True), field="channels.telegram.enabled")
            and token
        ):
            telegram = TelegramChannelConfig(
                token=token,
                allow_from=[
                    str(u) for u in tg.get("allow_from", tg.get("allowFrom", []))
                ],
                channel_name=str(tg.get("channel_name", "telegram")),
            )

    qq = None
    qq_data = _as_dict(channels_data.get("qq"), field="channels.qq")
    if qq_data:
        bot_uin = _normalize_optional_config_text(str(qq_data.get("bot_uin", "")))
        if (
            _as_bool(qq_data.get("enabled", True), field="channels.qq.enabled")
            and bot_uin
        ):
            groups = [
                QQGroupConfig(
                    group_id=str(g["group_id"] if "group_id" in g else g["groupId"]),
                    allow_from=[
                        str(u) for u in g.get("allow_from", g.get("allowFrom", []))
                    ],
                    require_at=_as_bool(
                        g.get("require_at", g.get("requireAt", True)),
                        field="channels.qq.groups[].require_at",
                    ),
                )
                for g in qq_data.get("groups", [])
            ]
            qq = QQChannelConfig(
                bot_uin=bot_uin,
                allow_from=[
                    str(u)
                    for u in qq_data.get("allow_from", qq_data.get("allowFrom", []))
                ],
                groups=groups,
                websocket_open_timeout_seconds=float(
                    qq_data.get("websocket_open_timeout_seconds", 5.0)
                ),
            )

    if "socket" in channels_data or "cli" in channels_data:
        raise ValueError(
            '旧 channels.socket/channels.cli 配置已删除；请改用 [app_server] listen = ""'
        )
    chat_data = _as_dict(channels_data.get("chat"), field="channels.chat")
    if "channel_name" in chat_data:
        raise ValueError("channels.chat.channel_name 已移除；内建渠道固定为 akashic")
    chat = WebChatConfig(
        enabled=_as_bool(chat_data.get("enabled", True), field="channels.chat.enabled"),
    )
    channels = ChannelsConfig(
        telegram=telegram,
        qq=qq,
        chat=chat,
    )
    return channels


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


def _load_mobile_realtime_config(data: dict) -> MobileRealtimeConfig:
    """在配置边界建立只允许 WSS 和加密 keyset 的移动网关配置。"""

    # 1. 解析主网关与密钥保护配置
    raw = _as_dict(data.get("mobile_realtime"), field="mobile_realtime")
    key_raw = _as_dict(
        raw.get("key_encryption"),
        field="mobile_realtime.key_encryption",
    )
    provider = str(key_raw.get("provider", "secret_service") or "")
    namespace = str(
        key_raw.get("master_key_namespace", "akasic/mobile-realtime") or ""
    ).strip()
    master_key_file = _relative_data_path(
        key_raw.get("master_key_file", "data/mobile/master-keys.json"),
        field="mobile_realtime.key_encryption.master_key_file",
    )
    keyset_manifest = _relative_data_path(
        key_raw.get("keyset_manifest", "data/mobile/keys/current.json"),
        field="mobile_realtime.key_encryption.keyset_manifest",
    )
    config = MobileRealtimeConfig(
        enabled=_as_bool(raw.get("enabled", False), field="mobile_realtime.enabled"),
        host=str(raw.get("host", "0.0.0.0") or "").strip(),
        port=int(raw.get("port", 6323)),
        database=_relative_data_path(
            raw.get("database", "data/mobile_realtime.db"),
            field="mobile_realtime.database",
        ),
        lan_hostname=str(raw.get("lan_hostname", "akashic.local") or "").strip(),
        public_url=str(raw.get("public_url", "") or "").strip(),
        max_attachment_mb=int(raw.get("max_attachment_mb", 50)),
        inbox_retention_days=int(raw.get("inbox_retention_days", 7)),
        key_encryption=MobileKeyEncryptionConfig(
            provider=provider,
            master_key_namespace=namespace,
            master_key_file=master_key_file,
            keyset_manifest=keyset_manifest,
        ),
    )

    # 2. 拒绝会弱化认证、TLS 或资源边界的配置
    if not config.host:
        raise ValueError("mobile_realtime.host 不能为空")
    if not 1 <= config.port <= 65535:
        raise ValueError("mobile_realtime.port 必须在 1..65535")
    if not config.lan_hostname or any(
        token in config.lan_hostname for token in ("/", "\\", " ")
    ):
        raise ValueError("mobile_realtime.lan_hostname 格式无效")
    if config.max_attachment_mb <= 0:
        raise ValueError("mobile_realtime.max_attachment_mb 必须大于 0")
    if config.inbox_retention_days <= 0:
        raise ValueError("mobile_realtime.inbox_retention_days 必须大于 0")
    if config.key_encryption.provider not in {"secret_service", "file"}:
        raise ValueError(
            "mobile_realtime.key_encryption.provider 只支持 secret_service 或 file"
        )
    if (
        config.key_encryption.provider == "secret_service"
        and not config.key_encryption.master_key_namespace
    ):
        raise ValueError("mobile_realtime.key_encryption.master_key_namespace 不能为空")
    if config.key_encryption.keyset_manifest.name != "current.json":
        raise ValueError(
            "mobile_realtime.key_encryption.keyset_manifest 必须指向 current.json"
        )
    if config.public_url:
        public = urlsplit(config.public_url)
        if (
            public.scheme != "wss"
            or not public.netloc
            or public.path != "/ws"
            or public.query
            or public.fragment
            or public.username
            or public.password
        ):
            raise ValueError("mobile_realtime.public_url 必须是无凭据的 wss://.../ws")
    return config


def _relative_data_path(value: object, *, field: str) -> Path:
    text = str(value or "").strip()
    path = Path(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} 必须是 workspace 内的安全相对路径")
    return path


def _reject_removed_peer_configuration(data: dict) -> None:
    """Reject removed Peer configuration before loading any runtime settings."""

    # 1. Check both legacy locations at the configuration boundary.
    if "peer_agents" in data:
        raise ValueError(
            "unsupported capability: peer_agents; Peer capability has been removed"
        )
    integrations = data.get("integrations")
    if isinstance(integrations, dict) and "peer_agents" in integrations:
        raise ValueError(
            "unsupported capability: integrations.peer_agents; Peer capability has been removed"
        )


def _reject_removed_context_configuration(
    data: dict,
    agent_context: dict,
) -> None:
    """Fail loudly when a pre-ledger context key bypasses migration."""

    # 1. Legacy message-count and runtime-percent keys are no longer accepted.
    raw_compaction = agent_context.get("compaction")
    if (
        "memory_window" in data
        or "memory_window" in agent_context
        or (isinstance(raw_compaction, dict) and "memory_window" in raw_compaction)
    ):
        raise ValueError(
            "removed configuration: memory_window; run the session compaction migration"
        )
    if isinstance(raw_compaction, dict) and "trigger_percent" in raw_compaction:
        raise ValueError(
            "removed configuration: agent.context.compaction.trigger_percent; "
            "run the session compaction migration"
        )
    if raw_compaction is not None:
        raise ValueError(
            "agent.context.compaction 已移除；请将 keep_recent_tokens 写入 "
            "plugin-data/compaction-builtin/config.local.toml"
        )


def _load_wiring_config(data: dict) -> WiringConfig:
    """加载运行时装配配置，并拒绝会改变工具集语义的错误结构。"""

    # 1. 选择新版 agent.wiring；空表继续兼容旧版顶层 wiring。
    agent_cfg = _as_dict(data.get("agent"), field="agent")
    agent_wiring = agent_cfg.get("wiring")
    if agent_wiring is not None and not isinstance(agent_wiring, dict):
        raise ValueError("agent.wiring 必须是 TOML table")
    raw = agent_wiring or data.get("wiring", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("wiring 必须是 TOML table")
    retired = sorted(set(raw).intersection({"memory", "memory_engine"}))
    if retired:
        raise ValueError(
            f"removed configuration: agent.wiring.{retired[0]}; "
            "Markdown memory is an ordinary plugin"
        )

    # 2. 缺失时使用默认工具集；显式数组中的名称必须非空。
    raw_toolsets = raw.get("toolsets")
    if raw_toolsets is None:
        toolsets = list(_DEFAULT_TOOLSETS)
    elif not isinstance(raw_toolsets, list) or any(
        not isinstance(name, str) or not name.strip() for name in raw_toolsets
    ):
        raise ValueError("agent.wiring.toolsets 必须是字符串数组")
    else:
        toolsets = cast(list[str], raw_toolsets)
    return WiringConfig(
        context=str(raw.get("context", "default") or "default"),
        toolsets=list(toolsets),
    )


def _as_dict(value: object, *, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 TOML table")
    return value


def _resolve(value: str, workspace: Path) -> str:
    resolved = re.sub(
        r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value
    )
    # 若仍是未展开的占位符，尝试从 workspace/memory/<VAR_NAME> 文件读取
    m = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if m:
        key_file = workspace / "memory" / m.group(1)
        if key_file.exists():
            resolved = key_file.read_text(encoding="utf-8").strip()
    return resolved


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


def _normalize_optional_config_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\$\{(\w+)\}", text):
        return ""
    return text


def _load_config_data(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"主配置仅支持 TOML: {path.suffix}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _reject_removed_proactive_configuration(data: dict) -> None:
    """Reject the retired proactive table before any workspace-backed store opens."""
    if "proactive" in data:
        raise ValueError(
            "[proactive] 已移除；请删除旧配置并使用 Content、Wake、Drift 与 Timer 插件"
        )


def _reject_retired_model_configuration(data: dict) -> None:
    """Reject model and memory facts after their plugin-owner migrations."""

    retired = [name for name in ("llm", "memory") if name in data]
    if retired:
        names = ", ".join(f"[{name}]" for name in retired)
        raise ValueError(f"{names} 已迁移到普通模型插件；请先运行 workspace migration")


__all__ = [
    "ChannelsConfig",
    "Config",
    "DEFAULT_SOCKET",
    "resolve_app_server_endpoint",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "load_config",
]
