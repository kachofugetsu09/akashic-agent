"""
配置加载模块
从 config.toml 读取配置，支持 ${ENV_VAR} 格式的环境变量插值。
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
import zlib
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.config_models import (
    ChannelsConfig,
    Config,
    MemoryConfig,
    MemoryEmbeddingConfig,
    PeerAgentConfig,
    QQChannelConfig,
    QQGroupConfig,
    TelegramChannelConfig,
    WebChatConfig,
    WiringConfig,
)
from proactive_v2.config import ProactiveConfig
from proactive_v2.config_loader import ProactiveConfigError, load_proactive_config

_PRESETS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}
_DEFAULT_TOOLSETS = ("meta_common", "spawn", "schedule", "mcp")

# 空值表示由 workspace 派生 IPC 端点，避免多个实例争用全局路径。
DEFAULT_SOCKET = ""


def _normalize_cli_socket_endpoint(value: str | None) -> str:
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


def resolve_cli_socket_endpoint(value: str, workspace: Path) -> str:
    """解析当前 workspace 独占的 IPC 端点。"""

    # 1. 显式配置保持原样
    if value:
        return value

    # 2. 缺省配置按 workspace 稳定派生
    if os.name != "nt":
        return str(workspace / "akashic.sock")
    port_seed = zlib.crc32(str(workspace).encode("utf-8")) % 20000
    return f"127.0.0.1:{20000 + port_seed}"


def _validated_timezone(tz_name: str, *, enabled: bool) -> str:
    """仅当 anyaction_enabled=True 时校验时区合法性，无效则启动时 fail-fast。"""
    if not enabled:
        return tz_name
    try:
        _ = ZoneInfo(tz_name)
        return tz_name
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"proactive.anyaction_timezone 无效: {tz_name!r}，"
            "请使用 IANA 格式，如 'Asia/Shanghai'"
        ) from exc


def load_config(path: str | Path = "config.toml") -> Config:
    data = _load_config_data(path)

    llm = _as_dict(data.get("llm"), field="llm")
    llm_main = _as_dict(llm.get("main"), field="llm.main")
    llm_fast = _as_dict(llm.get("fast"), field="llm.fast")
    llm_agent = _as_dict(llm.get("agent"), field="llm.agent")
    llm_vl = _as_dict(llm.get("vl"), field="llm.vl")
    agent_cfg = _as_dict(data.get("agent"), field="agent")
    agent_context = _as_dict(agent_cfg.get("context"), field="agent.context")
    agent_tools = _as_dict(agent_cfg.get("tools"), field="agent.tools")
    agent_maintenance = _as_dict(
        agent_cfg.get("maintenance"), field="agent.maintenance"
    )
    provider = str(llm.get("provider") or data["provider"])
    channels = _load_channels_config(data)
    proactive = _load_proactive_config(data)
    memory = _load_memory_config(data)
    peer_agents = _load_peer_agents_config(data)
    wiring = _load_wiring_config(data)

    return Config(
        provider=provider,
        model=str(llm_main.get("model") or data["model"]),
        api_key=_resolve(str(llm_main.get("api_key") or data.get("api_key", ""))),
        system_prompt=str(
            agent_cfg.get("system_prompt")
            or data.get("system_prompt", "You are a helpful assistant.")
        ),
        max_tokens=int(agent_cfg.get("max_tokens", data.get("max_tokens", 8192))),
        max_iterations=int(
            agent_cfg.get("max_iterations", data.get("max_iterations", 10))
        ),
        memory_window=int(
            agent_context.get("memory_window", data.get("memory_window", 40))
        ),
        base_url=str(llm_main.get("base_url") or data.get("base_url") or _PRESETS.get(provider) or ""),
        extra_body=_load_extra_body(data),
        channels=channels,
        proactive=proactive,
        memory_optimizer_enabled=_as_bool(
            agent_maintenance.get(
                "memory_optimizer_enabled",
                data.get("memory_optimizer_enabled", True),
            ),
            field="agent.maintenance.memory_optimizer_enabled",
        ),
        memory_optimizer_interval_seconds=int(
            agent_maintenance.get(
                "memory_optimizer_interval_seconds",
                data.get("memory_optimizer_interval_seconds", 64800),
            )
        ),
        light_model=str(llm_fast.get("model") or data.get("light_model", "")),
        light_api_key=_resolve(
            str(llm_fast.get("api_key") or data.get("light_api_key", ""))
        ),
        light_base_url=str(
            llm_fast.get("base_url") or data.get("light_base_url", "")
        ),
        agent_model=str(llm_agent.get("model") or data.get("agent_model", "")),
        agent_api_key=_resolve(
            str(llm_agent.get("api_key") or data.get("agent_api_key", ""))
        ),
        agent_base_url=str(
            llm_agent.get("base_url") or data.get("agent_base_url", "")
        ),
        memory=memory,
        tool_search_enabled=_as_bool(
            agent_tools.get("search_enabled", data.get("tool_search_enabled", False)),
            field="agent.tools.search_enabled",
        ),
        spawn_enabled=_as_bool(
            agent_tools.get("spawn_enabled", data.get("spawn_enabled", True)),
            field="agent.tools.spawn_enabled",
        ),
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
        multimodal=_as_bool(
            llm_main.get("multimodal", True), field="llm.main.multimodal"
        ),
        vl_model=str(llm_vl.get("model") or data.get("vl_model", "")),
        vl_api_key=_resolve(str(llm_vl.get("api_key") or data.get("vl_api_key", ""))),
        vl_base_url=str(llm_vl.get("base_url") or data.get("vl_base_url", "")),
        peer_agents=peer_agents,
        wiring=wiring,
    )


def _load_channels_config(data: dict) -> ChannelsConfig:
    channels_data = _as_dict(data.get("channels"), field="channels")

    telegram = None
    tg = _as_dict(channels_data.get("telegram"), field="channels.telegram")
    if tg:
        token = _normalize_optional_config_text(_resolve(str(tg.get("token", ""))))
        if _as_bool(
            tg.get("enabled", True), field="channels.telegram.enabled"
        ) and token:
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
        if _as_bool(
            qq_data.get("enabled", True), field="channels.qq.enabled"
        ) and bot_uin:
            groups = [
                QQGroupConfig(
                    group_id=str(
                        g["group_id"] if "group_id" in g else g["groupId"]
                    ),
                    allow_from=[
                        str(u)
                        for u in g.get("allow_from", g.get("allowFrom", []))
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

    cli_data = _as_dict(channels_data.get("cli"), field="channels.cli")
    chat_data = _as_dict(channels_data.get("chat"), field="channels.chat")
    chat = WebChatConfig(
        enabled=_as_bool(
            chat_data.get("enabled", True), field="channels.chat.enabled"
        ),
        host=str(chat_data.get("host", "127.0.0.1") or "127.0.0.1"),
        port=int(chat_data.get("port", 6322)),
        channel_name=str(chat_data.get("channel_name", "web") or "web"),
    )
    socket_value = channels_data.get("socket") or cli_data.get(
        "socket", DEFAULT_SOCKET
    )
    cli_session_key = str(cli_data.get("session_key") or "").strip()
    cli_channel = str(cli_data.get("channel") or "").strip()
    cli_chat_id = str(cli_data.get("chat_id") or "").strip()
    if not cli_session_key and cli_channel and cli_chat_id:
        cli_session_key = f"{cli_channel}:{cli_chat_id}"
    channels = ChannelsConfig(
        telegram=telegram,
        qq=qq,
        chat=chat,
        socket=_normalize_cli_socket_endpoint(socket_value),
        cli_session_key=cli_session_key,
    )
    return channels


def _load_proactive_config(data: dict) -> ProactiveConfig:
    proactive = ProactiveConfig()
    if p := data.get("proactive"):
        try:
            proactive = load_proactive_config(p)
        except ProactiveConfigError as e:
            print(f"❌ Proactive 配置错误: {e}", file=sys.stderr)
            sys.exit(1)
    return proactive


def _load_memory_config(data: dict) -> MemoryConfig:
    memory = _as_dict(data.get("memory"), field="memory")
    embedding = _as_dict(memory.get("embedding"), field="memory.embedding")
    raw_output_dimensionality = embedding.get("output_dimensionality")
    output_dimensionality = (
        int(raw_output_dimensionality)
        if raw_output_dimensionality not in (None, "")
        else None
    )
    if output_dimensionality is not None and output_dimensionality <= 0:
        raise ValueError("memory.embedding.output_dimensionality 必须大于 0")
    return MemoryConfig(
        enabled=_as_bool(memory.get("enabled", False), field="memory.enabled"),
        engine=str(memory.get("engine", "") or ""),
        embedding=MemoryEmbeddingConfig(
            model=str(embedding.get("model", "text-embedding-v3")),
            api_key=_resolve(str(embedding.get("api_key", ""))),
            base_url=str(embedding.get("base_url", "")),
            output_dimensionality=output_dimensionality,
        ),
    )


def _load_peer_agents_config(data: dict) -> list[PeerAgentConfig]:
    integrations = _as_dict(data.get("integrations"), field="integrations")
    peer_agents = integrations.get("peer_agents", data.get("peer_agents", []))
    return [
        PeerAgentConfig(
            name=pa["name"],
            base_url=pa["base_url"],
            launcher=pa["launcher"],
            cwd=pa.get("cwd"),
            description=pa.get("description", ""),
            health_path=pa.get("health_path", "/health"),
            startup_timeout_s=int(pa.get("startup_timeout_s", 30)),
            shutdown_timeout_s=int(pa.get("shutdown_timeout_s", 10)),
        )
        for pa in peer_agents
    ]


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
        memory=str(raw.get("memory", "default") or "default"),
        toolsets=list(toolsets),
    )


def _load_extra_body(data: dict) -> dict:
    llm = _as_dict(data.get("llm"), field="llm")
    llm_main = _as_dict(llm.get("main"), field="llm.main")
    extra_body = dict(_as_dict(data.get("extra_body"), field="extra_body"))
    thinking = llm_main.get("thinking")
    if thinking is not None and not isinstance(thinking, dict):
        raise ValueError("llm.main.thinking 必须是 TOML table")
    if thinking is not None:
        extra_body["thinking"] = thinking
    if "enable_thinking" in llm_main:
        extra_body["enable_thinking"] = _as_bool(
            llm_main["enable_thinking"], field="llm.main.enable_thinking"
        )
    if "reasoning_effort" in llm_main:
        effort = str(llm_main.get("reasoning_effort") or "").strip()
        if effort:
            extra_body["reasoning_effort"] = effort
    return extra_body


def _as_dict(value: object, *, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 TOML table")
    return value


def _resolve(value: str) -> str:
    resolved = re.sub(
        r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value
    )
    # 若仍是未展开的占位符，尝试从 workspace/memory/<VAR_NAME> 文件读取
    m = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if m:
        key_file = Path.home() / ".akashic" / "workspace" / "memory" / m.group(1)
        if key_file.exists():
            resolved = key_file.read_text(encoding="utf-8").strip()
    return resolved


def _as_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} 必须是布尔值")
    return value


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


__all__ = [
    "ChannelsConfig",
    "Config",
    "DEFAULT_SOCKET",
    "resolve_cli_socket_endpoint",
    "MemoryConfig",
    "MemoryEmbeddingConfig",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "_validated_timezone",
    "load_config",
]
