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
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.config_models import (
    AppServerConfig,
    ChannelsConfig,
    Config,
    MemoryConfig,
    MemoryEmbeddingConfig,
    MobileKeyEncryptionConfig,
    MobileRealtimeConfig,
    ModelRuntimeConfig,
    PeerAgentConfig,
    QQChannelConfig,
    QQGroupConfig,
    TelegramChannelConfig,
    WebChatConfig,
    WiringConfig,
)
from proactive_v2.config import ProactiveConfig
from proactive_v2.config_loader import ProactiveConfigError, load_proactive_config
from agent.model_runtime.auth.store import CredentialStore
from agent.model_runtime.context_policy import recommended_context_settings
from agent.model_runtime.provider_profiles import get_provider_profile

_PRESETS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}
_DEFAULT_TOOLSETS = ("meta_common", "spawn", "schedule")

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


def load_config(
    path: str | Path = "config.toml",
    *,
    workspace: str | Path,
) -> Config:
    workspace_path = Path(workspace)
    data = _load_config_data(path)

    llm = _as_dict(data.get("llm"), field="llm")
    runtime_id, llm_main, model_runtimes = _load_llm_runtimes(llm, workspace_path)
    fast_runtime_id, llm_fast = _load_role_runtime(llm, "fast", runtime_id)
    agent_runtime_id, llm_agent = _load_role_runtime(llm, "agent", runtime_id)
    vl_runtime_id, llm_vl = _load_role_runtime(llm, "vl", runtime_id)
    agent_cfg = _as_dict(data.get("agent"), field="agent")
    agent_context = _as_dict(agent_cfg.get("context"), field="agent.context")
    agent_tools = _as_dict(agent_cfg.get("tools"), field="agent.tools")
    agent_maintenance = _as_dict(
        agent_cfg.get("maintenance"), field="agent.maintenance"
    )
    provider = str(llm_main.get("provider") or "").lower()
    if not provider:
        raise ValueError("必须配置 llm provider")
    channels = _load_channels_config(data, workspace_path)
    app_server = _load_app_server_config(data)
    mobile_realtime = _load_mobile_realtime_config(data)
    if mobile_realtime.enabled and (
        not channels.chat.enabled
        or channels.chat.host not in {"127.0.0.1", "localhost", "::1"}
    ):
        raise ValueError(
            "mobile_realtime 启用时，本机配对入口 channels.chat 必须监听 loopback"
        )
    proactive = _load_proactive_config(data)
    memory = _load_memory_config(data, workspace_path)
    peer_agents = _load_peer_agents_config(data)
    wiring = _load_wiring_config(data)

    return Config(
        provider=provider,
        model=str(llm_main.get("model") or ""),
        api_key=(
            ""
            if provider == "codex"
            else _load_api_key(
                auth_id=str(llm_main.get("auth") or ""),
                inline_value=str(llm_main.get("api_key") or ""),
                workspace=workspace_path,
            )
        ),
        system_prompt=str(
            agent_cfg.get("system_prompt")
            or data.get("system_prompt", "You are a helpful assistant.")
        ),
        max_tokens=int(
            llm_main.get(
                "max_output_tokens",
                agent_cfg.get("max_tokens", data.get("max_tokens", 8192)),
            )
        ),
        max_iterations=int(
            agent_cfg.get("max_iterations", data.get("max_iterations", 10))
        ),
        memory_window=_load_memory_window(data, agent_context, llm_main),
        base_url=_model_base_url(
            provider, llm_main.get("base_url")
        ),
        extra_body=_load_extra_body(data, llm_main),
        channels=channels,
        app_server=app_server,
        mobile_realtime=mobile_realtime,
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
        light_model=str(llm_fast.get("model") or ""),
        light_api_key=_load_api_key(
            auth_id=str(llm_fast.get("auth") or ""),
            inline_value=str(llm_fast.get("api_key") or ""),
            workspace=workspace_path,
        ),
        light_base_url=str(
            llm_fast.get("base_url") or ""
        ),
        agent_model=str(llm_agent.get("model") or ""),
        agent_api_key=_load_api_key(
            auth_id=str(llm_agent.get("auth") or ""),
            inline_value=str(llm_agent.get("api_key") or ""),
            workspace=workspace_path,
        ),
        agent_base_url=str(
            llm_agent.get("base_url") or ""
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
        multimodal="image" in model_runtimes[runtime_id].input_modalities,
        vl_model=str(llm_vl.get("model") or ""),
        vl_api_key=_load_api_key(
            auth_id=str(llm_vl.get("auth") or ""),
            inline_value=str(llm_vl.get("api_key") or ""),
            workspace=workspace_path,
        ),
        vl_base_url=str(llm_vl.get("base_url") or ""),
        peer_agents=peer_agents,
        wiring=wiring,
        runtime_id=runtime_id,
        auth_id=str(llm_main.get("auth") or ""),
        context_window=int(llm_main.get("context_window") or 0),
        reasoning_effort=str(llm_main.get("reasoning_effort") or ""),
        input_modalities=tuple(str(item) for item in llm_main.get("input_modalities", ["text"])),
        effective_context_percent=float(llm_main.get("effective_context_percent", 0.9)),
        use_responses_lite=_as_bool(
            llm_main.get("use_responses_lite", False),
            field="llm.main.use_responses_lite",
        ),
        supports_parallel_tool_calls=_as_bool(
            llm_main.get("supports_parallel_tool_calls", True),
            field="llm.main.supports_parallel_tool_calls",
        ),
        reasoning_summary=_as_reasoning_summary(
            llm_main.get(
                "reasoning_summary",
                "auto" if provider == "codex" else None,
            ),
            field="llm.main.reasoning_summary",
        ),
        model_runtimes=model_runtimes,
        fast_runtime_id=fast_runtime_id,
        agent_runtime_id=agent_runtime_id,
        vl_runtime_id=vl_runtime_id,
    )


def _load_channels_config(data: dict, workspace: Path) -> ChannelsConfig:
    channels_data = _as_dict(data.get("channels"), field="channels")

    telegram = None
    tg = _as_dict(channels_data.get("telegram"), field="channels.telegram")
    if tg:
        token = _normalize_optional_config_text(
            _resolve(str(tg.get("token", "")), workspace)
        )
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

    if "socket" in channels_data or "cli" in channels_data:
        raise ValueError(
            "旧 channels.socket/channels.cli 配置已删除；请改用 [app_server] listen = \"\""
        )
    chat_data = _as_dict(channels_data.get("chat"), field="channels.chat")
    chat = WebChatConfig(
        enabled=_as_bool(
            chat_data.get("enabled", True), field="channels.chat.enabled"
        ),
        host=str(chat_data.get("host", "127.0.0.1") or "127.0.0.1"),
        port=int(chat_data.get("port", 6322)),
        channel_name=str(chat_data.get("channel_name", "web") or "web"),
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
    for name in ("max_connections", "ingress_queue_size", "outbound_queue_size", "max_message_bytes"):
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
    if config.key_encryption.provider != "secret_service":
        raise ValueError(
            "mobile_realtime.key_encryption.provider 只支持 secret_service"
        )
    if not config.key_encryption.master_key_namespace:
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


def _load_proactive_config(data: dict) -> ProactiveConfig:
    proactive = ProactiveConfig()
    if p := data.get("proactive"):
        try:
            proactive = load_proactive_config(p)
        except ProactiveConfigError as e:
            print(f"❌ Proactive 配置错误: {e}", file=sys.stderr)
            sys.exit(1)
    return proactive


def _load_memory_config(data: dict, workspace: Path) -> MemoryConfig:
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
            api_key=_load_api_key(
                auth_id=str(embedding.get("auth") or ""),
                inline_value=str(embedding.get("api_key", "")),
                workspace=workspace,
            ),
            base_url=str(embedding.get("base_url", "")),
            output_dimensionality=output_dimensionality,
            auth=str(embedding.get("auth") or ""),
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


def _load_extra_body(data: dict, llm_main: dict | None = None) -> dict:
    llm = _as_dict(data.get("llm"), field="llm")
    if llm_main is None:
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


def _load_llm_runtimes(
    llm: dict,
    workspace: Path,
) -> tuple[str, dict, dict[str, ModelRuntimeConfig]]:
    """在配置边界解析 named runtimes，并拒绝未迁移的旧结构。"""
    main_value = llm.get("main")
    if not isinstance(main_value, str):
        raise ValueError("llm.main 必须引用 named runtime；请先执行启动迁移")
    runtimes = _as_dict(llm.get("runtimes"), field="llm.runtimes")
    raw_main = _as_dict(runtimes.get(main_value), field=f"llm.runtimes.{main_value}")
    if not raw_main:
        raise ValueError(f"llm.main 引用不存在的 runtime: {main_value}")
    parsed: dict[str, ModelRuntimeConfig] = {}
    for runtime_id, raw in runtimes.items():
        item = _as_dict(raw, field=f"llm.runtimes.{runtime_id}")
        modalities = item.get("input_modalities", ["text"])
        if not isinstance(modalities, list) or not all(isinstance(v, str) for v in modalities):
            raise ValueError(f"llm.runtimes.{runtime_id}.input_modalities 必须是字符串数组")
        provider = str(item.get("provider") or "").lower()
        auth_id = str(item.get("auth") or "")
        parsed[runtime_id] = ModelRuntimeConfig(
            runtime_id=runtime_id,
            provider=provider,
            model=str(item.get("model") or ""),
            auth=auth_id,
            api_key=(
                ""
                if provider == "codex"
                else _load_api_key(
                    auth_id=auth_id,
                    inline_value=str(item.get("api_key") or ""),
                    workspace=workspace,
                )
            ),
            base_url=_model_base_url(provider, item.get("base_url")),
            reasoning_effort=str(item.get("reasoning_effort") or ""),
            context_window=int(item.get("context_window") or 0),
            max_output_tokens=int(item.get("max_output_tokens") or 8192),
            input_modalities=tuple(modalities),
            effective_context_percent=float(item.get("effective_context_percent", 0.9)),
            use_responses_lite=_as_bool(
                item.get("use_responses_lite", False),
                field=f"llm.runtimes.{runtime_id}.use_responses_lite",
            ),
            supports_parallel_tool_calls=_as_bool(
                item.get("supports_parallel_tool_calls", True),
                field=f"llm.runtimes.{runtime_id}.supports_parallel_tool_calls",
            ),
            reasoning_summary=_as_reasoning_summary(
                item.get(
                    "reasoning_summary",
                    "auto" if provider == "codex" else None,
                ),
                field=f"llm.runtimes.{runtime_id}.reasoning_summary",
            ),
        )
    return main_value, raw_main, parsed


def _load_memory_window(data: dict, agent_context: dict, llm_main: dict) -> int:
    """显式配置优先，否则根据主模型有效上下文推导历史窗口。"""

    # 1. 保留现有配置的精确覆盖语义。
    configured = agent_context.get("memory_window", data.get("memory_window"))
    if configured is not None:
        return int(configured)

    # 2. 新 runtime 自动使用统一上下文策略；旧配置继续沿用 40。
    context_window = int(llm_main.get("context_window") or 0)
    if context_window <= 0:
        return 40
    effective_percent = float(llm_main.get("effective_context_percent", 0.9))
    return recommended_context_settings(context_window, effective_percent).memory_window


def _load_role_runtime(
    llm: dict, role: str, main_runtime_id: str
) -> tuple[str, dict]:
    value = llm.get(role)
    if value is None:
        return "", {}
    if not isinstance(value, str):
        raise ValueError(f"llm.{role} 必须引用 named runtime；请先执行启动迁移")
    if value == main_runtime_id:
        return value, {}
    runtimes = _as_dict(llm.get("runtimes"), field="llm.runtimes")
    runtime = _as_dict(runtimes.get(value), field=f"llm.runtimes.{value}")
    if not runtime:
        raise ValueError(f"llm.{role} 引用不存在的 runtime: {value}")
    return value, runtime


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


def _load_api_key(*, auth_id: str, inline_value: str, workspace: Path) -> str:
    if auth_id:
        return CredentialStore().api_key(auth_id)
    return _resolve(inline_value, workspace)


def _model_base_url(provider: str, configured: object) -> str:
    profile = get_provider_profile(provider)
    return str(
        configured
        or ("https://chatgpt.com/backend-api/codex" if provider == "codex" else "")
        or (profile.default_base_url if profile is not None else "")
        or _PRESETS.get(provider)
        or ""
    )


def _as_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} 必须是布尔值")
    return value


def _as_reasoning_summary(value: object, *, field: str) -> str:
    summary = str(value or "none")
    if summary not in {"none", "auto", "concise", "detailed"}:
        raise ValueError(f"{field} 必须是 none、auto、concise 或 detailed")
    return summary


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
    "resolve_app_server_endpoint",
    "MemoryConfig",
    "MemoryEmbeddingConfig",
    "QQChannelConfig",
    "QQGroupConfig",
    "TelegramChannelConfig",
    "_validated_timezone",
    "load_config",
]
