from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from agent.config import DEFAULT_SOCKET, _PRESETS, load_config
from agent.memory import MemoryStore
from core.observe.db import open_db
from memory2.store import MemoryStore2
from session.store import SessionStore

_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_LIGHT_MODEL = ""
_DEFAULT_EMBED_MODEL = "text-embedding-v3"
_DEFAULT_WORKSPACE = Path.home() / ".akashic" / "workspace"
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_NOW_TEMPLATE = "## 近期进行中\n\n## 待确认事项\n"
_MARKDOWN_DEFAULTS = {
    "MEMORY.md": "# 长期资料\n",
    "SELF.md": "# 自我模型\n",
    "NOW.md": _NOW_TEMPLATE,
    "HISTORY.md": "",
    "PENDING.md": "",
}
_PROACTIVE_STATE_DEFAULT = {
    "version": 5,
    "seen_items": {},
    "deliveries": {},
    "semantic_items": [],
    "rejection_cooldown": {},
    "bg_context_last_main_at": None,
    "context_only_last_at": {},
    "context_only_sent_timestamps": {},
    "drift_last_at": {},
}
_PROACTIVE_QUOTA_DEFAULT = {
    "version": 1,
    "window_key": "",
    "next_reset_at": "",
    "used": 0,
    "last_action_at": "",
}
_PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写用户当前对主动推送的明确要求和规则。

- 主 agent 负责维护这份文件。
- proactive agent 每轮都会读取它，并把它视为需要遵守的规则，不是普通参考建议。
- 这里适合写白名单、黑名单、过滤条件、优先级、必须先验证的步骤。
- 这里不提供新闻事实，不提供候选内容，只定义规则。
- 写结论即可，不要写冗长过程。
"""
_MEME_MANIFEST_DEFAULT = {"categories": {}}


def build_default_config_dict(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str = "",
    light_model: str = "",
    light_api_key: str = "",
    light_base_url: str = "",
    enable_memory_v2: bool = True,
    embed_model: str = _DEFAULT_EMBED_MODEL,
    embed_api_key: str = "",
    embed_base_url: str = "",
    enable_proactive: bool = False,
    proactive_channel: str = "telegram",
    proactive_chat_id: str = "",
    proactive_preset: str = "daily",
) -> dict:
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    if not provider:
        raise ValueError("provider 不能为空")
    if not model:
        raise ValueError("model 不能为空")
    if enable_memory_v2 and not str(embed_model or "").strip():
        raise ValueError("启用 memory_v2 时 embed_model 不能为空")

    preset_url = _PRESETS.get(provider)
    resolved_base_url = str(base_url or "").strip()
    if not resolved_base_url and preset_url:
        resolved_base_url = preset_url
    if not resolved_base_url and provider not in _PRESETS:
        raise ValueError(f"provider={provider!r} 不是内置预设，必须填写 base_url")

    raw: dict[str, object] = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "system_prompt": _DEFAULT_SYSTEM_PROMPT,
    }
    if resolved_base_url and resolved_base_url != preset_url:
        raw["base_url"] = resolved_base_url
    if str(light_model or "").strip():
        raw["light_model"] = str(light_model).strip()
        if str(light_api_key or "").strip():
            raw["light_api_key"] = str(light_api_key).strip()
        if str(light_base_url or "").strip():
            raw["light_base_url"] = str(light_base_url).strip()
    if enable_memory_v2:
        raw["memory_v2"] = {
            "enabled": True,
            "embed_model": str(embed_model).strip(),
        }
        if str(embed_api_key or "").strip():
            raw["memory_v2"]["api_key"] = str(embed_api_key).strip()
        if str(embed_base_url or "").strip():
            raw["memory_v2"]["base_url"] = str(embed_base_url).strip()
    if enable_proactive:
        raw["proactive"] = {
            "enabled": True,
            "default_channel": proactive_channel,
            "default_chat_id": proactive_chat_id,
            "preset": proactive_preset,
        }

    loaded = _materialize_loaded_config(raw)

    config: dict[str, object] = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "system_prompt": loaded.system_prompt,
    }
    if resolved_base_url and resolved_base_url != preset_url:
        config["base_url"] = resolved_base_url
    if str(loaded.light_model or "").strip():
        config["light_model"] = loaded.light_model
        if str(light_api_key or "").strip():
            config["light_api_key"] = light_api_key
        if str(light_base_url or "").strip():
            config["light_base_url"] = light_base_url
    if loaded.channels.socket != DEFAULT_SOCKET:
        config["channels"] = {"socket": loaded.channels.socket}
    if enable_memory_v2:
        config["memory_v2"] = {
            "enabled": True,
            "embed_model": loaded.memory_v2.embed_model,
        }
        if str(embed_api_key or "").strip():
            config["memory_v2"]["api_key"] = embed_api_key
        if str(embed_base_url or "").strip():
            config["memory_v2"]["base_url"] = embed_base_url
    if enable_proactive:
        config["proactive"] = {
            "enabled": True,
            "default_channel": loaded.proactive.default_channel,
            "default_chat_id": loaded.proactive.default_chat_id,
            "preset": proactive_preset,
        }
    return config


def initialize_workspace(workspace: Path, *, enable_memory_v2: bool, enable_proactive: bool) -> list[Path]:
    workspace.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for name in ("observe", "uploads", "skills", "memes", "subagent-runs"):
        path = workspace / name
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    _write_json_if_missing(workspace / "memes" / "manifest.json", _MEME_MANIFEST_DEFAULT)
    created.append(workspace / "memes" / "manifest.json")

    memory_store = MemoryStore(workspace)
    created.append(memory_store.memory_dir)
    created.append(memory_store._consolidation_db)
    for filename, content in _MARKDOWN_DEFAULTS.items():
        path = memory_store.memory_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        created.append(path)

    session_store = SessionStore(workspace / "sessions.db")
    session_store.close()
    created.append(workspace / "sessions.db")

    observe_conn = open_db(workspace / "observe" / "observe.db")
    observe_conn.close()
    created.append(workspace / "observe" / "observe.db")

    if enable_memory_v2:
        mem2 = MemoryStore2(workspace / "memory" / "memory2.db")
        mem2.close()
        created.append(workspace / "memory" / "memory2.db")

    _write_json_if_missing(workspace / "mcp_servers.json", {"servers": {}})
    created.append(workspace / "mcp_servers.json")
    _write_json_if_missing(workspace / "schedules.json", [])
    created.append(workspace / "schedules.json")

    if enable_proactive:
        _write_json_if_missing(workspace / "proactive_state.json", _PROACTIVE_STATE_DEFAULT)
        created.append(workspace / "proactive_state.json")
        _write_json_if_missing(workspace / "proactive_quota.json", _PROACTIVE_QUOTA_DEFAULT)
        created.append(workspace / "proactive_quota.json")
        proactive_context_path = workspace / "PROACTIVE_CONTEXT.md"
        if not proactive_context_path.exists():
            proactive_context_path.write_text(
                _PROACTIVE_CONTEXT_TEMPLATE,
                encoding="utf-8",
            )
        created.append(proactive_context_path)

    return created


def write_config(config_path: Path, config: dict) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def setup_telegram_channel(
    *,
    config_path: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    channels = data.setdefault("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("config.channels 必须是对象")

    telegram = channels.get("telegram")
    if telegram is None:
        telegram = {}
        channels["telegram"] = telegram
    if not isinstance(telegram, dict):
        raise ValueError("config.channels.telegram 必须是对象")

    proactive = data.get("proactive")
    proactive_exists = isinstance(proactive, dict)

    token_default = str(telegram.get("token", "${TELEGRAM_BOT_TOKEN}") or "${TELEGRAM_BOT_TOKEN}")
    token = _ask(input_fn, "telegram token", token_default)
    allow_default = ",".join(str(x) for x in telegram.get("allowFrom", []))
    allow_text = _ask(input_fn, "telegram allowFrom（逗号分隔，可留空）", allow_default)
    allow_from = [item.strip() for item in allow_text.split(",") if item.strip()]

    telegram["token"] = token
    telegram["allowFrom"] = allow_from

    bind_proactive_default = "y" if proactive_exists else "n"
    bind_proactive = _is_yes(
        _ask(input_fn, "设置为 proactive 默认渠道? [y/N]", bind_proactive_default),
        default=proactive_exists,
    )
    if bind_proactive:
        proactive = data.setdefault("proactive", {})
        if not isinstance(proactive, dict):
            raise ValueError("config.proactive 必须是对象")
        proactive["enabled"] = bool(proactive.get("enabled", True))
        proactive["preset"] = str(proactive.get("preset", "daily") or "daily")
        proactive["default_channel"] = "telegram"
        chat_id_default = str(proactive.get("default_chat_id", "") or "")
        proactive["default_chat_id"] = _ask(
            input_fn,
            "proactive 默认 telegram chat_id",
            chat_id_default,
        )

    write_config(config_path, data)
    output_fn(f"已写入 Telegram 渠道配置: {config_path}")
    output_fn(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def run_init(
    *,
    config_path: Path,
    workspace: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[Path, Path, dict, list[Path]]:
    if config_path.exists():
        raise FileExistsError(f"配置文件已存在: {config_path}")

    provider = _ask(input_fn, "provider", _DEFAULT_PROVIDER)
    model = _ask(input_fn, "model", _DEFAULT_MODEL)
    api_key = _ask(input_fn, "api_key（可留空，也可填 ${ENV_VAR}）", "")
    default_base_url = _PRESETS.get(provider, "")
    base_url = _ask(
        input_fn,
        "base_url（内置 provider 可留空）",
        default_base_url,
    )
    if provider not in _PRESETS and not base_url.strip():
        raise ValueError("自定义 provider 必须填写 base_url")

    light_model = _ask(input_fn, "light_model（可留空）", _DEFAULT_LIGHT_MODEL)
    light_api_key = ""
    light_base_url = ""
    if light_model.strip():
        light_api_key = _ask(
            input_fn,
            "light_api_key（可留空，留空则回退主 api_key）",
            "",
        )
        light_base_url = _ask(
            input_fn,
            "light_base_url（可留空，留空则回退主 base_url）",
            "",
        )

    memory_v2_answer = _ask(input_fn, "启用 memory_v2? [Y/n]", "y")
    enable_memory_v2 = _is_yes(memory_v2_answer, default=True)
    embed_model = _DEFAULT_EMBED_MODEL
    embed_api_key = ""
    embed_base_url = ""
    if enable_memory_v2:
        embed_model = _ask(input_fn, "embedding_model", _DEFAULT_EMBED_MODEL)
        if not embed_model.strip():
            raise ValueError("embedding_model 不能为空")
        embed_api_key = _ask(
            input_fn,
            "embedding_api_key（可留空，留空则回退主 api_key）",
            "",
        )
        embed_base_url = _ask(
            input_fn,
            "embedding_base_url（可留空，留空则回退主 base_url）",
            "",
        )

    proactive_answer = _ask(input_fn, "启用 proactive? [y/N]", "n")
    enable_proactive = _is_yes(proactive_answer, default=False)
    proactive_channel = "telegram"
    proactive_chat_id = ""
    if enable_proactive:
        proactive_channel = _ask(input_fn, "proactive 默认 channel", "telegram")
        proactive_chat_id = _ask(input_fn, "proactive 默认 chat_id", "")

    config = build_default_config_dict(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        light_model=light_model,
        light_api_key=light_api_key,
        light_base_url=light_base_url,
        enable_memory_v2=enable_memory_v2,
        embed_model=embed_model,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        enable_proactive=enable_proactive,
        proactive_channel=proactive_channel,
        proactive_chat_id=proactive_chat_id,
    )
    created = initialize_workspace(
        workspace,
        enable_memory_v2=enable_memory_v2,
        enable_proactive=enable_proactive,
    )
    write_config(config_path, config)
    output_fn(f"已生成配置: {config_path}")
    output_fn(f"已初始化 workspace: {workspace}")
    output_fn(json.dumps(config, ensure_ascii=False, indent=2))
    return config_path, workspace, config, created


def _materialize_loaded_config(raw: dict) -> object:
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=True) as f:
        json.dump(raw, f, ensure_ascii=False)
        f.flush()
        return load_config(f.name)


def _write_json_if_missing(path: Path, payload: object) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ask(input_fn: Callable[[str], str], label: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input_fn(f"{label}{suffix}: ").strip()
    return value or default


def _is_yes(value: str, *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"y", "yes", "1", "true"}
