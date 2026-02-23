"""
配置加载模块
从 config.json 读取配置，支持 ${ENV_VAR} 格式的环境变量插值。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from proactive.loop import ProactiveConfig

_PRESETS: dict[str, str] = {
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai":   "https://api.openai.com/v1",
}

# CLI channel 默认 Unix socket 路径
DEFAULT_SOCKET = "/tmp/akasic.sock"


@dataclass
class TelegramChannelConfig:
    token: str
    allow_from: list[str] = field(default_factory=list)  # 空 = 允许所有人


@dataclass
class QQGroupConfig:
    group_id: str
    allow_from: list[str] = field(default_factory=list)  # 空 = 群内所有人
    require_at: bool = True                               # 仅响应 @ 消息


@dataclass
class QQChannelConfig:
    bot_uin: str                                              # Bot 的 QQ 号
    allow_from: list[str] = field(default_factory=list)      # 私聊白名单，空 = 允许所有人
    groups: list[QQGroupConfig] = field(default_factory=list) # 群组配置


@dataclass
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
    qq: QQChannelConfig | None = None
    socket: str = DEFAULT_SOCKET        # IPC server 监听路径


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    base_url: str | None = None
    extra_body: dict = field(default_factory=dict)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)

    @classmethod
    def load(cls, path: str | Path = "config.json") -> Config:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        provider = data["provider"]
        channels_data = data.get("channels", {})

        telegram = None
        if tg := channels_data.get("telegram"):
            telegram = TelegramChannelConfig(
                token=_resolve(tg["token"]),
                allow_from=[str(u) for u in tg.get("allowFrom", [])],
            )

        qq = None
        if qq_data := channels_data.get("qq"):
            groups = [
                QQGroupConfig(
                    group_id=str(g["groupId"]),
                    allow_from=[str(u) for u in g.get("allowFrom", [])],
                    require_at=g.get("requireAt", True),
                )
                for g in qq_data.get("groups", [])
            ]
            qq = QQChannelConfig(
                bot_uin=str(qq_data["bot_uin"]),
                allow_from=[str(u) for u in qq_data.get("allowFrom", [])],
                groups=groups,
            )

        channels = ChannelsConfig(
            telegram=telegram,
            qq=qq,
            socket=channels_data.get("cli", {}).get("socket", DEFAULT_SOCKET),
        )

        # ── proactive ──
        proactive = ProactiveConfig()
        if p := data.get("proactive"):
            proactive = ProactiveConfig(
                enabled=p.get("enabled", False),
                interval_seconds=p.get("interval_seconds", 1800),
                threshold=p.get("threshold", 0.70),
                items_per_source=p.get("items_per_source", 3),
                recent_chat_messages=p.get("recent_chat_messages", 20),
                model=p.get("model", ""),
                default_channel=p.get("default_channel", "telegram"),
                default_chat_id=str(p.get("default_chat_id", "")),
                dedupe_seen_ttl_hours=int(p.get("dedupe_seen_ttl_hours", 24 * 14)),
                delivery_dedupe_hours=int(p.get("delivery_dedupe_hours", 24)),
                only_new_items_trigger=bool(p.get("only_new_items_trigger", True)),
                semantic_dedupe_enabled=bool(p.get("semantic_dedupe_enabled", True)),
                semantic_dedupe_threshold=float(p.get("semantic_dedupe_threshold", 0.90)),
                semantic_dedupe_window_hours=int(p.get("semantic_dedupe_window_hours", 72)),
                semantic_dedupe_max_candidates=int(p.get("semantic_dedupe_max_candidates", 200)),
                semantic_dedupe_ngram=int(p.get("semantic_dedupe_ngram", 3)),
                semantic_dedupe_text_max_chars=int(p.get("semantic_dedupe_text_max_chars", 240)),
                use_global_memory=bool(p.get("use_global_memory", True)),
                global_memory_max_chars=int(p.get("global_memory_max_chars", 3000)),
                # ── 能量/冲动 ──
                energy_cool_threshold=float(p.get("energy_cool_threshold", 0.20)),
                energy_crisis_threshold=float(p.get("energy_crisis_threshold", 0.05)),
                energy_min_urge=float(p.get("energy_min_urge", 0.10)),
                # ── 静默时间窗口（本地时间）──
                quiet_hours_start=int(p.get("quiet_hours_start", 23)),
                quiet_hours_end=int(p.get("quiet_hours_end", 8)),
                quiet_hours_weight=float(p.get("quiet_hours_weight", 0.0)),
                # ── tick 间隔（秒）──
                tick_interval_high=int(p.get("tick_interval_high", 7200)),
                tick_interval_normal=int(p.get("tick_interval_normal", 1800)),
                tick_interval_low=int(p.get("tick_interval_low", 900)),
                tick_interval_crisis=int(p.get("tick_interval_crisis", 600)),
                tick_jitter=float(p.get("tick_jitter", 0.3)),
            )

        return cls(
            provider=provider,
            model=data["model"],
            api_key=_resolve(data.get("api_key", "")),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            max_tokens=data.get("max_tokens", 8192),
            max_iterations=data.get("max_iterations", 10),
            base_url=data.get("base_url") or _PRESETS.get(provider),
            extra_body=data.get("extra_body", {}),
            channels=channels,
            proactive=proactive,
        )


def _resolve(value: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
