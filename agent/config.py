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
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
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
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)

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

        channels = ChannelsConfig(
            telegram=telegram,
            socket=channels_data.get("cli", {}).get("socket", DEFAULT_SOCKET),
        )

        return cls(
            provider=provider,
            model=data["model"],
            api_key=_resolve(data.get("api_key", "")),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            max_tokens=data.get("max_tokens", 8192),
            max_iterations=data.get("max_iterations", 10),
            base_url=data.get("base_url") or _PRESETS.get(provider),
            channels=channels,
        )


def _resolve(value: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
