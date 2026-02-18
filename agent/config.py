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

# 内置 OpenAI 兼容服务的 base_url 预设
_PRESETS: dict[str, str] = {
    "qwen":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai":   "https://api.openai.com/v1",
}

@dataclass
class Config:
    provider: str           # "qwen" | "deepseek" | "openai" | 自定义名
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    base_url: str | None = None     # 内置预设或显式指定

    @classmethod
    def load(cls, path: str | Path = "config.json") -> Config:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        provider = data["provider"]
        api_key = _resolve(data.get("api_key", ""))

        # base_url 优先级：显式配置 > 内置预设 > None（Anthropic 不需要）
        base_url = data.get("base_url") or _PRESETS.get(provider)

        return cls(
            provider=provider,
            model=data["model"],
            api_key=api_key,
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            max_tokens=data.get("max_tokens", 8192),
            max_iterations=data.get("max_iterations", 10),
            base_url=base_url,
        )


def _resolve(value: str) -> str:
    """将 ${VAR_NAME} 替换为对应环境变量值"""
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
