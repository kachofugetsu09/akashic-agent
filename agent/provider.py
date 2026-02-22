"""
LLM Provider — OpenAI 兼容格式
支持所有兼容 OpenAI Chat Completions API 的服务：DeepSeek、Qwen、OpenAI 等。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from openai import AsyncOpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider:

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        system_prompt: str = "",
        extra_body: dict | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._system = system_prompt
        self._extra_body = extra_body or {}

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        # 系统提示作为第一条消息（若 messages 已自带 system 消息则不再重复添加）
        already_has_system = messages and messages[0].get("role") == "system"
        full_messages = (
            [{"role": "system", "content": self._system}, *messages]
            if self._system and not already_has_system else messages
        )
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=full_messages)
        if tools:
            kwargs["tools"] = tools
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        resp = await self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(content=msg.content, tool_calls=tool_calls)
