"""
LLM Provider — OpenAI 兼容格式
支持所有兼容 OpenAI Chat Completions API 的服务：DeepSeek、Qwen、OpenAI 等。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


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
        request_timeout_s: float = 180.0,
        max_retries: int = 1,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._system = system_prompt
        self._extra_body = extra_body or {}
        self._request_timeout_s = max(1.0, float(request_timeout_s))
        self._max_retries = max(0, int(max_retries))

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

        resp = await self._create_with_retry(kwargs)
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

    async def _create_with_retry(self, kwargs: dict) -> object:
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=self._request_timeout_s,
                )
            except Exception as e:
                last_err = e
                retryable = self._is_retryable(e)
                exhausted = attempt >= self._max_retries
                if (not retryable) or exhausted:
                    raise
                wait_s = min(8.0, 1.0 * (2**attempt))
                logger.warning(
                    "[llm] 请求失败，将重试 attempt=%d/%d wait=%.1fs err=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    wait_s,
                    type(e).__name__,
                )
                await asyncio.sleep(wait_s)
        if last_err:
            raise last_err
        raise RuntimeError("LLM request failed without exception")

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        if isinstance(err, TimeoutError):
            return True
        text = str(err).lower()
        keywords = (
            "timeout",
            "timed out",
            "connect",
            "connection",
            "temporarily unavailable",
            "server error",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
        )
        return any(k in text for k in keywords)
