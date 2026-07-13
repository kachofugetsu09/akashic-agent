from __future__ import annotations

from typing import Any, Protocol

from agent.model_runtime.transports import CodexResponsesTransport
from agent.model_runtime.types import LLMResponse, ModelRequest


class ModelRuntime(Protocol):
    async def send(self, request: ModelRequest) -> LLMResponse: ...


class ChatRuntimeAdapter:
    """把既有 Chat Completions 实现适配为规范化 runtime。"""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    async def send(self, request: ModelRequest) -> LLMResponse:
        return await self.implementation.chat(
            messages=request.messages,
            tools=request.tools,
            model=request.model,
            max_tokens=request.max_output_tokens,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
            disable_thinking=request.disable_thinking,
            on_content_delta=request.on_delta,
        )


class ResponsesRuntime:
    """把 Responses transport 适配为规范化 runtime。"""

    def __init__(self, transport: CodexResponsesTransport) -> None:
        self.transport = transport

    async def send(self, request: ModelRequest) -> LLMResponse:
        return await self.transport.send(request)
