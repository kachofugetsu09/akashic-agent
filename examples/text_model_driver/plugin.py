"""独立的文本 HTTP 驱动示例；通过现有 Models 注册口替换默认驱动。"""
from __future__ import annotations

import json
import httpx

from agent.plugin_composition import Context
from plugins.ledger.contract import json_value
from plugins.models.contract import (
    MODEL_DRIVERS, BoundModelDescriptor, CredentialHandle, DriverConnection,
    DriverConnectionDescriptor, LLMResponse, ModelDriverDefinition, ModelRequest,
)

api_version = 3
name = "text-model-driver"
version = "1.0.0"
inject = (MODEL_DRIVERS,)


class TextModel:
    """示例只读取完整文本响应；不实现流、工具调用或 embedding。"""
    max_tool_schemas = None

    def __init__(self, client: httpx.AsyncClient, model: str):
        self._client, self._model = client, model

    def estimate_context_tokens(self, messages, tools=()) -> int:
        return len(json.dumps(json_value([messages, tools]), ensure_ascii=False).encode())

    def estimate_appended_message_tokens(self, messages) -> int:
        return self.estimate_context_tokens(messages)

    async def complete(self, request: ModelRequest) -> LLMResponse:
        response = await self._client.post("chat/completions", json={
            "model": self._model, "messages": json_value(request.messages), "stream": False,
        })
        response.raise_for_status()
        value = response.json()["choices"][0]["message"]
        if value.get("tool_calls") or not isinstance(value.get("content"), str):
            raise ValueError("示例驱动只接受文本回答")
        return LLMResponse(value["content"])


async def open_connection(descriptor: DriverConnectionDescriptor, credential: CredentialHandle) -> DriverConnection:
    """每次打开只持有本次 credential 与 HTTP client，关闭由 Models 租约执行。"""
    if credential.connection_id != descriptor.connection_id or credential.auth_identity != descriptor.auth_identity:
        raise PermissionError("credential scope does not match connection")
    secret = await credential.read()
    client = httpx.AsyncClient(base_url=descriptor.endpoint.rstrip("/") + "/",
        headers={"Authorization": "Bearer " + secret["api_key"]}, trust_env=False)
    def chat(model: BoundModelDescriptor, _config) -> TextModel:
        return TextModel(client, model.model)
    def embedding(_model, _config):
        raise NotImplementedError("示例驱动不提供 embedding")
    return DriverConnection(chat, embedding, close=client.aclose)


async def apply(ctx: Context) -> None:
    await ctx.require(MODEL_DRIVERS).register(ctx, ModelDriverDefinition(
        driver_id="openai-compatible", contract_version="1", open=open_connection))
