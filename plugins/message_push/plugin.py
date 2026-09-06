from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import cast

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_WRITERS
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.tools.plugin import TOOLS

from .tool import MessagePush, PushInput

api_version = 3
name = "message_push"
version = "1.0.0"
desc = "保存目标消息并通过原发送绑定推送文本、文件或图片"
inject = (TOOLS, DELIVERY, DELIVERY_SENDERS, BINDINGS, MESSAGE_WRITERS, ARTIFACT_IMPORT)


async def apply(ctx: Context, config: object) -> None:
    """普通工具注册不取得附件、Message writer 或发送资源。"""
    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[MessagePush]:
        if any(not isinstance(value, str) or not value for value in state.values()):
            raise ValueError("原推送 binding 缺少有效发送者引用")
        yield MessagePush(ctx, cast(Mapping[str, str], state))

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("message_push 不接收 binding 配置")
        return ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS))

    _ = await ctx.require(TOOLS).register(
        ctx, name="message_push", description="向指定渠道和会话发送消息、文件或图片",
        parameters=PushInput.model_json_schema(), open=open_tool, idempotent=True,
        capture=capture,
        risk="external-side-effect",
    )
