from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import cast

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS
from agent.plugin_composition.frames import CONTROL_FRAMES
from agent.plugin_composition.restart import RESTART_GATE
from agent.plugin_contracts.delivery_api import FINAL_OUTPUT_DELIVERY
from agent.plugin_contracts.delivery_api import DELIVERY
from agent.plugin_contracts.delivery import DELIVERY_SENDERS
from agent.plugin_contracts.tools import TOOLS
from agent.plugin_contracts.turn_projection import TURN_PROJECTION

from .tool import MessagePush, PushInput
from .restart import register_restart

api_version = 3
name = "message_push"
version = "1.0.0"
desc = "推送消息，并在 supervisor runtime 提供安全重启"
inject = (
    TOOLS,
    DELIVERY,
    DELIVERY_SENDERS,
    BINDINGS,
    MESSAGE_WRITERS,
    ARTIFACT_IMPORT,
    MESSAGE_CATALOG,
    RESTART_GATE,
    CONTROL_FRAMES,
)

_RESTART_DEPS = (
    BINDINGS,
    MESSAGE_CATALOG,
    TURN_PROJECTION,
    FINAL_OUTPUT_DELIVERY,
    RESTART_GATE,
    CONTROL_FRAMES,
)


async def apply(ctx: Context, config: object) -> None:
    """普通工具注册不取得附件、Message writer 或发送资源。"""
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[MessagePush]:
        if any(not isinstance(value, str) or not value for value in state.values()):
            raise ValueError("原推送 binding 缺少有效发送者引用")
        yield MessagePush(ctx, cast(Mapping[str, str], state))

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("message_push 不接收 binding 配置")
        return ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS))

    _ = await catalog.register(
        ctx,
        name="message_push",
        description="向指定渠道和会话发送消息、文件或图片",
        parameters=PushInput.model_json_schema(),
        open=open_tool,
        idempotent=True,
        capture=capture,
        risk="external-side-effect",
    )

    async def restart(child: Context) -> None:
        _ = await register_restart(child)

    _ = await ctx.inject(_RESTART_DEPS, restart, name="restart")
