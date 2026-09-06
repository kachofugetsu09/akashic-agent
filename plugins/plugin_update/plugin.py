from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, Context, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES, UpdateStatus
from agent.plugin_composition.tasks import TASKS
from plugins.content.plugin import CONTENT, check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import CONTEXT
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.message import ContentPart, Output
from session.message_codec import json_value

from .tool import InstallPlugin, InstallInput, Request
from .validation import PLUGIN_VALIDATION, Validation

logger = logging.getLogger(__name__)
api_version = 3
name = "plugin_update"
version = "1.0.0"
desc = "按实际要求验证候选，排空后发布，并用原渠道报告结果"
inject = (PLUGIN_UPDATES, TOOLS, BINDINGS, OWNER_STATE, MESSAGE_CATALOG, MESSAGE_WRITERS,
          SESSION_ADMISSION, TASKS, DELIVERY, DELIVERY_SENDERS, CHAT_MODELS, CONTENT,
          CONTEXT, MATERIALS, MODEL_CALLS, REACT, TURN_PROJECTION)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=40, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)


async def apply(ctx: Context, config: Config) -> None:
    """工具只准备候选；普通来源拥有验证策略和通知，发布由 Core 排空。"""
    watcher: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[InstallPlugin]:
        if any(not isinstance(value, str) or not value for value in state.values()):
            raise ValueError("原更新 binding 缺少有效发送者引用")
        yield InstallPlugin(ctx, cast(Mapping[str, str], state))

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("plugin_install 不接收 binding 配置")
        return ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS))

    _ = await ctx.require(TOOLS).register(ctx, name="plugin_install",
        description="安装或更新插件，并按 validation_prompt 验证后发布；稍后单独报告结果",
        parameters=InstallInput.model_json_schema(), open=open_tool, capture=capture,
        idempotent=False, risk="external-side-effect")
    _ = await ctx.provide(PLUGIN_VALIDATION, Validation(ctx,
        max_steps=config.max_steps, max_output_tokens=config.max_output_tokens))

    async def validate(identity: str, request: Request) -> None:
        """本次存活运行验证一次；失败清理候选，进程重启只报告 Core 的回退。"""
        async with ctx.runtime_scope():
            updates = ctx.require(PLUGIN_UPDATES)
            try:
                async with updates.open_validation(ctx, identity) as scope:
                    result = await scope.require(PLUGIN_VALIDATION).run(identity, request.install)
            except Exception as error:
                # 原验证 owner 已记录真实错误；这里只能尝试撤销本次候选。
                try:
                    await updates.discard(ctx, identity, reason=str(error) or type(error).__name__)
                except Exception:
                    logger.exception("验证失败且资源尚未确认清理 update=%s", identity)
                return
            if result.passed:
                updates.publish(ctx, identity)
            else:
                await updates.discard(ctx, identity, reason=result.reason)

    async def report(identity: str, request: Request, status: UpdateStatus) -> None:
        """完成正文只写一次；重启沿原 Message 和发送回执查询，不重做更新。"""
        async with ctx.runtime_scope():
            terminal = status.phase in {"committed", "rolled_back"}
            message_id = identity + (":complete" if terminal else ":problem")
            reader = ctx.require(MESSAGE_CATALOG).reader(request.session_id)
            previous = reader.get(message_id)
            if previous is None:
                text = (f"插件 {status.plugin_id} 更新已完成。" if status.phase == "committed"
                        else f"插件 {status.plugin_id} 未发布：{status.error or '候选已回退'}")
                body = Output((ContentPart("text", text),), "complete")
            else:
                if not isinstance(previous.body, Output):
                    raise ValueError("原插件更新报告不是 Output")
                body = previous.body
            writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="plugin_update", source="plugin_update",
                body_types=(Output,), content={"text": check_text})(request.session_id)
            try:
                delivery = ctx.require(DELIVERY).open(ctx)
                sinks = () if request.sink is None else (request.sink,)
                _, selected = delivery.publish(writer, message_id, body, sinks, passive=True)
            finally:
                writer.expire()
            for sink in selected.sinks:
                receipt = await delivery.send(message_id, sink)
                if receipt.status != "delivered":
                    logger.warning("更新报告尚未确认发送 update=%s sink=%s receipt=%s", identity, sink, receipt)

    async def follow() -> None:
        """通知只驱动读取；没有持久执行队列、父 Turn barrier 或恢复后重跑。"""
        changed = asyncio.Event()
        active: set[str] = set()
        attempted: set[str] = set()
        reported: set[tuple[str, str]] = set()

        async def changes() -> None:
            async for _ in ctx.require(PLUGIN_UPDATES).changes(ctx):
                changed.set()

        async def run(identity: str, request: Request, status: UpdateStatus, *, validating: bool) -> None:
            try:
                if validating:
                    await validate(identity, request)
                else:
                    await report(identity, request, status)
            except Exception:
                # 保留原请求和领域回执；一个报告失败不抹掉其他已提交更新。
                logger.exception("插件更新来源未完成 update=%s", identity)
            finally:
                active.remove(identity)
                changed.set()

        async with asyncio.TaskGroup() as group:
            _ = group.create_task(changes())
            while True:
                _ = await changed.wait()
                changed.clear()
                async with ctx.runtime_scope():
                    updates = ctx.require(PLUGIN_UPDATES)
                    for identity, record in ctx.require(OWNER_STATE).open(ctx).list():
                        if identity in active:
                            continue
                        status = updates.read(ctx, identity)
                        if status is None or status.publishing:
                            continue
                        request = Request.model_validate(json_value(record.value))
                        validating = status.ready and not status.error and identity not in attempted
                        if validating:
                            attempted.add(identity)
                        elif status.phase in {"committed", "rolled_back"} or status.error:
                            phase = "complete" if status.phase in {"committed", "rolled_back"} else "problem"
                            if (identity, phase) in reported:
                                continue
                            reported.add((identity, phase))
                        else:
                            continue
                        active.add(identity)
                        _ = group.create_task(run(identity, request, status, validating=validating))

    async def start(_event: object) -> None:
        nonlocal watcher
        watcher = await ctx.spawn(follow(), name="plugin-updates")

    async def stop(_event: object) -> None:
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
