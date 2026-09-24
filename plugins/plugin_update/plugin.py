from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import cast

from agent.plugin_composition import Context, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES, UpdateStatus
from .inputs import CONTENT
from .inputs import DELIVERY, INPUT_ORIGIN
from .inputs import DELIVERY_SENDERS
from .inputs import TOOLS
from agent.plugin_contracts import ContentPart, Output

from .tool import InstallPlugin, InstallInput, Request, decode_request

logger = logging.getLogger(__name__)
api_version = 3
name = "plugin_update"
version = "1.0.0"
desc = "安装插件并在 selection accepted 或 active/failed 后用原渠道报告结果"
inject = (
    CONTENT,
    INPUT_ORIGIN,
    PLUGIN_UPDATES,
    TOOLS,
    BINDINGS,
    OWNER_STATE,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    DELIVERY,
    DELIVERY_SENDERS,
)


def result_message_id(identity: str, status: UpdateStatus) -> str:
    """Return the immutable result message identity for one terminal outcome."""
    if status.state not in {"active", "failed"}:
        raise ValueError(f"插件更新不是终态: {status.state}")
    return identity + ":result-" + status.state

async def apply(ctx: Context) -> None:
    """Register install and report durable runtime outcomes."""
    watcher: asyncio.Task[None] | None = None
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc)

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[InstallPlugin]:
        if any(not isinstance(value, str) or not value for value in state.values()):
            raise ValueError("原更新 binding 缺少有效发送者引用")
        yield InstallPlugin(ctx, cast(Mapping[str, str], state))

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("plugin_install 不接收 binding 配置")
        return ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS))

    _ = await catalog.register(
        ctx,
        name="plugin_install",
        description="安装或更新插件；selection accepted 后由宿主继续挂载并报告 active/failed",
        parameters=InstallInput.model_json_schema(),
        open=open_tool,
        capture=capture,
        idempotent=False,
        risk="external-side-effect",
    )
    async def report(identity: str, request: Request, status: UpdateStatus) -> None:
        """Send one durable result without overwriting historical result messages."""
        async with ctx.runtime_scope():
            message_id = result_message_id(identity, status)
            reader = ctx.require(MESSAGE_CATALOG).reader(request.session_id)
            previous = reader.get(message_id)
            if previous is None:
                text = (f"插件 {status.plugin_id} 已激活。" if status.state == "active"
                        else f"插件 {status.plugin_id} 更新失败：{status.error or '未知错误'}")
                body = Output((ContentPart("text", text),), "complete")
            else:
                if not isinstance(previous.body, Output):
                    raise ValueError("原插件更新报告不是 Output")
                body = previous.body
            writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="plugin_update", source="plugin_update",
                body_types=(Output,), content={"text": ctx.require(CONTENT).check_text})(request.session_id)
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
        reported: set[tuple[str, str]] = set()

        async def changes() -> None:
            async for _ in ctx.require(PLUGIN_UPDATES).changes(ctx):
                changed.set()

        async def run(identity: str, request: Request, status: UpdateStatus) -> None:
            try:
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
                        if status is None or status.state not in {"active", "failed"}:
                            continue
                        request = decode_request(record.value)
                        result = status.state
                        if (identity, result) in reported:
                            continue
                        reported.add((identity, result))
                        active.add(identity)
                        _ = group.create_task(run(identity, request, status))

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
