"""真实 Message/SourceSession 验证 Telegram 控制跟随异名来源路由。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.plugin_composition import CompositionRoot, CredentialRef
from agent.plugin_composition.channels import CHANNEL_INPUT, CHANNELS, ChannelInboundMessage, RawInbound
from agent.plugin_composition.credentials import CREDENTIALS
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.model import FiberState, PluginRuntime
from agent.plugin_composition.tasks import Tasks
from agent.plugin_contracts.sources import SOURCES
from plugins.sources import plugin as sources
from plugins.sources.session import SourceSession
from plugins.telegram_channel import plugin as telegram
from session.log import MessageCatalog, MessageLog
from session.message import ContentPart, ContentReferences, Control, Input


async def check(workspace: Path) -> None:
    root, tasks = CompositionRoot("source-route-scenario"), Tasks()
    log = MessageLog(workspace / "sessions.db")
    opened = {}
    writers = []
    definitions = []
    release = asyncio.Event()

    class CaptureChannel:
        async def register(self, _ctx, definition):
            definitions.append(definition)

    class RejectCredentials:
        def create(self, *args, **kwargs):
            raise AssertionError("此场景不得连接 Telegram 或读取凭据")

    def registration(name, channels):
        async def apply(ctx):
            def open(session_id):
                key = name, session_id
                if key not in opened:
                    inputs = log.writer(session_id, source=name, author="user", body_types=(Input,),
                                        content={"text": lambda _: ContentReferences()})
                    controls = log.writer(session_id, source=name, author="app", body_types=(Control,), content={})
                    writers.extend((inputs, controls))
                    opened[key] = SourceSession(reader=log.reader(session_id), inputs=inputs, controls=controls, tasks=tasks)
                return opened[key]

            async def accept(session_id, message_id, message):
                return await open(session_id).accept(message_id, Input((ContentPart("text", message.content),)))
            await ctx.require(SOURCES).register(ctx, name=name, channels=channels, open=open, accept=accept,
                                                needs_reply=lambda reader: SourceSession.needs_reply(reader, name))
        return apply

    def raw(message_id, text, room="room"):
        return RawInbound(message_id=message_id, provider_identity="user", recipient=room,
                          message=ChannelInboundMessage("telegram", "user", room, text, datetime.now(UTC), {}))

    try:
        await root.context.provide(CHANNELS, CaptureChannel())
        await root.context.provide(CREDENTIALS, RejectCredentials())
        await root.context.provide(MESSAGE_CATALOG, MessageCatalog(log))
        runtime = PluginRuntime("telegram", "generation", workspace, workspace / "telegram", workspace,
                                {"enabled": True, "token": CredentialRef(("scenario", "unused"))})
        async def legacy_routes(ctx):
            router = sources.Sources(ctx)
            await ctx.provide(SOURCES, router)
            await ctx.provide(CHANNEL_INPUT, router.accept)
        legacy = await root.mount(legacy_routes, name="legacy-routes")
        waiting = await root.mount(telegram.run, name="old-pair", inject=telegram.function_inject, runtime=runtime)
        assert waiting.state is FiberState.PENDING and not definitions
        await waiting.dispose()
        await legacy.dispose()
        await root.mount(sources.apply, name="routes")
        await root.mount(registration("default_lane", None), name="default", inject=(SOURCES,))
        dedicated = await root.mount(registration("assistant_lane", ("telegram",)), name="dedicated", inject=(SOURCES,))
        runtime = PluginRuntime("telegram", "generation", workspace, workspace / "telegram", workspace,
                                {"enabled": True, "token": CredentialRef(("scenario", "unused"))})
        channel = await root.mount(telegram.run, name="telegram", inject=telegram.function_inject, runtime=runtime)
        assert channel.state is FiberState.ACTIVE, channel.error
        interrupt = definitions[0].interrupt
        assert await interrupt(raw("empty-stop", "/stop", "empty")) is False
        assert log.reader("telegram:empty").head() == -1
        router = root.context.require(SOURCES)
        message = raw("input", "work")
        accepted = await router.accept("telegram:room", message.message_id, message.message)
        assert accepted.source == "assistant_lane"

        # 中断先追加原来源 Control，等待旧工作排空后才允许渠道确认。
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def program(_task, _reader, _source):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
                await release.wait()
        task = await opened["assistant_lane", "telegram:room"].start(program)
        await entered.wait()
        stopping = asyncio.create_task(interrupt(raw("stop", "/stop")))
        await asyncio.wait_for(cancelled.wait(), 5)
        assert not stopping.done()
        release.set()
        assert await stopping is True
        result = await asyncio.gather(task.join(), return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        rows = log.reader("telegram:room").snapshot()
        assert [row.message_id for row in rows] == ["input", "stop"]
        assert rows[-1].source == "assistant_lane" and rows[-1].body == Control("pause", accepted.seq)

        # 卸载专属路由后，输入和中断一起回到默认来源。
        await dedicated.dispose()
        next_input = raw("next", "next work")
        accepted = await router.accept("telegram:room", next_input.message_id, next_input.message)
        assert accepted.source == "default_lane"
        assert await interrupt(raw("next-stop", "/stop")) is True
        assert log.reader("telegram:room").get("next-stop").source == "default_lane"
    finally:
        release.set()
        await tasks.close()
        await root.dispose()
        for writer in writers:
            writer.expire()
        log.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-source-route-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: alternate/default route, append-only pause, drain before ack; no Telegram network")
