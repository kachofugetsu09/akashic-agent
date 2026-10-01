"""实际安装 Source 读取能力组，核对旧归档恢复和提交占位交错。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--previous-source", type=Path, required=True,
                    help="仍提供 sources.v3/source.session.v2 的完整旧源码")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT_V2, ChannelInboundMessage
from agent.plugin_composition.model import FiberState
from agent.plugin_contracts.reply import REPLY_COMPLETION
from agent.plugin_contracts.sources import SOURCES_V3, SOURCE_SESSION_V2, SOURCES_V4, SOURCE_CHANGED_V3
from session.message import Input, Output, ToolResult
from tests.test_default_reply import application


def copy_previous(destination, names):
    """只复制显式给出的旧插件源码，安装 cache 完全由临时 Manager 拥有。"""
    for name in names:
        shutil.rmtree(destination / name)
        shutil.copytree(args.previous_source / "plugins" / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__"))


async def versions(path):
    """半组不能启用；完整旧组执行后，破坏性局部更新拒绝并恢复旧组。"""
    old_keys = (SOURCES_V3, SOURCE_SESSION_V2)
    async with application(path / "current", replying=True) as (_log, host):
        root = host.live_root
        applied = []
        async def old_actor(_ctx):
            applied.append(True)
        actor = await root.mount(old_actor, name="old-source-actor", inject=old_keys)
        assert actor.state is FiberState.PENDING and not applied
        for key in old_keys:
            try:
                root.context.require(key)
            except CompositionError:
                pass
            else:
                raise AssertionError("当前 provider 发布了旧 alias")
        await actor.dispose()
        assert root.context.require(SOURCES_V4) is not None

    async with application(path / "reverse", replying=True,
                           extra_sources=lambda destination: copy_previous(destination, ("sources",))) as (_log, host):
        root = host.live_root
        assert root.context.require(SOURCES_V3) is not None
        try:
            root.context.require(SOURCES_V4)
        except CompositionError:
            pass
        else:
            raise AssertionError("旧 provider 启用了新 actor 所需的能力")

    names = ("sources", "conversation", "reply")
    async with application(path / "restore", replying=True,
                           extra_sources=lambda destination: copy_previous(destination, names)) as (log, host):
        root = host.live_root
        accepted = await root.context.require(CHANNEL_INPUT_V2)("test:room", "old-input", inbound("old"))
        async def finished():
            async for _ in log.catalog().follow():
                rows = log.reader("test:room").snapshot()
                if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                    return rows
        rows = await asyncio.wait_for(finished(), 10)
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
        destination = path / "restore/plugins"
        for name in names:
            shutil.rmtree(destination / name)
            shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / name, destination / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        try:
            await host.reconcile_changed()
        except RuntimeError as error:
            assert "PENDING" in str(error)
        else:
            raise AssertionError("跨能力组的逐插件更新意外启用")
        assert host.live_root is root
        assert log.reader("test:room").snapshot() == rows
        assert log.reader("test:room").get("old-input") == accepted
        for key in old_keys:
            assert root.context.require(key) is not None
    return {"old_actor_with_current_provider": "PENDING", "current_actor_with_old_provider": "missing v4",
            "complete_old_group": "Input/Output/ToolResult/Output",
            "breaking_partial_update": "rejected; old Root and exact messages restored"}


def inbound(text):
    return ChannelInboundMessage("test", "user", "room", text, datetime.now(UTC), {})


async def notifications(path):
    """实际安装 Reply 在 Input ACK 前取得占位，重放不重复通知或执行。"""
    async with application(path, replying=True) as (log, host):
        root = host.live_root
        entered, release = asyncio.Event(), asyncio.Event()
        holds, events = 0, []

        class CompletionTrace:
            @contextmanager
            def activity(self, _reader, _source):
                nonlocal holds
                holds += 1
                try:
                    yield
                finally:
                    holds -= 1

            @asynccontextmanager
            async def __call__(self, reader, source, *, child_permit=None):
                with self.activity(reader, source):
                    entered.set()
                    await release.wait()
                    yield

        await root.context.provide(REPLY_COMPLETION, CompletionTrace())
        await root.context.on(SOURCE_CHANGED_V3, lambda event: events.append((event.pending, holds)))
        try:
            message = inbound("work")
            accepted = await root.context.require(CHANNEL_INPUT_V2)("test:room", "input", message)
            assert holds >= 1 and events == [(True, 1)], events
            await asyncio.wait_for(entered.wait(), 5)
            before = [tuple(row) for row in log._connection.execute("SELECT * FROM messages")]
            replay = await root.context.require(CHANNEL_INPUT_V2)("test:room", "input", message)
            assert replay == accepted and events == [(True, 1)] and holds == 2
            assert before == [tuple(row) for row in log._connection.execute("SELECT * FROM messages")]
            assert root.context.require(ServiceKey("fixture.calls")) == []
            next_message = inbound("next")
            await root.context.require(CHANNEL_INPUT_V2)("test:room", "next-input", next_message)
            assert holds >= 1 and len(events) == 2 and events[-1][0] is True
            await root.context.require(CHANNEL_INPUT_V2)("test:room", "next-input", next_message)
            assert len(events) == 2 and holds >= 1
            return {"same_id_notifications": "one per created Input", "activity_at_input_ack": True,
                    "replay_rows_equal": True, "model_calls_before_release": 0,
                    "consumer": "actual installed Reply and Conversation", "paid_provider": "unrun"}
        finally:
            release.set()


async def main(path):
    return {"previous_source": str(args.previous_source.resolve()),
            "versions": await versions(path), "notifications": await notifications(path / "notifications")}


with tempfile.TemporaryDirectory(prefix="source-read-cohort-") as temporary:
    print(json.dumps(asyncio.run(main(Path(temporary))), ensure_ascii=False, indent=2))
