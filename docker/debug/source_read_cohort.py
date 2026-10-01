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
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--previous-source", type=Path, required=True,
                    help="仍提供 sources.v4/source.session.v3 的完整旧源码")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT_V2, ChannelInboundMessage
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.control_frames import CONTROL_FRAMES
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts.reply import REPLY_COMPLETION
from agent.plugin_contracts.sources import SOURCES_V4, SOURCE_SESSION_V3, SOURCES_V5, SOURCE_CHANGED_V3
from plugins.programmatic.control import PROGRAMMATIC, AdmitParams, SendParams, PauseParams, ResumeParams, ResultParams
from plugins.programmatic.result import TURN_PROJECTION, read_result
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
    old_keys = (SOURCES_V4, SOURCE_SESSION_V3)
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
        assert root.context.require(SOURCES_V5) is not None

    async with application(path / "reverse", replying=True,
                           extra_sources=lambda destination: copy_previous(destination, ("sources",))) as (_log, host):
        root = host.live_root
        assert root.context.require(SOURCES_V4) is not None
        try:
            root.context.require(SOURCES_V5)
        except CompositionError:
            pass
        else:
            raise AssertionError("旧 provider 启用了新 actor 所需的能力")

    names = ("sources", "conversation", "reply", "react", "reply_program")
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
    return {"old_actor_with_current_provider": "PENDING", "current_actor_with_old_provider": "missing v5",
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


async def programmatic_routes(path):
    """真实 Programmatic 的旧读取不清理恢复通道，也不误判尚未提交的 Input。"""
    def add_programmatic(destination):
        shutil.copytree(Path(__file__).resolve().parents[2] / "plugins/programmatic",
                        destination / "programmatic", ignore=shutil.ignore_patterns("__pycache__"))

    results = []
    for mode in ("uncommitted", "settlement", "result"):
        entered, release, watcher_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_async = MessageReader.read_async
        operation = None
        waiter = None

        async def blocked_read(self, consume):
            if asyncio.current_task().get_name() == "plugin-task:programmatic-frame-settlement":
                await watcher_release.wait()
            value = await read_async(self, consume)
            if asyncio.current_task() is operation:
                entered.set()
                await release.wait()
            return value

        @asynccontextmanager
        async def release_reads():
            try:
                yield
            finally:
                release.set()
                watcher_release.set()
                if operation is not None:
                    await asyncio.gather(operation, return_exceptions=True)
                if waiter is not None:
                    if not waiter.done():
                        waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)

        # 1. 安装真实服务；只延迟后台读取，手动调用同一结算方法固定交错。
        MessageReader.read_async = blocked_read
        try:
            async with application(path / mode, replying=False,
                                   extra_sources=add_programmatic) as (log, host), release_reads():
                root = host.live_root
                service = root.context.require(PROGRAMMATIC)
                frames = root.context.require(CONTROL_FRAMES)
                projection = root.context.require(TURN_PROJECTION)
                session_id = "programmatic:" + mode
                await service.call("programmatic/session/admit", AdmitParams(session_id=session_id))
                reader = log.reader(session_id)

                async def settle():
                    async with service.ctx.runtime_scope():
                        await service.settle_changed(reader, "programmatic")

                send = SendParams(session_id=session_id, message_id="input", text="work")
                if mode == "uncommitted":
                    operation = asyncio.create_task(service.call("programmatic/message/send", send,
                                                    SimpleNamespace(connection_id="new")))
                    await asyncio.wait_for(entered.wait(), 5)
                    assert reader.get("input") is None
                    assert frames.active_input_ids(session_id) == ("input",)
                    await settle()
                    assert frames.active_input_ids(session_id) == ("input",)
                    release.set()
                    await operation
                    before = reader.snapshot()
                else:
                    await service.call("programmatic/message/send", send, SimpleNamespace(connection_id="old"))
                    await service.call("programmatic/message/pause", PauseParams(
                        session_id=session_id, message_id="pause"))
                    before = reader.snapshot()
                    assert read_result(reader, "input", projection)["status"] == "pause"
                    operation = asyncio.create_task(settle() if mode == "settlement" else service.call(
                        "programmatic/message/result", ResultParams(session_id=session_id, input_id="input")))
                    await asyncio.wait_for(entered.wait(), 5)
                    # 2. 真实 resume 提交 Control 并切换连接；旧读取仍停在屏障。
                    await service.call("programmatic/message/resume", ResumeParams(
                        session_id=session_id, message_id="resume", input_id="input"),
                        SimpleNamespace(connection_id="new"))
                    release.set()
                    result = await operation
                    if mode == "result":
                        assert result["status"] == "pause" and result["through_seq"] == before[-1].seq
                assert read_result(reader, "input", projection)["status"] == "open"
                assert frames.active_input_ids(session_id) == ("input",), mode
                await settle()
                assert frames.active_input_ids(session_id) == ("input",)

                # 3. 新通道绑定真实最终 Output，且等待受控 writer future 完成。
                output = log.writer(session_id, author="fixture", source="programmatic",
                                    body_types=(Output,), content={}).append("output", Output((), "complete"))
                waiter = asyncio.create_task(frames.wait_input(session_id, "input", output.message_id))
                ready = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(ready.set_result, None)
                await ready
                assert not waiter.done()
                tracked = frames.resolve_page("new", {"items": [{"id": output.message_id,
                    "session_id": session_id, "body": {"kind": "output", "finish": "complete", "parts": []}}]})
                assert len(tracked) == 1
                written = asyncio.get_running_loop().create_future()
                frames.attach_page(tracked, written)
                assert not waiter.done()
                written.set_result(None)
                await asyncio.wait_for(waiter, 5)
                assert frames.active_input_ids(session_id) == ()

                # 4. 稳定终态仍回收通道，原消息保持；没有请求真实模型或传输。
                frames.route_input(session_id, "input", "late", lambda: output.message_id)
                await settle()
                assert frames.active_input_ids(session_id) == ()
                assert reader.snapshot()[:len(before)] == before
                assert root.context.require(ServiceKey("fixture.calls")) == []
                assert log._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert not log._connection.execute("PRAGMA foreign_key_check").fetchall()
                results.append({"case": mode, "current_route": "preserved",
                                "final_output_frame": "writer future completed", "stable_cleanup": "passed"})
                watcher_release.set()
        finally:
            MessageReader.read_async = read_async
    return results


async def main(path):
    return {"previous_source": str(args.previous_source.resolve()),
            "versions": await versions(path), "notifications": await notifications(path / "notifications"),
            "programmatic_routes": await programmatic_routes(path / "programmatic")}


with tempfile.TemporaryDirectory(prefix="source-read-cohort-") as temporary:
    print(json.dumps(asyncio.run(main(Path(temporary))), ensure_ascii=False, indent=2))
