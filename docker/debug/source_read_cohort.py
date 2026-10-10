"""实际安装当前 Source，核对旧消费者拒绝和提交占位交错。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
import tempfile
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT_V2, ChannelInboundMessage
from agent.plugin_composition.model import FiberState
from plugins.gateway.contract import CONTROL_FRAMES

from agent.plugin_composition.messages import MessageReader
from plugins.reply.contract import (
    REPLY_COMPLETION,
)
from plugins.sources.contract import (
    SOURCES_V5,
    SOURCE_CHANGED_V3,
)
from plugins.programmatic.control import PROGRAMMATIC, AdmitParams, SendParams, PauseParams, ResumeParams, ResultParams
from plugins.programmatic.result import TURN_PROJECTION, read_result
from session.message import Input, Output, ToolResult
from tests.test_default_reply import application


@dataclass
class Transport:
    connection_id: str


async def versions(path):
    """当前组合明确拒绝旧来源能力，不隐式升级回调与权限。"""
    old_keys = (ServiceKey[object]("sources.v4"), ServiceKey[object]("source.session.v3"))
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

    return {"old_actor_with_current_provider": "PENDING", "current_provider": "sources.v5"}


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
            def activity(self, reader, source):
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
        for name in ("programmatic", "gateway"):
            shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / name,
                            destination / name, ignore=shutil.ignore_patterns("__pycache__"))

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
                                                    Transport(connection_id="new")))
                    await asyncio.wait_for(entered.wait(), 5)
                    assert reader.get("input") is None
                    assert frames.active_input_ids(session_id) == ("input",)
                    await settle()
                    assert frames.active_input_ids(session_id) == ("input",)
                    release.set()
                    await operation
                    before = reader.snapshot()
                else:
                    await service.call("programmatic/message/send", send, Transport(connection_id="old"))
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
                        Transport(connection_id="new"))
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
                written = asyncio.get_running_loop().create_future()
                assert frames.track_page("new", {"items": [{"id": output.message_id,
                    "session_id": session_id, "body": {"kind": "output", "finish": "complete", "parts": []}}]}, written)
                assert not waiter.done()
                written.set_result(None)
                await asyncio.wait_for(waiter, 5)
                assert frames.active_input_ids(session_id) == ()

                # 4. 稳定终态仍回收通道，原消息保持；没有请求真实模型或传输。
                frames.route_input_with_owner(session_id, "input", "late", lambda: output.message_id)
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
    return {"versions": await versions(path), "notifications": await notifications(path / "notifications"),
            "programmatic_routes": await programmatic_routes(path / "programmatic")}


with tempfile.TemporaryDirectory(prefix="source-read-cohort-") as temporary:
    print(json.dumps(asyncio.run(main(Path(temporary))), ensure_ascii=False, indent=2))
