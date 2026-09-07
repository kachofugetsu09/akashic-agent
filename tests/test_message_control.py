import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil

import pytest

from agent.config_models import Config
from agent.control.service import ControlService
from agent.control.protocol.router import ConnectionRouter
from bootstrap import tools as bootstrap
from bootstrap.app_server import build_control_service
from core.net.http import SharedHttpResources
from session.message import Input
from session.artifacts import AttachmentRef
from session.log import MessageCatalog, MessageLog
from session.message import Message


@asynccontextmanager
async def runtime(tmp_path, monkeypatch, *, programmatic=False):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins/conversation", source / "conversation")
    shutil.copytree(Path(__file__).parents[1] / "plugins/sources", source / "sources")
    if programmatic:
        for name in ("programmatic", "turn_projection"):
            shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setattr(bootstrap, "_resolve_plugin_dirs", lambda _: [source])
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    service = None
    try:
        await core.start()
        service = build_control_service(core)
        yield core, service
    finally:
        if service is not None:
            await service.shutdown()
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


@pytest.mark.asyncio
async def test_control_v2_uses_real_message_input_and_cancellable_read_subscription(tmp_path, monkeypatch):
    async with runtime(tmp_path, monkeypatch) as (core, service):
        frames = []
        changed = asyncio.Event()
        async def send(frame):
            frames.append(frame)
            changed.set()
        router = ConnectionRouter(service, send)
        request_id = 0
        async def request(method, params):
            nonlocal request_id
            request_id += 1
            await router.handle_line(json.dumps({"jsonrpc": "2.0", "id": request_id,
                                                "method": method, "params": params}).encode())
            return next(frame for frame in frames if frame.get("id") == request_id)
        try:
            refused = await request("initialize", {"protocolVersion": "1.0", "clientInfo": {"name": "test", "version": "1"}})
            assert refused["error"]["data"]["supported"] == ["2.0"]
            await request("initialize", {"protocolVersion": "2.0", "clientInfo": {"name": "test", "version": "2"}})
            await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')
            assert (await request("thread/start", {}))["error"]["code"] == -32601
            session = (await request("session/create", {}))["result"]["session_id"]
            assert not core.message_log.catalog().snapshot_heads()
            await request("session/follow", {"session_id": session, "subscription_id": "watch-one"})
            params = {"session_id": session, "message_id": "original", "text": "完整原文"}
            first = await request("message/send", params)
            second = await request("message/send", params)
            assert first["result"] == second["result"]
            async with asyncio.timeout(5):
                while not any(frame.get("method") == "session/event" and
                              frame["params"]["event"]["type"] == "messages.appended" for frame in frames):
                    changed.clear()
                    await changed.wait()
            page = (await request("message/read", {"session_id": session}))["result"]
            assert [item["id"] for item in page["items"]] == ["original"]
        finally:
            await router.close()
        messages = core.message_log.reader(session).snapshot()
        assert len(messages) == 1 and isinstance(messages[0].body, Input)


@pytest.mark.asyncio
async def test_connection_eof_releases_full_queue_and_blocked_writer(tmp_path):
    from infra.control.connection import NdjsonConnection

    async def reject_accept(_session_id: str, _message_id: str, _incoming: object) -> Message:
        raise AssertionError("EOF 测试不应接纳控制输入")

    async def reject_reply_status(_session_id: str) -> AsyncGenerator[dict[str, object], None]:
        raise AssertionError("EOF 测试不应读取回复状态")
        yield {}

    def reject_attachments(_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]:
        raise AssertionError("EOF 测试不应读取附件")

    log = MessageLog(tmp_path / "sessions.db")
    service = ControlService(
        MessageCatalog(log),
        tmp_path,
        accept=reject_accept,
        reply_status=reject_reply_status,
        attachments=reject_attachments,
    )

    class BlockedTransport(asyncio.WriteTransport):
        def __init__(self, closed: asyncio.Event) -> None:
            self.closed = closed

        def abort(self) -> None:
            self.closed.set()

        def close(self) -> None:
            self.abort()

        def is_closing(self) -> bool:
            return self.closed.is_set()

    class BlockedWriter(asyncio.StreamWriter):
        def __init__(self, closed: asyncio.Event) -> None:
            self.draining = asyncio.Event()
            self._closed = closed
            super().__init__(
                BlockedTransport(closed),
                asyncio.Protocol(),
                None,
                asyncio.get_running_loop(),
            )

        def write(self, payload: bytes) -> None:
            pass
        async def drain(self):
            self.draining.set()
            await asyncio.Event().wait()
        async def wait_closed(self):
            await self._closed.wait()

    reader = asyncio.StreamReader()
    closed = asyncio.Event()
    writer = BlockedWriter(closed)
    connection = NdjsonConnection(reader, writer, service, max_message_bytes=1024,
                                  max_pending_requests=2, outbound_queue_size=1)
    task = asyncio.create_task(connection.run())
    try:
        await connection.send({"jsonrpc": "2.0", "id": 1, "result": "one"})
        await writer.draining.wait()
        await connection.send({"jsonrpc": "2.0", "id": 2, "result": "two"})
        reader.feed_eof()
        await asyncio.wait_for(task, 2)
        assert closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.shutdown()
        log.close()
