import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil

import pytest

from agent.config_models import Config
from agent.control.protocol.router import ConnectionRouter
from bootstrap import tools as bootstrap
from bootstrap.app_server import build_control_service
from core.net.http import SharedHttpResources
from session.message import Input


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
    core = bootstrap.build_core_runtime(Config(system_prompt=""), workspace, http)
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
async def test_connection_eof_releases_full_queue_and_blocked_writer():
    from infra.control.connection import NdjsonConnection
    from types import SimpleNamespace

    class BlockedWriter:
        def __init__(self):
            self.draining = asyncio.Event()
            self.closed = asyncio.Event()
            self.transport = self
        def write(self, payload):
            pass
        async def drain(self):
            self.draining.set()
            await asyncio.Event().wait()
        def abort(self):
            self.closed.set()
        def close(self):
            self.abort()
        async def wait_closed(self):
            await self.closed.wait()

    reader = asyncio.StreamReader()
    writer = BlockedWriter()
    connection = NdjsonConnection(reader, writer, SimpleNamespace(methods={}), max_message_bytes=1024,
                                  max_pending_requests=2, outbound_queue_size=1)
    task = asyncio.create_task(connection.run())
    try:
        await connection.send({"jsonrpc": "2.0", "id": 1, "result": "one"})
        await writer.draining.wait()
        await connection.send({"jsonrpc": "2.0", "id": 2, "result": "two"})
        reader.feed_eof()
        await asyncio.wait_for(task, 2)
        assert writer.closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
