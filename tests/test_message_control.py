import ast
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

import pytest

from agent.config_models import Config
from agent.control.service import ControlService
from agent.control.protocol.router import ConnectionRouter
from agent.plugin_composition.channels import ChannelInboundMessage
from agent.plugin_composition import FiberState, ServiceKey
from agent.plugins.install import install_git_plugin
from agent.restart import RestartGate
from bootstrap import tools as bootstrap
from bootstrap.app_server import build_control_service
from core.net.http import SharedHttpResources
from session.message import ContentPart, Input, Message
from session.artifacts import AttachmentRef
from session.log import MessageCatalog, MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_install import _commit


def _write_plugin_source(path: Path, source: str) -> None:
    """Parse and compile a generated plugin before writing it to the fixture workspace."""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source, encoding="utf-8")


@asynccontextmanager
async def runtime(tmp_path, monkeypatch, *, programmatic=False):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # 新 workspace 经合法初始化入口建立空插件选择；load_all 不再容忍缺失 stable。
    from tests.fixtures.plugin_workspace import initialize_plugin_workspace
    initialize_plugin_workspace(workspace)
    source = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins/conversation", source / "conversation")
    shutil.copytree(Path(__file__).parents[1] / "plugins/sources", source / "sources")
    for name in ("commands", "ui", "content", "models"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name)
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


def _write_live_input_plugin(source: Path, *, version: str) -> None:
    """Write the real CHANNEL_INPUT owner used by the live Root test."""
    source.mkdir(parents=True, exist_ok=True)
    _write_plugin_source(source.joinpath("plugin.py"),
        "from agent.plugin_composition.channels import CHANNEL_INPUT\n"
        "from agent.plugin_composition.messages import MESSAGE_WRITERS, SESSION_ADMISSION\n"
        "from session.log import SessionAttributes\n"
        "from session.message import ContentPart, ContentReferences, Input\n"
        "api_version = 3\n"
        "name = 'target'\n"
        "version = '1.0.0'\n"
        f"VERSION = {version!r}\n"
        "ENTERED = None\n"
        "ENTERED_COUNT = 0\n"
        "RELEASE = None\n"
        "CLOSES = 0\n"
        "inject = (MESSAGE_WRITERS, SESSION_ADMISSION)\n"
        "def check_text(part):\n"
        "    if not isinstance(part.value, str):\n"
        "        raise ValueError('text must be a string')\n"
        "    return ContentReferences()\n"
        "async def apply(ctx):\n"
        "    writers = ctx.require(MESSAGE_WRITERS)\n"
        "    admissions = ctx.require(SESSION_ADMISSION)\n"
        "    open_input = writers.bind(\n"
        "        ctx, author='user', source='conversation', body_types=(Input,),\n"
        "        content={'text': check_text},\n"
        "    )\n"
        "    async def accept(session_id, message_id, incoming):\n"
        "        global ENTERED_COUNT\n"
        "        ENTERED_COUNT += 1\n"
        "        if ENTERED is not None:\n"
        "            ENTERED.set()\n"
        "        if RELEASE is not None:\n"
        "            await RELEASE.wait()\n"
        "        admissions.ensure(ctx, session_id, SessionAttributes())\n"
        "        body = Input((ContentPart('text', VERSION + ':' + incoming.content),))\n"
        "        return open_input(session_id).append(message_id, body)\n"
        "    await ctx.provide(CHANNEL_INPUT, accept)\n",
    )


def _write_live_peer_plugin(source: Path) -> None:
    """Write an unrelated live plugin whose Fiber identity must not change."""
    source.mkdir(parents=True, exist_ok=True)
    _write_plugin_source(source.joinpath("plugin.py"),
        "api_version = 3\n"
        "name = 'peer'\n"
        "version = '1.0.0'\n"
        "STARTS = 0\n"
        "CLOSES = 0\n"
        "async def apply(ctx):\n"
        "    global STARTS\n"
        "    STARTS += 1\n"
        "    async def close():\n"
        "        global CLOSES\n"
        "        CLOSES += 1\n"
        "    await ctx.effect(lambda: close)\n",
    )


def _write_hard_input_consumer(source: Path) -> None:
    """Write a real CHANNEL_INPUT dependent Fiber for the drain barrier."""
    source.mkdir(parents=True, exist_ok=True)
    _write_plugin_source(source.joinpath("plugin.py"),
        "from agent.plugin_composition.channels import CHANNEL_INPUT\n"
        "api_version = 3\n"
        "name = 'hard'\n"
        "version = '1.0.0'\n"
        "inject = (CHANNEL_INPUT,)\n"
        "HARD_CLOSED = None\n"
        "STARTS = 0\n"
        "CLOSES = 0\n"
        "async def apply(ctx):\n"
        "    global STARTS\n"
        "    STARTS += 1\n"
        "    ctx.require(CHANNEL_INPUT)\n"
        "    async def close():\n"
        "        global CLOSES\n"
        "        CLOSES += 1\n"
        "        if HARD_CLOSED is not None:\n"
        "            HARD_CLOSED.set()\n"
        "    await ctx.effect(lambda: close)\n",
    )


@pytest.mark.asyncio
async def test_message_send_uses_live_channel_owner_during_local_replace(
    tmp_path: Path, monkeypatch,
) -> None:
    """Accepted input drains on the old owner while new input is rejected."""
    workspace = tmp_path / "workspace"
    plugin_home = tmp_path / "plugin-home"
    target_source = tmp_path / "target-source"
    peer_root = tmp_path / "plugins"
    _write_live_input_plugin(target_source, version="old")
    _commit(target_source)
    _write_live_peer_plugin(peer_root / "peer")
    _write_hard_input_consumer(peer_root / "hard")
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    install_git_plugin(
        workspace=workspace, source=str(target_source), marketplace="lab",
        plugins_home=plugin_home,
    )

    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(
        Config(), workspace, http, plugin_dirs=[peer_root],
    )
    service = None
    router = None
    old_request = None
    operation = None
    manager = None
    release = asyncio.Event()
    peer_scope_entered = asyncio.Event()
    peer_scope_release = asyncio.Event()
    peer_scope_finished = asyncio.Event()
    peer_scope_task = None
    session_id = "akashic:live-input"
    try:
        await core.start()
        service = build_control_service(core)
        frames: list[dict[str, object]] = []

        async def send(frame: dict[str, object]) -> None:
            frames.append(frame)

        router = ConnectionRouter(service, send)

        async def request(identity: int, method: str, params: dict[str, object]):
            await router.handle_line(json.dumps({
                "jsonrpc": "2.0", "id": identity,
                "method": method, "params": params,
            }).encode())
            return next(frame for frame in reversed(frames) if frame.get("id") == identity)

        await request(1, "initialize", {
            "protocolVersion": "2.0", "clientInfo": {"name": "test", "version": "1"},
        })
        await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')

        manager = core.plugin_manager
        root = manager.live_root
        target = manager.generation("target@lab")
        peer = manager.generation("peer")
        hard = manager.generation("hard")
        assert root is not None and target is not None and target.fiber is not None
        assert peer is not None and peer.fiber is not None
        assert hard is not None and hard.fiber is not None
        peer_fiber = peer.fiber
        peer_context = peer.fiber.context
        peer_token = peer_context.fiber.activation_token
        peer_module = peer.instance.module
        peer_counts = (peer_module.STARTS, peer_module.CLOSES)
        hard_module = hard.instance.module
        hard_module.HARD_CLOSED = asyncio.Event()

        async def probe_peer_scope():
            async with peer_context.runtime_scope():
                peer_scope_entered.set()
                await peer_scope_release.wait()
                peer_scope_finished.set()

        async def reject_snapshot_entry(*args, **kwargs):
            raise AssertionError("live message/send path must not acquire snapshot lease")

        def reject_snapshot_compile(
            generations, *, snapshot_revision="", composition_root=None,
        ):
            raise AssertionError("live message/send path must not compile snapshot")

        monkeypatch.setattr(manager.snapshot_store, "acquire", reject_snapshot_entry)
        monkeypatch.setattr(manager, "_replace_formal_root", reject_snapshot_entry)
        monkeypatch.setattr(manager._snapshot_compiler, "compile", reject_snapshot_compile)

        old_module = target.instance.module
        old_module.ENTERED = asyncio.Event()
        old_module.RELEASE = release

        old_request = asyncio.create_task(request(2, "message/send", {
            "session_id": session_id, "message_id": "old", "text": "accepted",
        }))
        await old_module.ENTERED.wait()
        old_entered_count = old_module.ENTERED_COUNT

        _write_live_input_plugin(target_source, version="new")
        _commit(target_source)
        accepted = await manager.install(
            source=str(target_source), marketplace="lab", ref_name="", sparse_paths=[],
            update_id="target-input-v2",
        )
        assert accepted.state == "accepted"
        operation = manager._operation
        await hard_module.HARD_CLOSED.wait()
        assert operation is not None and target.fiber.state is FiberState.UNLOADING
        assert not old_request.done()
        assert len(target.fiber._fiber._in_flight_calls) == 1
        peer_scope_task = asyncio.create_task(probe_peer_scope())
        await peer_scope_entered.wait()
        assert len(peer_fiber._fiber._in_flight_calls) == 1
        assert peer_fiber.state is FiberState.ACTIVE
        assert peer_context.fiber.activation_token is peer_token

        unavailable = await request(3, "message/send", {
            "session_id": session_id, "message_id": "blocked", "text": "unavailable",
        })
        assert unavailable["error"]["code"] == -32603
        assert old_module.ENTERED_COUNT == old_entered_count
        assert manager.generation("peer") is peer
        assert peer.fiber is peer_fiber
        assert peer.fiber.state is FiberState.ACTIVE
        assert peer_context.fiber.activation_token is peer_token
        assert (peer_module.STARTS, peer_module.CLOSES) == peer_counts

        peer_scope_release.set()
        await peer_scope_task
        assert peer_scope_finished.is_set()
        assert not peer_fiber._fiber._in_flight_calls

        release.set()
        old_result = await old_request
        assert old_result["result"]["message_id"] == "old"
        operation_result = await asyncio.gather(operation.task, return_exceptions=True)
        assert len(operation_result) == 1 and operation_result[0].state == "active"
        assert not target.fiber._fiber._in_flight_calls
        assert old_module.CLOSES == 1

        new_result = await request(4, "message/send", {
            "session_id": session_id, "message_id": "new", "text": "available",
        })
        assert new_result["result"]["message_id"] == "new"
        messages = core.message_log.reader(session_id).snapshot()
        assert [message.message_id for message in messages] == ["old", "new"]
        assert [message.body.parts[0].value for message in messages] == [
            "old:accepted", "new:available",
        ]
        page = core.message_log.reader(session_id).read_page()
        manager_display = root.service_value(ServiceKey("core.message_display.v1"))
        assert callable(manager_display)
        manager_rows = await manager_display(page, display_only=True)
        control_page = await request(5, "message/read", {"session_id": session_id})
        assert control_page["result"]["items"] == manager_rows
        assert manager.live_root is root
    finally:
        peer_scope_release.set()
        if peer_scope_task is not None:
            await asyncio.gather(peer_scope_task, return_exceptions=True)
        release.set()
        if old_request is not None:
            await asyncio.gather(old_request, return_exceptions=True)
        if operation is None and manager is not None:
            operation = manager._operation
        if operation is not None:
            await asyncio.gather(operation.task, return_exceptions=True)
        if router is not None:
            await router.close()
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
async def test_restart_pending_is_retryable_and_same_input_is_admitted_after_abort(tmp_path):
    """重启等待期间拒绝输入并标为可重试，abort 后原 message id 可再次接纳。"""
    gate = RestartGate(boot_id="fixture-boot", supervised=True, commit=lambda _: None)
    log = MessageLog(tmp_path / "sessions.db")
    accepted: list[str] = []

    async def accept(session_id: str, message_id: str, incoming: object) -> Message:
        permit = gate.acquire()
        try:
            assert isinstance(incoming, ChannelInboundMessage)
            accepted.append(message_id)
            return Message(
                message_id, session_id, 0, datetime.now(UTC), "user", "control",
                Input((ContentPart("text", incoming.content),)),
            )
        finally:
            permit.release()

    async def reply_status(_session_id: str) -> AsyncGenerator[dict[str, object], None]:
        if False:
            yield {}

    service = ControlService(
        MessageCatalog(log), tmp_path, accept=accept,
        reply_status=reply_status, attachments=lambda _ids: (),
    )
    frames: list[dict[str, object]] = []

    async def send(frame: dict[str, object]) -> None:
        frames.append(frame)

    router = ConnectionRouter(service, send)
    request_id = 0

    async def request(method: str, params: dict[str, object]) -> dict[str, object]:
        nonlocal request_id
        request_id += 1
        await router.handle_line(json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
        }).encode())
        return next(frame for frame in frames if frame.get("id") == request_id)

    session = "akashic:restart-pending"
    try:
        await request("initialize", {
            "protocolVersion": "2.0", "clientInfo": {"name": "test", "version": "1"},
        })
        await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')

        gate.prepare("restart-1")
        refused = await request("message/send", {
            "session_id": session, "message_id": "same", "text": "retry me",
        })
        assert refused["error"] == {
            "code": -32001,
            "message": "runtime 正在等待重启，暂不接纳新 Root",
            "data": {"retryable": True},
        }
        assert accepted == []

        gate.abort("restart-1")
        admitted = await request("message/send", {
            "session_id": session, "message_id": "same", "text": "retry me",
        })
        assert admitted["result"] == {
            "version": 2, "session_id": session, "message_id": "same", "seq": 0,
        }
        assert accepted == ["same"]
    finally:
        await router.close()
        await service.shutdown()
        log.close()


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
