"""用真实普通 Channel 组合检查客户端 follower 的局部生命周期。"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, closing, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from agent.plugin_composition import CompositionError, FiberState, PluginRuntime
from agent.plugin_composition.channels import (
    CHANNELS,
    CHANNEL_INPUT,
    ChannelCapability,
    ChannelDefinition,
    ChannelReady,
    InboundIdentity,
    StopReceipt,
)
from agent.plugin_composition.message_view import message_rows
from agent.plugin_composition.messages import MESSAGE_CATALOG
from plugins.akashic_clients import capabilities as client_capabilities
from plugins.akashic_clients.channel import (
    _GenerationAkashicAdapter,
    build_akashic_channel_factory,
)
from plugins.akashic_clients.capabilities import MESSAGE_DISPLAY, REPLY_STATUS
from plugins.akashic_clients.config import AkashicClientsConfig
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.akashic_clients.mobile_realtime.storage import DeviceRecord
from plugins.akashic_clients.mobile_realtime.protocol import GenericCommand
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input
from tests.test_akashic_clients_mobile_follow_scope import (
    _follow as _mobile_follow,
    _receive_appended_message as _mobile_receive_appended,
    append as _mobile_append,
    connected as _mobile_connected,
    gateway as mobile_gateway,
)
from tests.test_channel_provider import provider_root


class _ProbeAdapter:
    """Keep the provider root's unrelated binding alive during client unload."""

    def __init__(self, context: Any) -> None:
        self._context = context

    async def start(self) -> Any:
        return ChannelReady(self._context.binding_token)

    async def deliver(self, _request: Any) -> Any:
        raise AssertionError("生命周期测试不应投递 probe 消息")

    async def stop(self) -> Any:
        return StopReceipt(self._context.binding_token, resources_closed=True)


class _Socket:
    """Provide the Web send surface required by the real follower owner."""

    def __init__(self, *, gate_following: bool = False) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.frames: list[dict[str, object]] = []
        self.gate_following = gate_following
        self.following_gate_entered = asyncio.Event()
        self.release_following = asyncio.Event()

    async def send_json(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)
        if self.gate_following and frame.get("type") == "session.following":
            self.following_gate_entered.set()
            await self.release_following.wait()
            self.gate_following = False

    async def close(self, **_: object) -> None:
        self.application_state = WebSocketState.DISCONNECTED


class _IdleReplyReader:
    """Hold a real reply iterator until the owner lifecycle cancels it."""

    def __init__(self, *, expect_provider_unloading: bool) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.provider_closed = asyncio.Event()
        self.cleanup_gate = asyncio.Event()
        self.follow_count = 0
        self.second_follow_started = asyncio.Event()
        self.expect_provider_unloading = expect_provider_unloading
        self.provider_state: Callable[[], FiberState] = lambda: FiberState.ACTIVE
        if not expect_provider_unloading:
            self.cleanup_gate.set()

    async def follow(self, _session_id: str) -> AsyncGenerator[dict[str, object], None]:
        self.follow_count += 1
        if self.follow_count == 2:
            self.second_follow_started.set()
        try:
            self.started.set()
            await self.release.wait()
            if False:
                yield {}
        finally:
            if self.expect_provider_unloading:
                assert self.provider_state() is FiberState.UNLOADING
            self.closed.set()
            await self.cleanup_gate.wait()


def _client_config(tmp_path: Path, *, mobile: bool) -> dict[str, object]:
    """Build a future-test config while keeping all network endpoints local."""

    return {
        "web": {
            "enabled": not mobile,
            "socket_path": str(tmp_path / "chat.sock"),
        },
        "mobile_realtime": {
            "enabled": mobile,
            "host": "127.0.0.1",
            "port": 6323,
            "database": "mobile.db",
            "key_encryption": {
                "provider": "file",
                "master_key_file": "mobile/master.json",
                "keyset_manifest": "mobile/keys/current.json",
            },
        },
    }


@asynccontextmanager
async def _real_client_root(
    tmp_path: Path,
    log: MessageLog,
    reader: _IdleReplyReader,
    *,
    mobile: bool,
    display_seen: asyncio.Event,
    monkeypatch: pytest.MonkeyPatch | None = None,
    admission_opened: asyncio.Event | None = None,
    display_probe: dict[str, Any] | None = None,
    display_behavior: str | None = None,
    display_started: asyncio.Event | None = None,
    display_error: asyncio.Event | None = None,
    display_error_release: asyncio.Event | None = None,
    display_exception: RuntimeError | None = None,
    display_cancelled: asyncio.Event | None = None,
) -> AsyncGenerator[tuple[Any, Any, Any, Any, Any, Any], None]:
    """Mount the actual client factory through Root, Channels and Effects."""

    config_data = _client_config(tmp_path, mobile=mobile)
    config = AkashicClientsConfig.model_validate(config_data)
    factory = build_akashic_channel_factory(config, tmp_path)

    if monkeypatch is not None and admission_opened is not None:
        original_open_admission = _GenerationAkashicAdapter.open_admission

        def observe_open_admission(adapter: _GenerationAkashicAdapter) -> None:
            original_open_admission(adapter)
            admission_opened.set()

        monkeypatch.setattr(
            _GenerationAkashicAdapter,
            "open_admission",
            observe_open_admission,
        )

    async with provider_root(tmp_path, _ProbeAdapter) as (root, channels):
        for key in client_capabilities.CLIENT_CAPABILITIES:
            if key in {MESSAGE_CATALOG, MESSAGE_DISPLAY, REPLY_STATUS}:
                continue
            await root.context.provide(key, object())
        await root.context.provide(
            MESSAGE_CATALOG,
            cast(Any, log.catalog()),
        )

        async def display(page: Any, *, display_only: bool) -> list[dict[str, object]]:
            if display_probe is not None:
                fiber = display_probe.get("fiber")
                assert fiber is not None
                assert fiber._call_owned_by_current_task() is not None
                current_task = asyncio.current_task()
                assert current_task is not None
                parent_task = display_probe.get("parent_task")
                if parent_task is not None:
                    assert current_task is not parent_task
                display_probe["task"] = current_task
                display_probe["permit"] = True
            if display_started is not None:
                display_started.set()
            try:
                if display_behavior == "error":
                    if display_error_release is not None:
                        await display_error_release.wait()
                    if display_error is not None:
                        display_error.set()
                    raise (
                        RuntimeError("display callback failed")
                        if display_exception is None
                        else display_exception
                    )
                if display_behavior == "cancel":
                    if display_probe is None:
                        raise AssertionError("display cancel requires a display probe")
                    gate = display_probe.get("cancel_gate")
                    if not isinstance(gate, asyncio.Event):
                        raise AssertionError("display cancel requires an Event gate")
                    await gate.wait()
                display_seen.set()
                return message_rows(page, display_only=display_only)
            except asyncio.CancelledError:
                if display_cancelled is not None:
                    display_cancelled.set()
                raise

        await root.context.provide(MESSAGE_DISPLAY, display)

        async def reply_provider(ctx: Any) -> None:
            await ctx.provide(REPLY_STATUS, reader)
            await ctx.effect(lambda: reader.provider_closed.set, label="reply-owner-close")

        reply_fiber = await root.mount(
            reply_provider,
            name="lifecycle-reply-owner",
            runtime=PluginRuntime(
                "lifecycle-reply-owner",
                "lifecycle-reply-generation",
                tmp_path,
                tmp_path,
                tmp_path,
                {},
            ),
        )

        async def client_contribution(ctx: Any) -> None:
            await ctx.require(CHANNELS).register(
                ctx,
                ChannelDefinition(
                    "akashic",
                    frozenset(
                        {
                            ChannelCapability.INBOUND,
                            ChannelCapability.DURABLE_INBOUND,
                            ChannelCapability.OUTBOUND,
                            ChannelCapability.TURN_STREAM,
                        }
                    ),
                    factory,
                    InboundIdentity.PROVIDER_MESSAGE_ID,
                ),
            )

        client_fiber = await root.mount(
            client_contribution,
            name="lifecycle-client",
            inject=(CHANNELS, CHANNEL_INPUT, *client_capabilities.CLIENT_CAPABILITIES),
            runtime=PluginRuntime(
                "lifecycle-client",
                "lifecycle-client-generation",
                tmp_path,
                tmp_path,
                tmp_path,
                config_data,
            ),
        )
        state = next(
            state for state in channels._bindings.values() if state.channel_name == "akashic"
        )
        assert isinstance(state.adapter, _GenerationAkashicAdapter)
        assert state.factory_context is not None
        assert state.factory_context.open_scope is not None
        assert client_fiber.state is FiberState.ACTIVE
        if admission_opened is not None:
            await admission_opened.wait()
        reader.provider_state = lambda: reply_fiber.state
        if display_probe is not None:
            display_probe["fiber"] = client_fiber
        try:
            yield root, channels, client_fiber, reply_fiber, state, state.adapter
        finally:
            reader.release.set()
            reader.cleanup_gate.set()


def _mobile_host_app(
    tmp_path: Path,
    log: MessageLog,
    reader: _IdleReplyReader,
    display_seen: asyncio.Event,
    holder: dict[str, Any],
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    admission_opened: asyncio.Event | None = None,
    display_probe: dict[str, Any] | None = None,
) -> FastAPI:
    """Serve the real Mobile runtime while Root owns the adapter lifecycle."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        async with _real_client_root(
            tmp_path,
            log,
            reader,
            mobile=True,
            display_seen=display_seen,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
            display_probe=display_probe,
        ) as bundle:
            runtime = cast(Any, bundle[-1]._mobile_runtime)
            assert runtime is not None
            private = ec.generate_private_key(ec.SECP256R1())
            public = base64.b64encode(
                private.public_key().public_bytes(
                    serialization.Encoding.DER,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            ).decode()
            device = uuid4().hex
            runtime.storage.register_device(
                DeviceRecord(
                    device,
                    public,
                    "lifecycle",
                    datetime.now(timezone.utc),
                    None,
                    (),
                )
            )
            runtime.start()
            holder.update(runtime=runtime, device=device, private=private, bundle=bundle)
            try:
                yield
            finally:
                await runtime.stop()
                holder.clear()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def mobile_websocket(websocket: WebSocket) -> None:
        send_text = websocket.send_text

        async def gated_send_text(text: str) -> None:
            await send_text(text)
            frame = json.loads(text)
            holder.setdefault("wire_types", []).append(frame.get("type"))
            if not holder.get("gate_next_follow"):
                return
            if frame.get("type") != "session.follow.ok":
                return
            holder["follow_ok_gate_entered"].set()
            await holder["release_follow_ok"].wait()
            holder["gate_next_follow"] = False

        websocket.send_text = gated_send_text  # type: ignore[method-assign]
        await holder["runtime"].handle_websocket(websocket)

    return app


async def _dispose_with_gate(fiber: Any, release: asyncio.Event) -> None:
    """Start real owner disposal, release a held wire boundary, then join it."""

    started = asyncio.Event()

    async def dispose() -> None:
        started.set()
        await fiber.dispose()

    task = asyncio.create_task(dispose())
    try:
        await started.wait()
    finally:
        release.set()
    await task


async def _start_dispose(fiber: Any) -> asyncio.Task[None]:
    """Start real disposal and return its still-running task to the caller."""

    return asyncio.create_task(fiber.dispose())


async def _await_task(task: asyncio.Task[Any]) -> None:
    """Join a task created on the real client event loop."""

    await task


def _assert_display_error_group(error: BaseException, sentinel: RuntimeError) -> None:
    """Check the known follower TaskGroup error shape and exact sentinel."""

    assert type(error) is ExceptionGroup
    assert len(error.exceptions) == 1
    assert error.exceptions[0] is sentinel
    assert type(error.exceptions[0]) is RuntimeError
    assert str(error.exceptions[0]) == str(sentinel)


def _assert_display_driver_error(
    error: BaseException,
    sentinel: RuntimeError,
) -> None:
    """Check the driver TaskGroup wrapping the follower TaskGroup once."""

    assert type(error) is ExceptionGroup
    assert len(error.exceptions) == 1
    _assert_display_error_group(error.exceptions[0], sentinel)


def _install_mobile_close_before_child(
    adapter: _GenerationAkashicAdapter,
    holder: dict[str, Any],
) -> None:
    """Schedule real admission close before the registered child gets its first step."""

    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    holder["task_factory_loop"] = loop
    holder["task_factory_previous"] = previous
    holder["close_scheduled"] = False

    def factory(loop: asyncio.AbstractEventLoop, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        code = getattr(coro, "cr_code", None)
        if (
            not holder["close_scheduled"]
            and code is not None
            and code.co_name == "_follow_message_session_scoped"
        ):
            holder["close_scheduled"] = True

            def close() -> None:
                adapter.close_admission()
                holder["admission_closed"].set()

            loop.call_soon(close)
        if previous is not None:
            return previous(loop, coro, **kwargs)
        return asyncio.Task(coro, loop=loop, **kwargs)

    loop.set_task_factory(factory)


def _restore_mobile_task_factory(holder: dict[str, Any]) -> None:
    """Restore the loop factory after the real WebSocket handler returns."""

    loop = holder["task_factory_loop"]
    loop.set_task_factory(holder["task_factory_previous"])


async def _run_mobile_ready_send_lock_case(
    runtime: Any,
    device: str,
    adapter: _GenerationAkashicAdapter,
    session_id: str,
) -> None:
    """Close a real registered follow while its parent waits for the wire lock."""

    connection = runtime._connections[device]
    ready_observed = asyncio.Event()
    original_follow = runtime._follow_message_session_scoped

    async def observe_ready(
        request: Any,
        child_device: str,
        child_connection: Any,
        ready: asyncio.Future[Any],
        start_follow: asyncio.Event,
    ) -> None:
        ready.add_done_callback(lambda _done: ready_observed.set())
        await original_follow(
            request,
            child_device,
            child_connection,
            ready,
            start_follow,
        )

    runtime._follow_message_session_scoped = observe_ready
    frame = GenericCommand.model_validate(
        {
            "v": 1,
            "kind": "command",
            "type": "session.follow",
            "id": "01ARZ3NDEKTSV4RRFFQ69G5FB1",
            "connection_epoch": connection.connection_epoch,
            "session_id": session_id,
            "payload": {"message_log_version": 2, "after_seq": -1},
        }
    )
    await connection.send_lock.acquire()
    try:
        async with asyncio.TaskGroup() as tasks:
            parent = asyncio.create_task(
                runtime._start_message_follow(frame, device, connection, tasks)
            )
            try:
                await ready_observed.wait()
                assert not parent.done()
                adapter.close_admission()
                connection.send_lock.release()
                await parent
            finally:
                if connection.send_lock.locked():
                    connection.send_lock.release()
                if not parent.done():
                    parent.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await parent
    finally:
        runtime._follow_message_session_scoped = original_follow


@pytest.mark.parametrize("trigger", ("client", "provider"))
def test_real_mobile_ready_window_cancels_registered_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
) -> None:
    """Close admission in the real gap after ready and before child start."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    display_seen = asyncio.Event()
    reader = _IdleReplyReader(expect_provider_unloading=trigger == "provider")
    holder: dict[str, Any] = {
        "gate_next_follow": True,
        "follow_ok_gate_entered": asyncio.Event(),
        "release_follow_ok": asyncio.Event(),
    }
    session_id = "akashic:00000000-0000-4000-8000-000000000001"
    admission_opened = asyncio.Event()

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        app = _mobile_host_app(
            tmp_path,
            log,
            reader,
            display_seen,
            holder,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
        )
        with TestClient(app) as client:
            runtime = holder["runtime"]
            device = holder["device"]
            private = holder["private"]
            _root, _channels, client_fiber, reply_fiber, state, adapter = holder["bundle"]
            assert isinstance(adapter, _GenerationAkashicAdapter)
            gateway = (log, runtime, client, device, private, [])
            with _mobile_connected(gateway) as (websocket, epoch):
                _mobile_follow(
                    websocket,
                    epoch,
                    session_id,
                    "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                )
                assert client.portal.call(
                    holder["follow_ok_gate_entered"].is_set
                )
                assert client.portal.call(lambda: bool(runtime._message_followers))
                assert not display_seen.is_set()

                client.portal.call(adapter.close_admission)
                if trigger == "client":
                    client.portal.call(
                        _dispose_with_gate,
                        client_fiber,
                        holder["release_follow_ok"],
                    )
                    assert client.portal.call(lambda: reply_fiber.state is FiberState.ACTIVE)
                    client.portal.call(reply_fiber.dispose)
                else:
                    client.portal.call(
                        _dispose_with_gate,
                        reply_fiber,
                        holder["release_follow_ok"],
                    )
                    assert client.portal.call(lambda: client_fiber.state is FiberState.PENDING)

                client.portal.call(reader.provider_closed.wait)
                assert client.portal.call(lambda: not runtime._message_followers)
                assert client.portal.call(lambda: reader.follow_count == 0)
                assert client.portal.call(lambda: not reader.started.is_set())
                assert "session.follow.ok" in holder.get("wire_types", [])
                assert not state.listeners
                assert state.in_flight == 0
                assert state.adapter_stop_succeeded


def test_real_mobile_first_instruction_close_settles_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered child cancelled before its first step cannot strand ready."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    display_seen = asyncio.Event()
    admission_opened = asyncio.Event()
    reader = _IdleReplyReader(expect_provider_unloading=False)
    holder: dict[str, Any] = {"admission_closed": asyncio.Event()}
    session_id = "akashic:00000000-0000-4000-8000-000000000005"

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        app = _mobile_host_app(
            tmp_path,
            log,
            reader,
            display_seen,
            holder,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
        )
        with TestClient(app) as client:
            runtime = holder["runtime"]
            device = holder["device"]
            private = holder["private"]
            _root, _channels, _client_fiber, _reply_fiber, state, adapter = holder["bundle"]
            assert isinstance(adapter, _GenerationAkashicAdapter)
            gateway = (log, runtime, client, device, private, [])
            with _mobile_connected(gateway) as (websocket, epoch):
                client.portal.call(
                    _install_mobile_close_before_child,
                    adapter,
                    holder,
                )
                websocket.send_json(
                    {
                        "v": 1,
                        "kind": "command",
                        "type": "session.follow",
                        "id": "01ARZ3NDEKTSV4RRFFQ69G5FB0",
                        "connection_epoch": epoch,
                        "session_id": session_id,
                        "payload": {"message_log_version": 2, "after_seq": -1},
                    }
                )
                client.portal.call(holder["admission_closed"].wait)
                try:
                    with pytest.raises(WebSocketDisconnect):
                        websocket.receive_json()
                finally:
                    client.portal.call(_restore_mobile_task_factory, holder)
                assert client.portal.call(lambda: not runtime._message_followers)
                assert client.portal.call(lambda: reader.follow_count == 0)
                assert client.portal.call(lambda: not reader.started.is_set())
                assert not display_seen.is_set()
                assert not state.listeners
                assert state.in_flight == 0


def test_real_mobile_ready_send_lock_close_rejects_new_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close while the real parent waits on send_lock cannot start the reader."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    display_seen = asyncio.Event()
    admission_opened = asyncio.Event()
    reader = _IdleReplyReader(expect_provider_unloading=False)
    holder: dict[str, Any] = {}
    session_id = "akashic:00000000-0000-4000-8000-000000000007"

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        app = _mobile_host_app(
            tmp_path,
            log,
            reader,
            display_seen,
            holder,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
        )
        with TestClient(app) as client:
            runtime = holder["runtime"]
            device = holder["device"]
            private = holder["private"]
            _root, _channels, _client_fiber, _reply_fiber, state, adapter = holder["bundle"]
            assert isinstance(adapter, _GenerationAkashicAdapter)
            gateway = (log, runtime, client, device, private, [])
            with _mobile_connected(gateway):
                client.portal.call(
                    _run_mobile_ready_send_lock_case,
                    runtime,
                    device,
                    adapter,
                    session_id,
                )
            assert not display_seen.is_set()
            assert "session.follow.ok" not in holder.get("wire_types", [])
            assert not runtime._message_followers
            assert not reader.started.is_set()
            assert not state.listeners
            assert state.in_flight == 0


@pytest.mark.parametrize("trigger", ("client", "provider"))
def test_real_mobile_owner_cleanup_releases_idle_followers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
) -> None:
    """The registered Mobile child follows the real ready/stop ordering."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    session_id = "akashic:00000000-0000-4000-8000-000000000002"
    display_seen = asyncio.Event()
    admission_opened = asyncio.Event()
    display_probe: dict[str, Any] = {}
    reader = _IdleReplyReader(expect_provider_unloading=trigger == "provider")
    holder: dict[str, Any] = {}

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        app = _mobile_host_app(
            tmp_path,
            log,
            reader,
            display_seen,
            holder,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
            display_probe=display_probe,
        )
        with TestClient(app) as client:
            runtime = holder["runtime"]
            device = holder["device"]
            private = holder["private"]
            _root, channels, client_fiber, reply_fiber, state, adapter = holder["bundle"]
            assert isinstance(adapter, _GenerationAkashicAdapter)
            assert adapter._mobile is not None
            gateway = (log, runtime, client, device, private, [])
            with _mobile_connected(gateway) as (websocket, epoch):
                _mobile_follow(
                    websocket,
                    epoch,
                    session_id,
                    "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                )
                assert [
                    item["id"]
                    for item in _mobile_receive_appended(websocket)["items"]
                ] == ["first"]
                client.portal.call(reader.started.wait)
                assert client.portal.call(lambda: bool(runtime._message_followers))
                assert client.portal.call(lambda: not client_fiber._in_flight_calls)
                assert display_probe["permit"] is True
                assert state.in_flight == 0

                peer_state = next(
                    item for item in channels._bindings.values()
                    if item.channel_name == "probe"
                )
                assert peer_state.plugin_context is not None
                peer_context = peer_state.plugin_context
                peer_token = peer_context.fiber.activation_token
                peer_state_before = peer_context.fiber.state

                if trigger == "client":
                    client.portal.call(client_fiber.dispose)
                    assert client.portal.call(lambda: reply_fiber.state is FiberState.ACTIVE)
                    client.portal.call(reply_fiber.dispose)
                else:
                    provider_dispose = client.portal.call(
                        _start_dispose,
                        reply_fiber,
                    )

                client.portal.call(reader.closed.wait)
                if trigger == "provider":
                    assert client.portal.call(lambda: client_fiber.state is FiberState.PENDING)
                    client.portal.call(
                        _assert_peer_scope,
                        peer_context,
                        peer_token,
                    )
                    client.portal.call(reader.cleanup_gate.set)
                    client.portal.call(_await_task, provider_dispose)
                client.portal.call(reader.provider_closed.wait)
                assert client.portal.call(lambda: not runtime._message_followers)
                assert not state.listeners
                assert state.in_flight == 0
                assert state.adapter_stop_succeeded
                assert not log._listeners

                async def require_client_unavailable() -> None:
                    with pytest.raises(CompositionError) as unavailable:
                        async with client_fiber.context.runtime_scope():
                            raise AssertionError("inactive client accepted a new request")
                    assert unavailable.value.code == "OWNER_UNAVAILABLE"

                client.portal.call(require_client_unavailable)
                assert peer_context.fiber.state is peer_state_before is FiberState.ACTIVE
                assert peer_context.fiber.activation_token is peer_token
                client.portal.call(
                    _assert_peer_scope,
                    peer_context,
                    peer_token,
                )


async def _assert_peer_scope(peer_context: Any, peer_token: object) -> None:
    """Keep the unrelated peer usable while the client binding is gone."""

    async with peer_context.runtime_scope():
        assert peer_context.fiber.activation_token is peer_token


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ("client", "provider"))
async def test_real_web_close_during_follow_registration_rejects_late_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
) -> None:
    """A real close during following send cannot register a new Web follower."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    close_observed = asyncio.Event()
    original_close = WebChatChannel._close_v3_binding

    def observe_close(channel: WebChatChannel, adapter: Any) -> None:
        original_close(channel, adapter)
        close_observed.set()

    monkeypatch.setattr(WebChatChannel, "_close_v3_binding", observe_close)
    session_id = "akashic:00000000-0000-4000-8000-000000000006"
    display_seen = asyncio.Event()
    admission_opened = asyncio.Event()
    reader = _IdleReplyReader(expect_provider_unloading=trigger == "provider")

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        async with _real_client_root(
            tmp_path,
            log,
            reader,
            mobile=False,
            display_seen=display_seen,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
        ) as (_root, _channels, client_fiber, reply_fiber, state, adapter):
            web = cast(WebChatChannel, adapter._web)
            assert web is not None
            followers = asyncio.TaskGroup()
            await followers.__aenter__()
            old_socket = cast(WebSocket, _Socket())
            race_socket_value = _Socket(gate_following=True)
            race_socket = cast(WebSocket, race_socket_value)
            race_task: asyncio.Task[Any] | None = None
            dispose_task: asyncio.Task[Any] | None = None
            try:
                await web._follow_session(
                    old_socket,
                    "request-old",
                    {"version": 2, "session_id": session_id, "after_seq": -1},
                    followers,
                )
                await display_seen.wait()
                await reader.started.wait()
                assert reader.follow_count == 1
                race_task = asyncio.create_task(
                    web._follow_session(
                        race_socket,
                        "request-race",
                        {"version": 2, "session_id": session_id, "after_seq": -1},
                        followers,
                    )
                )
                await race_socket_value.following_gate_entered.wait()

                owner = client_fiber if trigger == "client" else reply_fiber
                dispose_task = asyncio.create_task(owner.dispose())
                await close_observed.wait()
                assert web._stopping is True
                race_socket_value.release_following.set()
                assert await race_task == ""
                await reader.closed.wait()
                if trigger == "provider":
                    reader.cleanup_gate.set()
                await dispose_task
                if trigger == "client":
                    assert reply_fiber.state is FiberState.ACTIVE
                    await reply_fiber.dispose()
                else:
                    assert client_fiber.state is FiberState.PENDING
                await reader.provider_closed.wait()
                assert reader.follow_count == 1
                assert not reader.second_follow_started.is_set()
                assert race_socket not in web._followers
                assert not web._followers
                assert not state.listeners
                assert state.in_flight == 0
                assert state.adapter_stop_succeeded
            finally:
                race_socket_value.release_following.set()
                reader.release.set()
                reader.cleanup_gate.set()
                await web._cancel_follow(old_socket)
                await web._cancel_follow(race_socket)
                if race_task is not None:
                    with suppress(asyncio.CancelledError):
                        await race_task
                if dispose_task is not None:
                    with suppress(asyncio.CancelledError):
                        await dispose_task
                await followers.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ("client", "provider"))
async def test_real_web_owner_cleanup_releases_idle_followers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
) -> None:
    """Real adapter stop closes admission, joins followers, then releases owners."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    session_id = "akashic:00000000-0000-4000-8000-000000000003"
    display_seen = asyncio.Event()
    admission_opened = asyncio.Event()
    display_probe: dict[str, Any] = {}
    reader = _IdleReplyReader(expect_provider_unloading=trigger == "provider")

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        async with _real_client_root(
            tmp_path,
            log,
            reader,
            mobile=False,
            display_seen=display_seen,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
            display_probe=display_probe,
        ) as (_root, channels, client_fiber, reply_fiber, state, adapter):
            web = cast(WebChatChannel, adapter._web)
            assert web is not None
            socket = cast(WebSocket, _Socket())
            followers = asyncio.TaskGroup()
            await followers.__aenter__()
            provider_dispose: asyncio.Task[Any] | None = None
            try:
                await web._follow_session(
                    socket,
                    "request-1",
                    {"version": 2, "session_id": session_id, "after_seq": -1},
                    followers,
                )
                await display_seen.wait()
                await reader.started.wait()
                assert web._followers
                assert display_probe["permit"] is True
                assert not client_fiber._in_flight_calls
                assert state.in_flight == 0

                peer_state = next(
                    item for item in channels._bindings.values()
                    if item.channel_name == "probe"
                )
                assert peer_state.plugin_context is not None
                peer_context = peer_state.plugin_context
                peer_token = peer_context.fiber.activation_token
                peer_state_before = peer_context.fiber.state

                if trigger == "client":
                    await client_fiber.dispose()
                    assert reply_fiber.state is FiberState.ACTIVE
                    await reply_fiber.dispose()
                else:
                    provider_dispose = asyncio.create_task(reply_fiber.dispose())
                    await reader.closed.wait()
                    async with peer_context.runtime_scope():
                        assert peer_context.fiber.activation_token is peer_token
                    reader.cleanup_gate.set()
                    await provider_dispose
                    assert client_fiber.state is FiberState.PENDING

                await reader.closed.wait()
                await reader.provider_closed.wait()
                assert not web._followers
                assert not state.listeners
                assert state.in_flight == 0
                assert state.adapter_stop_succeeded
                assert not log._listeners
                with pytest.raises(CompositionError) as unavailable:
                    async with client_fiber.context.runtime_scope():
                        raise AssertionError("inactive client accepted a new request")
                assert unavailable.value.code == "OWNER_UNAVAILABLE"

                assert peer_context.fiber.state is peer_state_before is FiberState.ACTIVE
                assert peer_context.fiber.activation_token is peer_token
                async with peer_context.runtime_scope():
                    assert peer_context.fiber.activation_token is peer_token
            finally:
                reader.release.set()
                reader.cleanup_gate.set()
                await web._cancel_follow(socket)
                if provider_dispose is not None:
                    with suppress(asyncio.CancelledError):
                        await provider_dispose
                await followers.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("display_behavior", ("error", "cancel"))
async def test_real_web_display_callback_failure_settles_scope_and_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    display_behavior: str,
) -> None:
    """A real page display failure or cancellation releases its client permit."""

    async def skip_os_server(
        _adapter: _GenerationAkashicAdapter,
        _server: Any,
        *,
        name: str,
    ) -> None:
        _ = name

    monkeypatch.setattr(_GenerationAkashicAdapter, "_start_server", skip_os_server)
    session_id = "akashic:00000000-0000-4000-8000-000000000008"
    display_seen = asyncio.Event()
    admission_opened = asyncio.Event()
    display_started = asyncio.Event()
    display_error = asyncio.Event()
    display_error_release = asyncio.Event()
    display_cancelled = asyncio.Event()
    display_gate = asyncio.Event()
    sentinel = RuntimeError("display callback sentinel")
    display_probe: dict[str, Any] = {
        "parent_task": asyncio.current_task(),
        "cancel_gate": display_gate,
    }
    reader = _IdleReplyReader(expect_provider_unloading=False)

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        writer = log.writer(
            session_id,
            author="test",
            source="conversation",
            body_types=(Input,),
            content={"text": lambda _part: ContentReferences()},
        )
        writer.append("first", Input((ContentPart("text", "hello"),)))
        async with _real_client_root(
            tmp_path,
            log,
            reader,
            mobile=False,
            display_seen=display_seen,
            monkeypatch=monkeypatch,
            admission_opened=admission_opened,
            display_probe=display_probe,
            display_behavior=display_behavior,
            display_started=display_started,
            display_error=display_error,
            display_error_release=display_error_release,
            display_exception=sentinel,
            display_cancelled=display_cancelled,
        ) as (_root, _channels, client_fiber, reply_fiber, state, adapter):
            web = cast(WebChatChannel, adapter._web)
            assert web is not None
            socket = cast(WebSocket, _Socket())
            driver_started = asyncio.Event()
            driver_release = asyncio.Event()
            driver_result_observed = False

            async def drive_follow() -> None:
                async with asyncio.TaskGroup() as followers:
                    await web._follow_session(
                        socket,
                        "request-display-failure",
                        {"version": 2, "session_id": session_id, "after_seq": -1},
                        followers,
                    )
                    driver_started.set()
                    await driver_release.wait()

            driver_task = asyncio.create_task(drive_follow())
            dispose_task: asyncio.Task[Any] | None = None
            try:
                await driver_started.wait()
                await display_started.wait()
                await reader.started.wait()
                assert web._followers
                assert display_probe["permit"] is True
                assert display_probe["task"] is not display_probe["parent_task"]

                if display_behavior == "error":
                    display_error_release.set()
                    await display_error.wait()
                    with pytest.raises(BaseException) as driver_error:
                        await driver_task
                    _assert_display_driver_error(driver_error.value, sentinel)
                    driver_result_observed = True
                    with pytest.raises(BaseException) as cleanup_error:
                        await web._cancel_follow(socket)
                    _assert_display_error_group(cleanup_error.value, sentinel)
                else:
                    await web._cancel_follow(socket)
                    await display_cancelled.wait()
                    assert not client_fiber._in_flight_calls

                if reader.started.is_set():
                    await reader.closed.wait()
                assert not web._followers
                assert not log._listeners
                assert not state.listeners

                driver_release.set()
                if not driver_result_observed:
                    await driver_task
                    driver_result_observed = True

                dispose_task = asyncio.create_task(client_fiber.dispose())
                await dispose_task
                await reply_fiber.dispose()
                await reader.provider_closed.wait()
                assert not client_fiber._in_flight_calls
                assert state.in_flight == 0
                assert state.adapter_stop_succeeded
            finally:
                display_gate.set()
                display_error_release.set()
                reader.release.set()
                reader.cleanup_gate.set()
                driver_release.set()
                if driver_started.is_set():
                    await web._cancel_follow(socket)
                if not driver_result_observed:
                    with suppress(asyncio.CancelledError):
                        await driver_task
                if dispose_task is not None:
                    with suppress(asyncio.CancelledError):
                        await dispose_task


def test_mobile_transport_scope_stays_local_after_reader_acquisition(
    mobile_gateway: Any,
) -> None:
    """The real Mobile gateway keeps its message/reply follower after short scope exit."""

    log, runtime, _client, _device, _private, scope_tasks = mobile_gateway
    session_id = "akashic:00000000-0000-4000-8000-000000000004"
    _mobile_append(log, session_id, "first", Input((ContentPart("text", "hello"),)))

    with _mobile_connected(mobile_gateway) as (websocket, epoch):
        scope_index = len(scope_tasks)
        _mobile_follow(websocket, epoch, session_id, "01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assert scope_tasks[scope_index][0] == "enter"
        assert scope_tasks[scope_index + 1][0] == "exit"
        assert scope_tasks[scope_index][1] is scope_tasks[scope_index + 1][1]
        assert [
            item["id"] for item in _mobile_receive_appended(websocket)["items"]
        ] == ["first"]
        assert runtime._message_followers
