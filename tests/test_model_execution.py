"""真实 ModelsState execution 的代际、重试和 task 绑定合同。"""

from __future__ import annotations

import asyncio
import json
import socket
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from aiohttp import web

from agent.config_models import Config
from agent.plugin_composition import (
    CHAT_MODELS,
    ModelRequest,
)
from agent.plugin_composition.models import DriverChatModel, DriverEmbeddingModel
from bootstrap.app_server import build_control_service
from bootstrap import tools as bootstrap
from bootstrap.init_workspace import init_workspace
from core.net.http import SharedHttpResources


async def _model_command(core, payload: dict[str, object]) -> dict[str, object]:
    """Configure the installed Models owner through its public RPC boundary."""

    service = build_control_service(core)
    resolve = service.resolve_method
    if resolve is None:
        raise AssertionError("ControlService 未提供动态 RPC resolver")
    async with resolve("models/command") as operation:
        if operation is None:
            raise AssertionError("live Root 未提供 models/command")
        params = operation.params.model_validate(payload)
        result = await operation.invoke(params, None)
    assert isinstance(result, dict)
    assert result.get("status") == 200, result
    body = result.get("body")
    assert isinstance(body, dict)
    return body


@pytest.mark.asyncio
async def test_model_execution_pins_binding_and_rejects_child_task_inheritance(
    tmp_path, monkeypatch,
):
    """模型 execution 固定一次 descriptor，不能被默认切换或子 task 改写。"""

    calls: list[dict[str, object]] = []
    transports: list[object] = []

    async def models(_request):
        return web.json_response({"data": [{"id": "first"}, {"id": "second"}]})

    async def completions(request):
        transports.append(request.transport)
        body = await request.json()
        calls.append(body)
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None},
                ],
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return web.Response(text=payload + "data: [DONE]\n\n", content_type="text/event-stream")

    provider = web.Application()
    provider.router.add_get("/v1/models", models)
    provider.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(provider)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    workspace = tmp_path / "workspace"
    init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    http = SharedHttpResources()
    sources = tmp_path / "plugins"
    for name in ("ui", "models", "openai_compatible"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    core = bootstrap.build_core_runtime(Config(), workspace, http, plugin_dirs=[sources])
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        await _model_command(core, {
            "type": "add_connection",
            "expected_revision": 0,
            "connection_id": "local",
            "name": "Local",
            "driver_id": "openai-compatible",
            "endpoint": f"http://127.0.0.1:{port}/v1",
            "auth_identity": "fixture",
            "credential": {"api_key": "fixture"},
        })
        capabilities = {
            "context_window": 32000,
            "max_output_tokens": 1024,
            "supports_tool_calls": True,
            "supported_reasoning_efforts": ["low", "high"],
        }
        await _model_command(core, {
            "type": "add_model",
            "expected_revision": 1,
            "model_id": "first",
            "connection_id": "local",
            "kind": "chat",
            "model": "first",
            "capabilities": capabilities,
            "capability_sources": {},
        })
        await _model_command(core, {
            "type": "set_default",
            "expected_revision": 2,
            "role": "default",
            "model_id": "first",
        })

        root = core.plugin_manager.live_root
        assert root is not None
        models_context, models_service = root._service_provider(CHAT_MODELS)
        async with models_context.runtime_scope():
            async with models_service.execution() as first_execution:
                first_descriptor = first_execution.chat("agent").descriptor
                assert first_descriptor.model_id == "first"
                response = await first_execution.chat("agent").complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "first"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )
                assert response.content == "ok"
                assert calls[-1]["model"] == "first"

                async with models_service.execution() as nested:
                    assert nested is first_execution
                    assert nested.chat("agent").descriptor == first_descriptor

                await _model_command(core, {
                    "type": "add_model",
                    "expected_revision": 3,
                    "model_id": "second",
                    "connection_id": "local",
                    "kind": "chat",
                    "model": "second",
                    "capabilities": capabilities,
                    "capability_sources": {},
                })
                await _model_command(core, {
                    "type": "set_default",
                    "expected_revision": 4,
                    "role": "default",
                    "model_id": "second",
                })

                # 已打开的 execution 固定旧 descriptor；默认切换只影响下一次 execution。
                async with models_service.execution() as still_first:
                    assert still_first is first_execution
                    assert still_first.chat("agent").descriptor == first_descriptor
                await first_execution.chat("agent").complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "retry"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )
                assert calls[-1]["model"] == "first"
                assert first_execution.chat("agent").descriptor == first_descriptor

                async def child_execution() -> None:
                    with pytest.raises(RuntimeError, match="不能由子 task 继承"):
                        async with models_service.execution():
                            raise AssertionError("子 task 不应取得 model execution")

                await asyncio.create_task(child_execution())
                assert len(calls) == 2
                assert transports[0] is transports[1]

            with pytest.raises(RuntimeError, match="连接已关闭"):
                await first_execution.chat("agent").complete(
                    ModelRequest(messages=({"role": "user", "content": "closed"},))
                )

            async with models_service.execution() as second_execution:
                second_descriptor = second_execution.chat("agent").descriptor
                assert second_descriptor.model_id == "second"
                assert second_descriptor.binding_id != first_descriptor.binding_id
                await second_execution.chat("agent").complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "second"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )

        assert [call["model"] for call in calls] == ["first", "first", "second"]
        assert transports[2] is not transports[0]
        assert not models_context._fiber._in_flight_calls
    finally:
        try:
            await core.bus.aclose()
        finally:
            try:
                await core.stop()
            finally:
                try:
                    await http.aclose()
                finally:
                    await runner.cleanup()


async def _noop_delta() -> None:
    """为真实流式 provider 请求提供最小 delta consumer。"""


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("driver_name,model_name", [
    ("openai_compatible", "fixture"),
    ("openai_compatible", "deepseek-v4-flash"),
    ("openai_compatible", "deepseek/deepseek-v4-flash-vision-exp"),
    ("opencode_go", "fixture"), ("codex", "fixture"),
])
async def test_driver_reuses_socket_and_reads_rotated_credentials(streaming, driver_name, model_name, caplog):
    """同一连接的两次请求复用 socket，并使用各自读取到的凭据。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from importlib import import_module
    definition = import_module(f"plugins.{driver_name}.driver").definition
    driver_id = driver_name.replace("_", "-")
    from tests.model_plugin_fakes import BoundChatModelFake

    seen = []
    dropped = 0
    partial = 0
    caplog.set_level("DEBUG", logger="core.net.http")

    async def complete(request):
        nonlocal dropped, partial
        seen.append((request.transport, request.headers["Authorization"]))
        assert "Cookie" not in request.headers
        if driver_name == "codex":
            return web.Response(
                text='data: {"type":"response.output_text.delta","delta":"ok"}\n\ndata: {"type":"response.completed","response":{}}\n\n',
                content_type="text/event-stream",
            )
        body = await request.json()
        deepseek = driver_name == "openai_compatible" and "deepseek-v4-" in model_name
        assert bool(body.get("stream")) == (streaming or deepseek)
        if deepseek and not streaming:
            assert body["thinking"] == {"type": "disabled"}
            assert "reasoning_effort" not in body
        elif driver_name == "openai_compatible":
            assert "thinking" not in body
        if streaming or deepseek:
            if dropped:
                # 断流发生在任何内容增量之前：provider 未产生可观察输出，
                # 该传输断流才可安全重试同一请求。
                dropped -= 1
                return web.Response(text=": keepalive\n\n", content_type="text/event-stream")
            if partial:
                # 协议层已观察到真实增量后断流：远端效果不可证，不得重试。
                partial -= 1
                return web.Response(
                    text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
                    content_type="text/event-stream",
                )
            payload = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            payload += 'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\ndata: [DONE]\n\n'
            return web.Response(text=payload, content_type="text/event-stream")
        return web.json_response({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    app = web.Application()
    async def add_cookie(_request, response):
        response.headers["Set-Cookie"] = "provider-session=previous-credential; Path=/"

    app.on_response_prepare.append(add_cookie)
    app.router.add_post("/v1/chat/completions", complete)
    app.router.add_post("/v1/responses", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"
        token = "first"

        async def read(self):
            return {"driver": "codex", "api_key": self.token, "access_token": self.token,
                    "account_id": "test", "expires_at": "2099-01-01T00:00:00+00:00"}

    credential = Credential()
    descriptor = replace(BoundChatModelFake(object()).descriptor, driver_id=driver_id, model=model_name)
    driver = await definition().open(
        DriverConnectionDescriptor("test-connection", "local", driver_id,
                                   f"http://127.0.0.1:{port}/v1", "test", {"max_retries": 1} if driver_name == "openai_compatible" else {}),
        credential,
    )
    deltas = []

    async def capture(delta):
        deltas.append(dict(delta))

    try:
        bound = driver.bind_chat(descriptor, {})
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            on_delta=capture if streaming else None,
            disable_reasoning=not streaming,
        )
        assert (await bound.complete(request)).content == "ok"
        credential.token = "second"
        assert (await bound.complete(request)).content == "ok"
        # 负载可让尾流超过 10 ms；只有明确记录放弃复用时才允许新 socket。
        if seen[0][0] is not seen[1][0]:
            assert "尾流未结束，关闭连接" in caplog.text
        assert [item[1] for item in seen] == ["Bearer first", "Bearer second"]
        if driver_name == "openai_compatible" and "deepseek-v4-" in model_name:
            from agent.plugin_composition.models import TransportError
            deltas.clear()
            if streaming:
                partial = 1
                with pytest.raises(TransportError, match="terminal marker") as failure:
                    await bound.complete(request)
                assert not failure.value.retryable
                assert len(seen) == 3, "an observed partial response must not replay"
                assert deltas == [{"content_delta": "ok"}]
            else:
                # HTTP 200 已进入流处理：零 delta 的 EOF 断流仍不可证——
                # provider 已接收请求，无论有无增量都不得自动重发。
                dropped = 1
                with pytest.raises(TransportError, match="terminal marker") as failure:
                    await bound.complete(request)
                assert not getattr(failure.value, "send_evidence", None), (
                    "zero deltas is not proof the request was unprocessed"
                )
                assert len(seen) == 3, "a zero-delta stream failure must not replay"
                partial = 2
                with pytest.raises(TransportError, match="terminal marker") as failure:
                    await bound.complete(request)
                assert not failure.value.retryable, "partial bytes observed, remote effect uncertain"
                assert len(seen) == 4, "a partial response must not replay even without a preview callback"
                assert deltas == []
        await driver.aclose()
        with pytest.raises(RuntimeError, match="连接已关闭"):
            await bound.complete(request)
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("observe", [False, True])
async def test_done_does_not_wait_for_a_stalled_http_tail(observe):
    """DONE 后服务端不结束正文时，仍交付已完成结果并关闭该连接。"""
    import httpx
    from plugins.openai_compatible.driver import _consume_stream

    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
            waiting.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    response = httpx.Response(200, stream=Body())
    try:
        async with asyncio.timeout(1):
            result = await _consume_stream(response, (lambda _delta: _noop_delta()) if observe else None)
        assert result.content == "ok"
        assert waiting.is_set() and cancelled.is_set()
    finally:
        await response.aclose()


async def _mount_model_driver_graph(
    tmp_path,
    *,
    block_close=False,
    with_driver_b_consumer=False,
    hold_models_cleanup=False,
    with_live_models=False,
    with_driver_dependency=False,
):
    """Mount Models and real driver Fibers used by connection-lifetime tests."""

    from agent.plugin_composition import (
        CHAT_MODELS,
        CapabilitySources,
        ChatModelSelection,
        CompositionRoot,
        DiscoveredModel,
        DriverConnection,
        EmbeddingResult,
        EMBEDDINGS,
        LLMResponse,
        ModelContinuation,
        ModelCapabilities,
        ModelKind,
        MODEL_DRIVERS,
        MODEL_CATALOG,
        ModelDriverDefinition,
        PluginRuntime,
        RUNTIME_STARTED,
        RUNTIME_STARTING,
        RUNTIME_STOPPING,
        ServiceKey,
    )
    from plugins.models.state import ModelsState
    from plugins.models.store import ModelsStore

    root = CompositionRoot("model-driver-scope")
    model_contexts = []
    driver_contexts = {}
    driver_fibers = {}
    driver_definitions = {}
    driver_registration_effects = {}
    close_events = []
    open_started = asyncio.Event()
    release_open = asyncio.Event()
    discover_started = asyncio.Event()
    release_discover = asyncio.Event()
    discover_calls = []
    discover_error = [None]
    probe_calls = []
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    open_calls = []
    bind_chat_calls = []
    close_snapshot_states = []
    auth_started = asyncio.Event()
    release_auth = asyncio.Event()
    auth_returned = asyncio.Event()
    release_auth_return = asyncio.Event()
    finish_started = asyncio.Event()
    release_finish = asyncio.Event()
    finish_returned = asyncio.Event()
    release_finish_return = asyncio.Event()
    auth_finish_result = [{
        "status": "complete",
        "name": "auth connection",
        "endpoint": "https://example.invalid/auth",
        "auth_identity": "auth-user",
        "credential": {"token": "auth-token"},
        "driver_config": {},
    }]
    auth_callback_events = []
    cancel_states = []
    cancel_error = [None]
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    consumer_cleanup_started = asyncio.Event()
    release_consumer = asyncio.Event()
    driver_b_consumer_cleanup_started = asyncio.Event()
    release_driver_b_consumer = asyncio.Event()
    models_cleanup_started = asyncio.Event()
    release_models_cleanup = asyncio.Event()
    fail_open_ids = set()
    fail_close_ids = set()
    credential_reads = []
    live_chat_calls = []
    live_chat_descriptors = []
    live_embedding_calls = []
    live_embedding_descriptors = []
    driver_resources = {
        driver_id: ServiceKey(f"test.{driver_id}.resource")
        for driver_id in ("driver-a", "driver-b")
    }
    driver_dependency = ServiceKey("test.driver.dependency")
    driver_dependency_fiber = None

    class LiveChat:
        def __init__(self, driver_id, tag, descriptor):
            self.driver_id = driver_id
            self.tag = tag
            self.descriptor = descriptor

        async def complete(self, request):
            live_chat_calls.append(
                (self.tag, self.descriptor.binding_id, request.messages)
            )
            return LLMResponse(
                content=f"{self.tag}:{self.descriptor.model}",
                continuation=ModelContinuation(
                    self.descriptor.binding_id,
                    {"tag": self.tag, "cursor": "next"},
                ),
            )

        def estimate_context_tokens(self, messages, tools=()):
            return len(messages) + len(tools)

        def estimate_appended_message_tokens(self, messages):
            return len(messages)

        @property
        def max_tool_schemas(self):
            return None

    class LiveEmbedding:
        def __init__(self, driver_id, tag, descriptor):
            self.driver_id = driver_id
            self.tag = tag
            self.descriptor = descriptor

        async def embed(self, texts):
            live_embedding_calls.append((self.tag, tuple(texts)))
            vector = tuple(float(index + 1) for index in range(self.descriptor.dimensions))
            return EmbeddingResult(
                vectors=tuple(vector for _text in texts),
            )

    def make_live_connection(driver_id, tag, descriptor, *, close=None):
        def bind_chat(bound_descriptor, _config):
            live_chat_descriptors.append((tag, bound_descriptor))
            return LiveChat(driver_id, tag, bound_descriptor)

        def bind_embedding(bound_descriptor, _config):
            live_embedding_descriptors.append((tag, bound_descriptor))
            return LiveEmbedding(driver_id, tag, bound_descriptor)

        return DriverConnection(bind_chat, bind_embedding, close=close)

    async def apply_models(ctx):
        store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
        store.initialize()
        state = ModelsState(
            store,
            context=ctx,
        )
        model_contexts.append((ctx, state))
        await ctx.provide(MODEL_DRIVERS, state.drivers)
        await ctx.provide(CHAT_MODELS, state.chat_models)
        await ctx.provide(EMBEDDINGS, state.embeddings)
        await ctx.provide(MODEL_CATALOG, state.catalog)
        from plugins.models.settings import MODEL_SETTINGS

        await ctx.provide(MODEL_SETTINGS, state.settings)
        if hold_models_cleanup:
            async def cleanup_models() -> None:
                models_cleanup_started.set()
                await release_models_cleanup.wait()

            await ctx.effect(lambda: cleanup_models, label="models-hard-cleanup")

    models_fiber = await root.mount(
        apply_models,
        name="models",
        runtime=PluginRuntime(
            "models",
            "models:g1",
            tmp_path,
            tmp_path / "models-data",
            tmp_path,
            {},
        ),
    )

    if with_driver_dependency:
        async def apply_driver_dependency(ctx):
            await ctx.provide(driver_dependency, object())

        driver_dependency_fiber = await root.mount(
            apply_driver_dependency,
            name="driver-dependency",
            runtime=PluginRuntime(
                "driver-dependency",
                "driver-dependency:g1",
                tmp_path,
                tmp_path / "driver-dependency-data",
                tmp_path,
                {},
            ),
        )

    def definition_for(driver_id):
        async def open_driver(descriptor, _credential):
            connection_id = descriptor.connection_id
            open_calls.append(connection_id)
            if with_live_models:
                credential_reads.append((driver_id, connection_id))
                await _credential.read()
            if driver_id == "driver-a" and connection_id == "first":
                open_started.set()
                await release_open.wait()
            if connection_id in fail_open_ids:
                raise RuntimeError(f"{connection_id} open failed")

            async def close():
                ctx = driver_contexts[driver_id]
                async with ctx.runtime_scope():
                    assert ctx.require(MODEL_DRIVERS) is model_contexts[0][1].drivers
                    close_events.append(connection_id)
                    close_snapshot_states.append(
                        model_contexts[0][1].store.read_snapshot()
                    )
                    if block_close and connection_id == "test":
                        close_started.set()
                        await release_close.wait()
                    if connection_id in fail_close_ids:
                        raise LookupError(f"{connection_id} close failed")

            def bind_chat(*_args):
                bind_chat_calls.append(_args[0].connection_id)
                if with_live_models:
                    return make_live_connection(
                        driver_id,
                        driver_id,
                        _args[0],
                    ).bind_chat(*_args)
                return cast(DriverChatModel, object())  # Inject an invalid driver result.

            def bind_embedding(*_args):
                if with_live_models:
                    return make_live_connection(
                        driver_id,
                        driver_id,
                        _args[0],
                    ).bind_embedding(*_args)
                return cast(DriverEmbeddingModel, object())  # Inject an invalid driver result.

            return DriverConnection(bind_chat, bind_embedding, close=close)

        async def discover_driver(descriptor, _credential):
            assert driver_contexts[driver_id].require(MODEL_DRIVERS) is model_contexts[0][1].drivers
            discover_calls.append(descriptor.connection_id)
            if driver_id == "driver-a":
                discover_started.set()
                await release_discover.wait()
            if discover_error[0] is not None:
                raise discover_error[0]
            return (
                DiscoveredModel(
                    kind=ModelKind.CHAT,
                    model="fixture-chat",
                    capabilities=ModelCapabilities(),
                    capability_sources=CapabilitySources(),
                ),
            )

        async def probe_driver(descriptor, _credential):
            probe_calls.append(descriptor.connection_id)
            if driver_id == "driver-b" and descriptor.connection_id == "created":
                probe_started.set()
                await release_probe.wait()

        async def start_auth(_input):
            auth_callback_events.append((driver_id, "start", dict(_input)))
            auth_started.set()
            await release_auth.wait()
            auth_returned.set()
            await release_auth_return.wait()
            return {"state": {"step": 1}, "challenge": {"kind": "test"}}

        async def finish_auth(_state):
            auth_callback_events.append((driver_id, "finish", dict(_state)))
            finish_started.set()
            await release_finish.wait()
            finish_returned.set()
            await release_finish_return.wait()
            return dict(auth_finish_result[0])

        async def cancel_auth(state):
            auth_callback_events.append((driver_id, "cancel", dict(state)))
            cancel_states.append(dict(state))
            if cancel_error[0] is not None:
                raise cancel_error[0]
            return dict(state)

        return ModelDriverDefinition(
            driver_id,
            "test-v1",
            open_driver,
            discover=discover_driver,
            probe=probe_driver if driver_id == "driver-b" else None,
            start_auth=start_auth,
            finish_auth=finish_auth,
            cancel_auth=cancel_auth,
        )

    async def mount_driver(driver_id):
        definition = definition_for(driver_id)

        async def apply_driver(ctx):
            driver_contexts[driver_id] = ctx
            if with_driver_dependency:
                ctx.require(driver_dependency)
            await ctx.provide(driver_resources[driver_id], object())
            effect = await ctx.require(MODEL_DRIVERS).register(ctx, definition)
            driver_definitions[driver_id] = definition
            driver_registration_effects[driver_id] = effect

        dependencies = (MODEL_DRIVERS,)
        if with_driver_dependency:
            dependencies += (driver_dependency,)

        driver_fibers[driver_id] = await root.mount(
            apply_driver,
            name=driver_id,
            inject=dependencies,
            runtime=PluginRuntime(
                driver_id,
                f"{driver_id}:g1",
                tmp_path,
                tmp_path / f"{driver_id}-data",
                tmp_path,
                {},
            ),
        )

    await mount_driver("driver-a")
    await mount_driver("driver-b")

    driver_b_consumer_fiber = None
    if with_driver_b_consumer:
        async def apply_driver_b_consumer(ctx):
            ctx.require(driver_resources["driver-b"])

            async def cleanup() -> None:
                driver_b_consumer_cleanup_started.set()
                await release_driver_b_consumer.wait()

            await ctx.effect(lambda: cleanup, label="hard-driver-b-consumer")

        driver_b_consumer_fiber = await root.mount(
            apply_driver_b_consumer,
            name="driver-b-consumer",
            inject=(driver_resources["driver-b"],),
            runtime=PluginRuntime(
                "driver-b-consumer",
                "driver-b-consumer:g1",
                tmp_path,
                tmp_path / "driver-b-consumer-data",
                tmp_path,
                {},
            ),
        )

    async def apply_consumer(ctx):
        ctx.require(driver_resources["driver-a"])

        async def cleanup():
            consumer_cleanup_started.set()
            await release_consumer.wait()

        await ctx.effect(lambda: cleanup, label="hard-driver-consumer")

    consumer_fiber = await root.mount(
        apply_consumer,
        name="driver-a-consumer",
        inject=(driver_resources["driver-a"],),
        runtime=PluginRuntime(
            "driver-a-consumer",
            "driver-a-consumer:g1",
            tmp_path,
            tmp_path / "consumer-data",
            tmp_path,
            {},
        ),
    )
    unrelated_fiber = await root.mount(
        lambda _ctx: None,
        name="unrelated-model-owner",
        runtime=PluginRuntime(
            "unrelated-model-owner",
            "unrelated-model-owner:g1",
            tmp_path,
            tmp_path / "unrelated-data",
            tmp_path,
            {},
        ),
    )
    unrelated_events = []
    await unrelated_fiber.context.on(
        RUNTIME_STARTING, lambda _event: unrelated_events.append("starting")
    )
    await unrelated_fiber.context.on(
        RUNTIME_STARTED, lambda _event: unrelated_events.append("started")
    )
    await unrelated_fiber.context.on(
        RUNTIME_STOPPING, lambda _event: unrelated_events.append("stopping")
    )
    assert model_contexts
    return {
        "tmp_path": tmp_path,
        "root": root,
        "models_fiber": models_fiber,
        "driver_dependency_fiber": driver_dependency_fiber,
        "driver_dependency": driver_dependency,
        "model_context": model_contexts[0][0],
        "state": model_contexts[0][1],
        "driver_contexts": driver_contexts,
        "driver_fibers": driver_fibers,
        "driver_definitions": driver_definitions,
        "driver_registration_effects": driver_registration_effects,
        "consumer_fiber": consumer_fiber,
        "unrelated_fiber": unrelated_fiber,
        "unrelated_events": unrelated_events,
        "open_started": open_started,
        "release_open": release_open,
        "discover_started": discover_started,
        "release_discover": release_discover,
        "discover_calls": discover_calls,
        "discover_error": discover_error,
        "probe_calls": probe_calls,
        "probe_started": probe_started,
        "release_probe": release_probe,
        "open_calls": open_calls,
        "bind_chat_calls": bind_chat_calls,
        "close_snapshot_states": close_snapshot_states,
        "auth_started": auth_started,
        "release_auth": release_auth,
        "auth_returned": auth_returned,
        "release_auth_return": release_auth_return,
        "finish_started": finish_started,
        "release_finish": release_finish,
        "finish_returned": finish_returned,
        "release_finish_return": release_finish_return,
        "auth_finish_result": auth_finish_result,
        "auth_callback_events": auth_callback_events,
        "cancel_states": cancel_states,
        "cancel_error": cancel_error,
        "fail_open_ids": fail_open_ids,
        "close_started": close_started,
        "release_close": release_close,
        "consumer_cleanup_started": consumer_cleanup_started,
        "release_consumer": release_consumer,
        "driver_b_consumer_fiber": driver_b_consumer_fiber,
        "driver_b_consumer_cleanup_started": driver_b_consumer_cleanup_started,
        "release_driver_b_consumer": release_driver_b_consumer,
        "models_cleanup_started": models_cleanup_started,
        "release_models_cleanup": release_models_cleanup,
        "fail_close_ids": fail_close_ids,
        "close_events": close_events,
        "credential_reads": credential_reads,
        "live_chat_calls": live_chat_calls,
        "live_chat_descriptors": live_chat_descriptors,
        "live_embedding_calls": live_embedding_calls,
        "live_embedding_descriptors": live_embedding_descriptors,
        "make_live_connection": make_live_connection,
    }


def _model_test_connection(connection_id, driver_id):
    from plugins.models.store import StoredConnection

    return StoredConnection(
        connection_id,
        connection_id,
        driver_id,
        "https://example.invalid",
        "test",
        {},
        True,
    )


@pytest.mark.asyncio
async def test_stale_models_context_rejects_sync_queries_after_same_fiber_reload(tmp_path):
    """A reloaded Models Fiber rejects every old public sync view at the old Context."""

    from agent.plugin_composition import (
        CHAT_MODELS,
        ChatModelSelection,
        CompositionError,
        CompositionRoot,
        EMBEDDINGS,
        MODEL_CATALOG,
        MODEL_DRIVERS,
        PluginRuntime,
        ServiceKey,
    )
    from plugins.models.settings import MODEL_SETTINGS
    from plugins.models.state import ModelsState
    from plugins.models.store import ModelsStore

    root = CompositionRoot("stale-models-context")
    dependency = ServiceKey("test.models.reload-dependency")
    states = []
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()

    async def apply_dependency(ctx):
        await ctx.provide(dependency, object())

    async def apply_models(ctx):
        ctx.require(dependency)
        state = ModelsState(store, context=ctx)
        states.append(state)
        await ctx.provide(MODEL_DRIVERS, state.drivers)
        await ctx.provide(CHAT_MODELS, state.chat_models)
        await ctx.provide(EMBEDDINGS, state.embeddings)
        await ctx.provide(MODEL_CATALOG, state.catalog)
        await ctx.provide(MODEL_SETTINGS, state.settings)

    runtime = PluginRuntime(
        "stale-models",
        "stale-models:g1",
        tmp_path,
        tmp_path / "models-data",
        tmp_path,
        {},
    )
    dependency_fiber = await root.mount(
        apply_dependency,
        name="models-reload-dependency",
        runtime=PluginRuntime(
            "models-reload-dependency",
            "models-reload-dependency:g1",
            tmp_path,
            tmp_path / "dependency-data",
            tmp_path,
            {},
        ),
    )
    models_fiber = await root.mount(
        apply_models,
        name="stale-models",
        inject=(dependency,),
        runtime=runtime,
    )
    try:
        old_state = states[0]
        await dependency_fiber.dispose()
        await root.mount(
            apply_dependency,
            name="models-reload-dependency",
            runtime=PluginRuntime(
                "models-reload-dependency",
                "models-reload-dependency:g2",
                tmp_path,
                tmp_path / "dependency-data-2",
                tmp_path,
                {},
            ),
        )
        assert models_fiber.context is states[1].context
        for read in (
            old_state.catalog_snapshot,
            lambda: old_state.validate_chat_selection(ChatModelSelection()),
            old_state.chat_contributors,
            lambda: old_state.describe_embedding(None),
        ):
            with pytest.raises(CompositionError) as excinfo:
                read()
            assert excinfo.value.code == "STALE_ACTIVATION"
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_models_owner_scope_survives_unloading_but_rejects_new_scope(tmp_path):
    """An admitted Models owner may drain while new Tasks are rejected."""

    from agent.plugin_composition import ChatModelSelection, CompositionError, FiberState

    graph = await _mount_model_driver_graph(
        tmp_path,
        hold_models_cleanup=True,
    )
    unloading = None
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        async with graph["model_context"].runtime_scope():
            unloading = asyncio.create_task(graph["models_fiber"].dispose())
            await graph["consumer_cleanup_started"].wait()
            assert graph["models_fiber"].state is FiberState.UNLOADING
            assert graph["state"].catalog_snapshot().revision == before.revision
            assert graph["state"].validate_chat_selection(
                ChatModelSelection()
            ) == ChatModelSelection()
            async with graph["model_context"].runtime_scope():
                assert graph["state"].catalog_snapshot().revision == before.revision

            async def acquire_new_scope():
                async with graph["model_context"].runtime_scope():
                    return None

            with pytest.raises(CompositionError) as excinfo:
                await asyncio.create_task(acquire_new_scope())
            assert excinfo.value.code == "OWNER_UNAVAILABLE"

        graph["release_consumer"].set()
        await graph["models_cleanup_started"].wait()
        graph["release_models_cleanup"].set()
        await unloading
        assert graph["models_fiber"].state is FiberState.DISPOSED
        assert not graph["models_fiber"]._in_flight_calls
    finally:
        graph["release_consumer"].set()
        graph["release_models_cleanup"].set()
        if unloading is not None:
            await unloading
        await graph["root"].dispose()


async def _start_graph_auth(graph, connection_id):
    """Run the fixture's real start-auth callback to a stored attempt."""

    from plugins.models.settings import StartConnectionAuth

    task = asyncio.create_task(
        graph["state"]._start_auth(
            StartConnectionAuth("driver-a", connection_id, {"prompt": "test"})
        )
    )
    await graph["auth_started"].wait()
    graph["release_auth"].set()
    await graph["auth_returned"].wait()
    graph["release_auth_return"].set()
    return await task


async def _apply_graph_start_auth(graph, connection_id):
    """Start one auth attempt through the public Models settings facade."""

    from plugins.models.settings import StartConnectionAuth

    task = asyncio.create_task(
        graph["state"].settings.apply(
            StartConnectionAuth("driver-a", connection_id, {"prompt": "test"})
        )
    )
    await graph["auth_started"].wait()
    graph["release_auth"].set()
    await graph["auth_returned"].wait()
    graph["release_auth_return"].set()
    return await task


async def _configure_live_models(graph):
    """Persist one chat and one embedding model through real settings calls."""

    from agent.plugin_composition import CapabilitySources, ModelCapabilities, ModelKind
    from plugins.models.settings import AddConnection, AddModel, SetDefaultModel

    settings = graph["state"].settings
    await settings.apply(
        AddConnection(
            0,
            "chat-connection",
            "Chat connection",
            "driver-a",
            "https://chat.example.invalid",
            "chat-user",
            {"token": "chat-token"},
        )
    )
    await settings.apply(
        AddModel(
            1,
            "chat-model",
            "chat-connection",
            ModelKind.CHAT,
            "chat-model",
            ModelCapabilities(
                context_window=4096,
                max_output_tokens=256,
                supports_tool_calls=True,
                supported_reasoning_efforts=("low", "high"),
            ),
            CapabilitySources(),
        )
    )
    await settings.apply(SetDefaultModel(2, "default", "chat-model"))
    await settings.apply(
        AddConnection(
            3,
            "embedding-connection",
            "Embedding connection",
            "driver-b",
            "https://embedding.example.invalid",
            "embedding-user",
            {"token": "embedding-token"},
        )
    )
    await settings.apply(
        AddModel(
            4,
            "embedding-model",
            "embedding-connection",
            ModelKind.EMBEDDING,
            "embedding-model",
            ModelCapabilities(
                embedding_dimensions=2,
                embedding_normalization="l2",
            ),
            CapabilitySources(),
        )
    )
    await settings.apply(SetDefaultModel(5, None, "embedding-model"))
    snapshot = graph["state"].store.read_snapshot()
    assert snapshot is not None and snapshot.revision == 6
    return snapshot


async def _replace_driver_with_same_definition(graph, driver_id, *, name):
    """Dispose one real registration Fiber, then register its same definition again."""

    from agent.plugin_composition import MODEL_DRIVERS, PluginRuntime

    old_fiber = graph["driver_fibers"][driver_id]
    definition = graph["driver_definitions"][driver_id]
    unloading = asyncio.create_task(old_fiber.dispose())
    if driver_id == "driver-a":
        await graph["consumer_cleanup_started"].wait()
        graph["release_consumer"].set()
    await unloading

    async def apply_replacement(ctx):
        graph["driver_contexts"][driver_id] = ctx
        effect = await ctx.require(MODEL_DRIVERS).register(ctx, definition)
        graph["driver_registration_effects"][driver_id] = effect

    replacement = await graph["root"].mount(
        apply_replacement,
        name=name,
        inject=(MODEL_DRIVERS,),
        runtime=PluginRuntime(
            f"{driver_id}-replacement",
            f"{driver_id}-replacement:g1",
            graph["tmp_path"],
            graph["tmp_path"] / f"{driver_id}-replacement-data",
            graph["tmp_path"],
            {},
        ),
    )
    graph["driver_fibers"][driver_id] = replacement
    return old_fiber, replacement


async def _reregister_driver_in_same_context(graph, driver_id):
    """Replace one registration Effect without reloading its existing Fiber."""

    from agent.plugin_composition import MODEL_DRIVERS

    fiber = graph["driver_fibers"][driver_id]
    context = graph["driver_contexts"][driver_id]
    definition = graph["driver_definitions"][driver_id]
    old_effect = graph["driver_registration_effects"][driver_id]
    await old_effect.aclose()
    new_effect = await context.require(MODEL_DRIVERS).register(context, definition)
    graph["driver_registration_effects"][driver_id] = new_effect
    return fiber, context, old_effect, new_effect


@pytest.mark.asyncio
async def test_live_models_facades_pin_execution_and_persist_call(tmp_path):
    """Real facades pin one execution while settings and child Tasks change."""

    from agent.plugin_composition import ModelKind, ModelRequest, ModelUnavailableError, PluginRuntime
    from plugins.models.settings import AddModel, SetDefaultModel

    graph = await _mount_model_driver_graph(tmp_path, with_live_models=True)
    try:
        await _configure_live_models(graph)
        state = graph["state"]
        models_identity = (
            graph["models_fiber"].context,
            graph["models_fiber"]._activation_token,
        )
        driver_identity = (
            graph["driver_fibers"]["driver-a"].context,
            graph["driver_fibers"]["driver-a"]._activation_token,
        )

        async with state.chat_models.execution() as first:
            first_chat = first.chat("agent")
            first_descriptor = first_chat.descriptor
            async with state.chat_models.execution() as nested:
                assert nested is first
                assert nested.chat("agent").descriptor == first_descriptor

            async def child_execution():
                with pytest.raises(RuntimeError, match="不能由子 task 继承"):
                    async with state.chat_models.execution():
                        raise AssertionError("child execution must not inherit")

            await asyncio.create_task(child_execution())

            async def independent_child():
                async with state.chat_models.independent_execution() as independent:
                    return independent.chat("agent").descriptor

            assert await asyncio.create_task(independent_child()) == first_descriptor

            await state.settings.apply(
                AddModel(
                    6,
                    "chat-model-next",
                    "chat-connection",
                    ModelKind.CHAT,
                    "chat-model-next",
                    first_descriptor.capabilities,
                    first_descriptor.capability_sources,
                )
            )
            await state.settings.apply(SetDefaultModel(7, "default", "chat-model-next"))
            response = await first_chat.complete(
                ModelRequest(messages=({"role": "user", "content": "pinned"},))
            )
            assert response.content == "driver-a:chat-model"
            assert first_chat.descriptor == first_descriptor

        async with state.chat_models.execution() as next_execution:
            next_descriptor = next_execution.chat("agent").descriptor
            assert next_descriptor.model_id == "chat-model-next"
            assert next_descriptor.binding_id != first_descriptor.binding_id

        call_id = response.call_record_id
        assert call_id is not None
        record = state.store.read_call(call_id)
        assert record["state"] == "success"
        assert record["binding"]["binding_id"] == first_descriptor.binding_id
        assert "credential" not in record["binding"]
        assert ("driver-a", "chat-connection") in graph["credential_reads"]

        before_invalid = state.store.read_snapshot()
        credential_handle = state.store.credential_handle(
            "chat-connection", "chat-user"
        )
        before_credential = dict(await credential_handle.read())
        before_credentials = tuple(graph["credential_reads"])
        with pytest.raises(ModelUnavailableError):
            await state.settings.apply(
                AddModel(
                    8,
                    "invalid-model",
                    "missing-connection",
                    ModelKind.CHAT,
                    "invalid-model",
                    first_descriptor.capabilities,
                    first_descriptor.capability_sources,
                )
            )
        assert state.store.read_snapshot() == before_invalid
        assert dict(await credential_handle.read()) == before_credential
        assert tuple(graph["credential_reads"]) == before_credentials

        old_unrelated_context = graph["unrelated_fiber"].context
        await graph["unrelated_fiber"].dispose()
        async def apply_unrelated(_ctx):
            return None

        graph["unrelated_fiber"] = await graph["root"].mount(
            apply_unrelated,
            name="unrelated-model-owner",
            runtime=PluginRuntime(
                "unrelated-model-owner",
                "unrelated-model-owner:g2",
                tmp_path,
                tmp_path / "unrelated-data-2",
                tmp_path,
                {},
            ),
        )
        assert (
            graph["models_fiber"].context,
            graph["models_fiber"]._activation_token,
        ) == models_identity
        assert (
            graph["driver_fibers"]["driver-a"].context,
            graph["driver_fibers"]["driver-a"]._activation_token,
        ) == driver_identity
        assert graph["unrelated_fiber"].context is not old_unrelated_context
        async with state.chat_models.execution() as after_unrelated:
            assert after_unrelated.chat("agent").descriptor == next_descriptor
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_same_context_reregistration_changes_identity_without_fiber_reload(tmp_path):
    """A registration Effect can be replaced while its Fiber and Context stay live."""

    graph = await _mount_model_driver_graph(tmp_path, with_live_models=True)
    try:
        await _configure_live_models(graph)
        state = graph["state"]
        fiber = graph["driver_fibers"]["driver-a"]
        context = graph["driver_contexts"]["driver-a"]
        activation = fiber._activation_token
        async with state.chat_models.execution() as before_execution:
            before_descriptor = before_execution.chat("agent").descriptor

        same_fiber, same_context, old_effect, new_effect = (
            await _reregister_driver_in_same_context(graph, "driver-a")
        )
        assert same_fiber is fiber
        assert same_context is context
        assert fiber.context is context
        assert fiber._activation_token is activation
        assert old_effect not in fiber.effects
        assert new_effect in fiber.effects

        async with state.chat_models.execution() as after_execution:
            after_descriptor = after_execution.chat("agent").descriptor
        assert state.store.read_snapshot().revision == 6
        assert after_descriptor.plugin_snapshot_id != before_descriptor.plugin_snapshot_id
        assert after_descriptor.binding_id != before_descriptor.binding_id
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_models_registration_replacement_rejects_old_continuation(tmp_path):
    """A same-id re-registration changes binding identity before new driver I/O."""

    from agent.plugin_composition import ModelRequest, ModelUnavailableError

    graph = await _mount_model_driver_graph(tmp_path, with_live_models=True)
    try:
        await _configure_live_models(graph)
        state = graph["state"]
        async with state.chat_models.execution() as execution:
            old_chat = execution.chat("agent")
            old_descriptor = old_chat.descriptor
            old_response = await old_chat.complete(
                ModelRequest(messages=({"role": "user", "content": "old"},))
            )
        assert old_response.call_record_id is not None
        assert old_response.continuation is not None
        old_record = state.store.read_call(old_response.call_record_id)
        call_count = len(state.store.read_calls("", 1000))
        old_context = graph["driver_fibers"]["driver-a"].context
        await _replace_driver_with_same_definition(
            graph,
            "driver-a",
            name="driver-a-replacement-for-continuation",
        )
        assert graph["driver_fibers"]["driver-a"].context is not old_context

        async with state.chat_models.execution() as replacement_execution:
            replacement_chat = replacement_execution.chat("agent")
            replacement_descriptor = replacement_chat.descriptor
            assert state.store.read_snapshot().revision == 6
            assert replacement_descriptor.plugin_snapshot_id != old_descriptor.plugin_snapshot_id
            assert replacement_descriptor.binding_id != old_descriptor.binding_id
            assert old_record["binding"]["binding_id"] == old_descriptor.binding_id
            live_calls = len(graph["live_chat_calls"])
            with pytest.raises(ModelUnavailableError):
                await replacement_chat.complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "continued"},),
                        continuation=old_response.continuation,
                    )
                )
            assert len(graph["live_chat_calls"]) == live_calls
            assert len(state.store.read_calls("", 1000)) == call_count
            current_response = await replacement_chat.complete(
                ModelRequest(messages=({"role": "user", "content": "new"},))
            )
            assert current_response.content == "driver-a:chat-model"
        assert state.store.read_call(old_response.call_record_id) == old_record
        assert len(state.store.read_calls("", 1000)) == call_count + 1
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_embedding_namespace_tracks_only_its_driver_registration(tmp_path):
    """Chat and embedding bindings keep independent live registration identities."""

    from agent.plugin_composition import ModelKind
    from plugins.models.settings import AddModel, SetDefaultModel

    graph = await _mount_model_driver_graph(tmp_path, with_live_models=True)
    try:
        await _configure_live_models(graph)
        state = graph["state"]
        async with state.chat_models.execution() as chat_execution:
            chat_descriptor = chat_execution.chat("agent").descriptor
            await state.settings.apply(
                AddModel(
                    6,
                    "chat-model-next",
                    "chat-connection",
                    ModelKind.CHAT,
                    "chat-model-next",
                    chat_descriptor.capabilities,
                    chat_descriptor.capability_sources,
                )
            )
            await state.settings.apply(SetDefaultModel(7, "default", "chat-model-next"))
            async with state.embeddings.bind() as nested_embedding:
                first_descriptor = nested_embedding.descriptor
                result = await nested_embedding.embed(("one", "two"))
                assert first_descriptor.driver_id == "driver-b"
                assert first_descriptor.dimensions == 2
                assert all(len(vector) == 2 for vector in result.vectors)
                assert first_descriptor.plugin_snapshot_id != chat_descriptor.plugin_snapshot_id
                assert chat_descriptor.model_revision == 6
                assert first_descriptor.model_revision == 6
                assert chat_execution.chat("agent").descriptor == chat_descriptor
        async with state.embeddings.bind() as external_embedding:
            external_descriptor = external_embedding.descriptor
            external_result = await external_embedding.embed(("outside",))
        assert external_descriptor.model_revision == 8
        assert external_descriptor.plugin_snapshot_id == first_descriptor.plugin_snapshot_id
        assert external_descriptor.identity == first_descriptor.identity
        assert external_descriptor.dimensions == first_descriptor.dimensions
        assert external_result.vectors == ((1.0, 2.0),)
        assert chat_descriptor.driver_id == "driver-a"

        await _replace_driver_with_same_definition(
            graph,
            "driver-a",
            name="driver-a-replacement-for-embedding",
        )
        async with state.embeddings.bind() as after_chat_replacement:
            after_chat_descriptor = after_chat_replacement.descriptor
        assert after_chat_descriptor.plugin_snapshot_id == first_descriptor.plugin_snapshot_id
        assert after_chat_descriptor.identity == first_descriptor.identity
        assert after_chat_descriptor.dimensions == first_descriptor.dimensions

        await _replace_driver_with_same_definition(
            graph,
            "driver-b",
            name="driver-b-replacement-for-embedding",
        )
        async with state.embeddings.bind() as after_embedding_replacement:
            after_embedding_descriptor = after_embedding_replacement.descriptor
            replacement_result = await after_embedding_replacement.embed(("three",))
        assert after_embedding_descriptor.plugin_snapshot_id != first_descriptor.plugin_snapshot_id
        assert after_embedding_descriptor.identity == first_descriptor.identity
        assert after_embedding_descriptor.dimensions == first_descriptor.dimensions
        assert replacement_result.vectors == ((1.0, 2.0),)
        assert graph["credential_reads"].count(("driver-b", "embedding-connection")) >= 2
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_driver_reactivation_changes_registration_identity_without_revision_change(tmp_path):
    """A hard dependency replacement reloads the same driver Fiber and Context."""

    from agent.plugin_composition import PluginRuntime

    graph = await _mount_model_driver_graph(
        tmp_path,
        with_live_models=True,
        with_driver_dependency=True,
    )
    try:
        await _configure_live_models(graph)
        state = graph["state"]
        async with state.chat_models.execution() as execution:
            old_descriptor = execution.chat("agent").descriptor
        driver_fiber = graph["driver_fibers"]["driver-a"]
        old_context = driver_fiber.context
        dependency = graph["driver_dependency_fiber"]
        unloading = asyncio.create_task(dependency.dispose())
        await graph["consumer_cleanup_started"].wait()
        graph["release_consumer"].set()
        await unloading

        async def apply_dependency(ctx):
            await ctx.provide(graph["driver_dependency"], object())

        await graph["root"].mount(
            apply_dependency,
            name="driver-dependency",
            runtime=PluginRuntime(
                "driver-dependency",
                "driver-dependency:g2",
                tmp_path,
                tmp_path / "driver-dependency-data-2",
                tmp_path,
                {},
            ),
        )
        assert graph["driver_fibers"]["driver-a"] is driver_fiber
        assert driver_fiber.context is not old_context
        async with state.chat_models.execution() as execution:
            new_descriptor = execution.chat("agent").descriptor
        assert state.store.read_snapshot().revision == 6
        assert new_descriptor.plugin_snapshot_id != old_descriptor.plugin_snapshot_id
        assert new_descriptor.binding_id != old_descriptor.binding_id
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_auth_start_unload_cleanup_preserves_callback_state(tmp_path):
    """Driver unload during start retains returned state for registration cleanup."""

    from agent.plugin_composition import FiberState
    from plugins.models.settings import StartConnectionAuth

    graph = await _mount_model_driver_graph(tmp_path)
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        start_task = asyncio.create_task(
            graph["state"]._start_auth(
                StartConnectionAuth("driver-a", "auth-start", {})
            )
        )
        await graph["auth_started"].wait()
        unloading = asyncio.create_task(graph["driver_fibers"]["driver-a"].dispose())
        await graph["consumer_cleanup_started"].wait()
        assert graph["driver_fibers"]["driver-a"].state is FiberState.UNLOADING
        graph["release_auth"].set()
        await graph["auth_returned"].wait()
        graph["release_auth_return"].set()
        with pytest.raises(ValueError, match="auth attempt 已取消"):
            await start_task
        graph["release_consumer"].set()
        await unloading
        assert not graph["driver_fibers"]["driver-a"]._in_flight_calls
        assert graph["cancel_states"] == [{"step": 1}]
        assert not graph["state"]._auth_attempts
        assert graph["state"].store.read_snapshot() == before
    finally:
        graph["release_auth"].set()
        graph["release_auth_return"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_idle_auth_attempt_does_not_hold_driver_owner_during_dispose(tmp_path):
    """A pending idle attempt is cancelled by registration cleanup after consumers drain."""

    from agent.plugin_composition import FiberState

    graph = await _mount_model_driver_graph(tmp_path)
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        receipt = await _apply_graph_start_auth(graph, "auth-idle")
        assert receipt.status == "pending"
        unloading = asyncio.create_task(graph["driver_fibers"]["driver-a"].dispose())
        await graph["consumer_cleanup_started"].wait()
        assert graph["driver_fibers"]["driver-a"].state is FiberState.UNLOADING
        graph["release_consumer"].set()
        await unloading
        assert graph["cancel_states"] == [{"step": 1}]
        assert graph["auth_callback_events"][-1] == (
            "driver-a",
            "cancel",
            {"step": 1},
        )
        assert not graph["state"]._auth_attempts
        assert graph["state"].store.read_snapshot() == before
    finally:
        graph["release_auth"].set()
        graph["release_auth_return"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_failed_auth_cleanup_keeps_old_effect_separate_from_new_registration(tmp_path):
    """A failed A cleanup is retried on A while a same-id B registration stays live."""

    from agent.plugin_composition import FiberState, MODEL_DRIVERS, ModelDriverDefinition, PluginRuntime
    from plugins.models.settings import CancelConnectionAuth, StartConnectionAuth

    graph = await _mount_model_driver_graph(tmp_path)
    old_fiber = None
    old_effect = None
    replacement = None
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        started = await _apply_graph_start_auth(graph, "auth-old")
        attempt_id = started.attempt_id
        assert attempt_id is not None
        old_fiber = graph["driver_fibers"]["driver-a"]
        old_effect = graph["driver_registration_effects"]["driver-a"]
        graph["cancel_error"][0] = RuntimeError("old cancel failed")
        unloading = asyncio.create_task(old_fiber.dispose())
        await graph["consumer_cleanup_started"].wait()
        graph["release_consumer"].set()
        with pytest.raises(BaseExceptionGroup) as cleanup_error:
            await unloading
        assert len(cleanup_error.value.exceptions) == 1
        failure = cleanup_error.value.exceptions[0]
        assert isinstance(failure, RuntimeError)
        assert str(failure) == "old cancel failed"
        assert old_effect in old_fiber.effects
        assert attempt_id in graph["state"]._auth_attempts

        b_events = []
        b_contexts = []
        b_effects = []

        async def b_open(descriptor, credential):
            b_events.append(("open", descriptor.connection_id))
            await credential.read()
            return graph["make_live_connection"]("driver-a", "B", descriptor)

        async def b_start(_input):
            b_events.append(("start", "B"))
            return {"state": {"tag": "B"}, "challenge": {"kind": "B"}}

        async def b_cancel(state):
            b_events.append(("cancel", dict(state)))
            return dict(state)

        b_definition = ModelDriverDefinition(
            "driver-a",
            "test-b",
            b_open,
            start_auth=b_start,
            cancel_auth=b_cancel,
        )

        async def apply_b(ctx):
            b_contexts.append(ctx)
            b_effects.append(await ctx.require(MODEL_DRIVERS).register(ctx, b_definition))

        replacement = await graph["root"].mount(
            apply_b,
            name="driver-a-auth-replacement",
            inject=(MODEL_DRIVERS,),
            runtime=PluginRuntime(
                "driver-a-auth-replacement",
                "driver-a-auth-replacement:g1",
                tmp_path,
                tmp_path / "driver-a-auth-replacement-data",
                tmp_path,
                {},
            ),
        )
        graph["cancel_error"][0] = None
        await old_effect.aclose()
        assert graph["cancel_states"] == [{"step": 1}, {"step": 1}]
        assert attempt_id not in graph["state"]._auth_attempts
        assert old_effect not in old_fiber.effects
        assert graph["state"]._registrations["driver-a"].context is b_contexts[0]
        assert b_effects[0] in replacement.effects
        await old_fiber.dispose()
        assert b_effects[0] in replacement.effects

        b_started = await graph["state"].settings.apply(
            StartConnectionAuth("driver-a", "auth-new", {})
        )
        b_cancelled = await graph["state"].settings.apply(
            CancelConnectionAuth(b_started.attempt_id or "")
        )
        assert b_cancelled.status == "cancelled"
        assert ("cancel", {"tag": "B"}) in b_events
        assert graph["state"].store.read_snapshot() == before
    finally:
        graph["cancel_error"][0] = None
        graph["release_auth"].set()
        graph["release_auth_return"].set()
        graph["release_consumer"].set()
        if old_effect is not None and old_fiber is not None and old_effect in old_fiber.effects:
            await old_effect.aclose()
        if old_fiber is not None and old_fiber.state is not FiberState.DISPOSED:
            await old_fiber.dispose()
        if replacement is not None and replacement.state is not FiberState.DISPOSED:
            await replacement.dispose()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_auth_expiry_waits_for_pending_finish_and_cancels_new_state(
    tmp_path, monkeypatch,
):
    """The real expiry task waits on finish.lock and cancels its published state."""

    from plugins.models.settings import FinishConnectionAuth

    graph = await _mount_model_driver_graph(tmp_path)
    original_sleep = asyncio.sleep
    timer_entered = asyncio.Event()
    release_timer = asyncio.Event()
    timer_fired = asyncio.Event()

    async def timer_sleep(delay, *args, **kwargs):
        if delay == 15 * 60 and not timer_entered.is_set():
            timer_entered.set()
            await release_timer.wait()
            timer_fired.set()
            return
        return await original_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", timer_sleep)
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        started = await _apply_graph_start_auth(graph, "auth-expiry")
        attempt_id = started.attempt_id or ""
        await timer_entered.wait()
        graph["auth_finish_result"][0] = {
            "status": "pending",
            "state": {"step": 2},
            "challenge": {"kind": "next"},
        }
        finish_task = asyncio.create_task(
            graph["state"].settings.apply(
                FinishConnectionAuth(0, attempt_id)
            )
        )
        await graph["finish_started"].wait()
        attempt = graph["state"]._auth_attempts[attempt_id]
        expiry_task = attempt.expiry_task
        assert expiry_task is not None
        release_timer.set()
        await timer_fired.wait()
        assert attempt.cancelled
        graph["release_finish"].set()
        await graph["finish_returned"].wait()
        graph["release_finish_return"].set()
        with pytest.raises(ValueError, match="auth attempt 已取消"):
            await finish_task
        await expiry_task
        assert graph["cancel_states"] == [{"step": 2}]
        assert graph["auth_callback_events"][-1] == (
            "driver-a",
            "cancel",
            {"step": 2},
        )
        assert not graph["state"]._auth_attempts
        assert graph["state"].store.read_snapshot() == before
    finally:
        graph["release_auth"].set()
        graph["release_auth_return"].set()
        graph["release_finish"].set()
        graph["release_finish_return"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_auth_pending_cancellation_keeps_new_state_before_live_check(tmp_path):
    """A pending finish publishes its new state before cancellation observes it."""

    from plugins.models.settings import CancelConnectionAuth, FinishConnectionAuth

    graph = await _mount_model_driver_graph(tmp_path)
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        started = await _start_graph_auth(graph, "auth-pending")
        graph["auth_finish_result"][0] = {
            "status": "pending",
            "state": {"step": 2},
            "challenge": {"kind": "next"},
        }
        finish_task = asyncio.create_task(
            graph["state"]._finish_auth(
                FinishConnectionAuth(0, started.attempt_id or "")
            )
        )
        await graph["finish_started"].wait()
        cancel_started = asyncio.Event()

        async def cancel_from_settings():
            cancel_started.set()
            return await graph["state"].settings.apply(
                CancelConnectionAuth(started.attempt_id or "")
            )

        cancel_task = asyncio.create_task(cancel_from_settings())
        await cancel_started.wait()
        graph["release_finish"].set()
        await graph["finish_returned"].wait()
        graph["release_finish_return"].set()
        with pytest.raises(ValueError, match="auth attempt 已取消"):
            await finish_task
        receipt = await cancel_task
        assert receipt.status == "cancelled"
        assert graph["cancel_states"] == [{"step": 2}]
        assert graph["state"].store.read_snapshot() == before
    finally:
        graph["release_auth"].set()
        graph["release_auth_return"].set()
        graph["release_finish"].set()
        graph["release_finish_return"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_auth_close_failure_happens_before_store_cas(tmp_path):
    """A finish close failure retains the attempt and leaves SQLite unchanged."""

    from plugins.models.settings import FinishConnectionAuth

    graph = await _mount_model_driver_graph(tmp_path)
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        started = await _start_graph_auth(graph, "auth-close")
        graph["fail_close_ids"].add("auth-close")
        finish_task = asyncio.create_task(
            graph["state"]._finish_auth(
                FinishConnectionAuth(0, started.attempt_id or "")
            )
        )
        await graph["finish_started"].wait()
        graph["release_finish"].set()
        await graph["finish_returned"].wait()
        graph["release_finish_return"].set()
        with pytest.raises(LookupError, match="auth-close close failed"):
            await finish_task
        assert graph["state"].store.read_snapshot() == before
        assert graph["close_snapshot_states"] == [before]
        assert started.attempt_id in graph["state"]._auth_attempts
        assert graph["close_events"] == ["auth-close"]
    finally:
        graph["fail_close_ids"].discard("auth-close")
        graph["release_auth"].set()
        graph["release_auth_return"].set()
        graph["release_finish"].set()
        graph["release_finish_return"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_create_connection_with_model_probes_and_opens_once_before_cas(tmp_path):
    """The new-connection probe, open, bind, close, and CAS share one owner scope."""

    from agent.plugin_composition import CapabilitySources, ModelCapabilities, ModelKind
    from plugins.models.settings import AddConnection, AddModel, CreateConnectionWithModel

    graph = await _mount_model_driver_graph(
        tmp_path,
        with_driver_b_consumer=True,
    )
    unloading = None
    try:
        before = graph["state"].store.read_snapshot()
        assert before is not None
        connection = AddConnection(
            expected_revision=0,
            connection_id="created",
            name="created",
            driver_id="driver-b",
            endpoint="https://example.invalid",
            auth_identity="test",
            credential={"token": "created"},
        )
        model = AddModel(
            expected_revision=0,
            model_id="created-model",
            connection_id="created",
            kind=ModelKind.CHAT,
            model="created-model",
            capabilities=ModelCapabilities(),
            capability_sources=CapabilitySources(),
        )
        operation = asyncio.create_task(
            graph["state"].apply_change(
                CreateConnectionWithModel(connection, model)
            )
        )
        await graph["probe_started"].wait()
        unloading = asyncio.create_task(
            graph["driver_fibers"]["driver-b"].dispose()
        )
        await graph["driver_b_consumer_cleanup_started"].wait()
        from agent.plugin_composition import FiberState

        assert graph["driver_fibers"]["driver-b"].state is FiberState.UNLOADING
        graph["release_probe"].set()
        receipt = await operation
        assert receipt.status == "committed"
        assert graph["probe_calls"] == ["created"]
        assert graph["open_calls"] == ["created"]
        assert graph["bind_chat_calls"] == ["created"]
        assert graph["close_events"] == ["created"]
        assert graph["close_snapshot_states"] == [before]
        snapshot = graph["state"].store.read_snapshot()
        assert snapshot is not None
        assert snapshot.connections["created"].driver_id == "driver-b"
        assert snapshot.models["created-model"].connection_id == "created"
        assert snapshot.revision == before.revision + 1
    finally:
        graph["release_probe"].set()
        graph["release_driver_b_consumer"].set()
        graph["release_consumer"].set()
        if unloading is not None:
            await unloading
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_driver_scope_normal_exit_releases_owner_and_connection(tmp_path):
    """A successful scope closes its connection before releasing owner calls."""

    from plugins.models.state import _driver_scope

    graph = await _mount_model_driver_graph(tmp_path)
    connection = _model_test_connection("normal", "driver-a")
    graph["release_open"].set()
    try:
        async with graph["model_context"].runtime_scope():
            async with _driver_scope(graph["state"], (connection,)) as opened:
                await graph["state"]._open_driver(connection, opened)
        assert graph["close_events"] == ["normal"]
        assert not graph["driver_fibers"]["driver-a"]._in_flight_calls
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
async def test_second_owner_admission_failure_does_not_hold_first_scope(tmp_path):
    """A later owner admission failure releases an earlier admitted owner."""

    from agent.plugin_composition import CompositionError, FiberState
    from plugins.models.state import _driver_scope

    graph = await _mount_model_driver_graph(tmp_path)
    first = _model_test_connection("first-admission", "driver-b")
    second = _model_test_connection("second-admission", "driver-a")
    selected = graph["state"]._select_driver_records((first, second))
    graph["release_open"].set()
    unloading = asyncio.create_task(graph["driver_fibers"]["driver-a"].dispose())
    try:
        await graph["consumer_cleanup_started"].wait()
        assert graph["driver_fibers"]["driver-a"].state is FiberState.UNLOADING
        with pytest.raises(CompositionError) as excinfo:
            async with graph["model_context"].runtime_scope():
                async with _driver_scope(
                    graph["state"],
                    (first, second),
                    selected=selected,
                ):
                    raise AssertionError("second owner admission must fail")
        assert excinfo.value.code == "OWNER_UNAVAILABLE"
        assert not graph["driver_fibers"]["driver-b"]._in_flight_calls
        graph["release_consumer"].set()
        await unloading
        assert not graph["driver_fibers"]["driver-a"]._in_flight_calls
    finally:
        graph["release_consumer"].set()
        graph["release_open"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_driver_scope_closes_all_connections_on_failure(tmp_path, failure):
    """Real selected connections survive owner UNLOADING and all close attempts run."""

    from agent.plugin_composition import FiberState
    from plugins.models.state import _driver_scope

    graph = await _mount_model_driver_graph(tmp_path)
    try:
        first = _model_test_connection("first", "driver-a")
        second = _model_test_connection("second", "driver-a")
        third = _model_test_connection("third", "driver-b")
        graph["fail_close_ids"].add("second")
        unrelated = graph["unrelated_fiber"]
        unrelated_before = (
            unrelated.context,
            unrelated._activation_token,
            unrelated.state,
            tuple(graph["unrelated_events"]),
        )

        async def use_connections():
            async with graph["model_context"].runtime_scope():
                async with _driver_scope(graph["state"], (first, second, third)) as opened:
                    await graph["state"]._open_driver(first, opened)
                    await graph["state"]._open_driver(second, opened)
                    await graph["state"]._open_driver(third, opened)
                    raise failure()

        operation = asyncio.create_task(use_connections())
        await graph["open_started"].wait()
        unloading = asyncio.create_task(graph["driver_fibers"]["driver-a"].dispose())
        await graph["consumer_cleanup_started"].wait()
        assert graph["driver_fibers"]["driver-a"].state is FiberState.UNLOADING
        async with unrelated.context.runtime_scope():
            unrelated_during = (
                unrelated.context,
                unrelated._activation_token,
                unrelated.state,
                tuple(graph["unrelated_events"]),
            )
        assert unrelated_during == unrelated_before
        graph["release_open"].set()
        with pytest.raises(LookupError, match="second close failed"):
            await operation
        assert graph["close_events"] == ["third", "second", "first"]
        assert not graph["driver_fibers"]["driver-a"]._in_flight_calls
        assert not graph["driver_fibers"]["driver-b"]._in_flight_calls

        graph["fail_close_ids"].remove("second")
        failed_effect = next(
            effect
            for effect in graph["driver_fibers"]["driver-a"].effects
            if effect.label == "model-connection:second"
        )
        await failed_effect.aclose()
        assert graph["close_events"] == ["third", "second", "first", "second"]
        assert not any(
            effect.label == "model-connection:second"
            for effect in graph["driver_fibers"]["driver-a"].effects
        )
        async with unrelated.context.runtime_scope():
            unrelated_after = (
                unrelated.context,
                unrelated._activation_token,
                unrelated.state,
                tuple(graph["unrelated_events"]),
            )
        assert unrelated_after == unrelated_before
        graph["release_consumer"].set()
        await unloading
    finally:
        graph["release_open"].set()
        graph["release_close"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_connection", ["first", "second"])
async def test_driver_scope_open_failure_leaves_no_connection_effect(
    tmp_path, failed_connection,
):
    """A first or later open failure leaves no outer connection residue."""

    from plugins.models.state import _driver_scope

    graph = await _mount_model_driver_graph(tmp_path)
    try:
        first = _model_test_connection("first", "driver-a")
        second = _model_test_connection("second", "driver-b")
        graph["fail_open_ids"].add(failed_connection)
        graph["release_open"].set()
        with pytest.raises(RuntimeError, match=f"{failed_connection} open failed"):
            async with graph["model_context"].runtime_scope():
                async with _driver_scope(graph["state"], (first, second)) as opened:
                    await graph["state"]._open_driver(first, opened)
                    await graph["state"]._open_driver(second, opened)
        assert graph["close_events"] == ([] if failed_connection == "first" else ["first"])
        for driver_fiber in graph["driver_fibers"].values():
            assert not any(
                effect.label.startswith("model-connection:")
                for effect in driver_fiber.effects
            )
            assert not driver_fiber._in_flight_calls
    finally:
        graph["release_open"].set()
        graph["release_close"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["new", "sync"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_discover_callbacks_hold_owner_until_success_error_or_cancel(
    tmp_path, consumer, outcome,
):
    """Both discovery consumers drain only after the admitted callback settles."""

    from agent.plugin_composition import DriverUnavailableError, FiberState
    from plugins.models.settings import AddConnection, SyncModels
    graph = await _mount_model_driver_graph(tmp_path)
    command = AddConnection(
        0,
        "first",
        "First",
        "driver-a",
        "https://example.invalid",
        "test",
        {"token": "fixture"},
    )
    if consumer == "sync":
        graph["state"].store.add_connection(command)
    if outcome == "error":
        graph["discover_error"][0] = RuntimeError("discover failed")
    try:
        async def run_discover():
            async with graph["model_context"].runtime_scope():
                if consumer == "new":
                    await graph["state"]._discover_new_connection(command)
                else:
                    await graph["state"]._sync_models(SyncModels(1, "first"))

        operation = asyncio.create_task(run_discover())
        await graph["discover_started"].wait()
        unloading = asyncio.create_task(graph["driver_fibers"]["driver-a"].dispose())
        await graph["consumer_cleanup_started"].wait()
        assert graph["driver_fibers"]["driver-a"].state is FiberState.UNLOADING
        if outcome == "cancel":
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            graph["release_discover"].set()
            if outcome == "error":
                with pytest.raises(RuntimeError, match="discover failed"):
                    await operation
            else:
                await operation
        assert graph["discover_calls"] == ["first"]
        assert not graph["driver_fibers"]["driver-a"]._in_flight_calls
        graph["release_consumer"].set()
        await unloading
        with pytest.raises(DriverUnavailableError):
            async with graph["model_context"].runtime_scope():
                await graph["state"]._discover_new_connection(command)
    finally:
        graph["release_discover"].set()
        graph["release_consumer"].set()
        graph["release_open"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True])
async def test_driver_close_finishes_before_releasing_lease_after_repeated_cancel(
    tmp_path, close_fails,
):
    """Repeated cancellation cannot cut off Effect cleanup or owner release."""

    from plugins.models.state import _driver_scope

    graph = await _mount_model_driver_graph(tmp_path, block_close=True)
    entered = asyncio.Event()
    events = []
    connection = _model_test_connection("test", "driver-a")
    try:
        graph["release_open"].set()
        if close_fails:
            graph["fail_close_ids"].add("test")

        async def run():
            try:
                async with graph["model_context"].runtime_scope():
                    async with _driver_scope(graph["state"], (connection,)) as opened:
                        await graph["state"]._open_driver(connection, opened)
                        entered.set()
                        await asyncio.Event().wait()
            finally:
                events.append("lease released")

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        await graph["close_started"].wait()
        task.cancel()
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        assert not task.done()
        assert events == []
        graph["release_close"].set()
        with pytest.raises(LookupError if close_fails else asyncio.CancelledError):
            await task
        assert events == ["lease released"]
        assert graph["close_events"] == ["test"]
        assert not graph["driver_fibers"]["driver-a"]._in_flight_calls
        if close_fails:
            graph["fail_close_ids"].remove("test")
            failed_effect = next(
                effect
                for effect in graph["driver_fibers"]["driver-a"].effects
                if effect.label == "model-connection:test"
            )
            await failed_effect.aclose()
            assert graph["close_events"] == ["test", "test"]
            assert not any(
                effect.label == "model-connection:test"
                for effect in graph["driver_fibers"]["driver-a"].effects
            )
    finally:
        graph["release_open"].set()
        graph["release_close"].set()
        graph["release_consumer"].set()
        await graph["root"].dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True])
async def test_public_driver_aclose_finishes_after_repeated_cancel(close_fails):
    """Public DriverConnection.aclose preserves cancellation and close errors."""

    from agent.plugin_composition import DriverConnection

    started = asyncio.Event()
    closing = asyncio.Event()
    release_close = asyncio.Event()
    events = []

    async def close():
        closing.set()
        await release_close.wait()
        events.append("closed")
        if close_fails:
            raise LookupError("public close failed")

    def no_chat(*_args) -> DriverChatModel:
        raise AssertionError("close test must not bind a chat model")

    def no_embedding(*_args) -> DriverEmbeddingModel:
        raise AssertionError("close test must not bind an embedding model")

    connection = DriverConnection(no_chat, no_embedding, close=close)

    async def run():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await connection.aclose()
            events.append("lease released")

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    await closing.wait()
    task.cancel()
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    await barrier.wait()
    assert not task.done()
    assert events == []
    release_close.set()
    with pytest.raises(LookupError if close_fails else asyncio.CancelledError):
        await task
    assert events == ["closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 500])
async def test_opencode_discovery_closes_temporary_client(status, monkeypatch):
    """目录发现成功或失败都会归还临时 HTTP 资源。"""
    import httpx
    from agent.plugin_composition import DriverConnectionDescriptor, ModelError
    from plugins.opencode_go import driver

    def respond(request):
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer fixture"
        return httpx.Response(status, json={"data": [{"id": "fixture"}]})

    client = httpx.AsyncClient(base_url="http://local.test/v1", transport=httpx.MockTransport(respond))
    monkeypatch.setattr(driver, "_client", lambda _connection: client)

    async def cli_catalog():
        return {}

    monkeypatch.setattr(driver, "_load_cli_catalog", cli_catalog)

    class Credential:
        connection_id = "local"
        auth_identity = "fixture"

        async def read(self):
            return {"api_key": "fixture"}

        async def refresh(self, payload):
            raise AssertionError("目录发现不刷新凭据")

        @asynccontextmanager
        async def exclusive(self):
            yield

    descriptor = DriverConnectionDescriptor(
        "local", "local", "opencode-go", "http://local.test/v1", "fixture", {"max_retries": 0},
    )
    try:
        if status == 200:
            models = await driver.definition().discover(descriptor, Credential())
            assert [model.model for model in models] == ["fixture"]
        else:
            with pytest.raises(ModelError):
                await driver.definition().discover(descriptor, Credential())
        assert client.is_closed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_driver_config_path_validates_max_attempts_and_keyed_single_attempt():
    """真实 open/bind 配置路径：max_attempts 经 _connection_config 校验；带
    request_key 的 accounted 调用 driver 恒单次，未记账直调保留有界重试。"""
    import httpx
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from plugins.openai_compatible import driver
    from tests.model_plugin_fakes import BoundChatModelFake

    hits = 0

    def respond(request):
        nonlocal hits
        assert request.url.path == "/v1/chat/completions"
        hits += 1
        # 429 是对本请求的明确限流拒绝（含 Retry-After），属于正面证据；
        # 5xx 不能证明后端未处理，不能用于驱动重试路径。
        return httpx.Response(
            429, headers={"retry-after": "0"},
            json={"error": {"message": "rate limited"}},
        )

    client = httpx.AsyncClient(
        base_url="http://local.test/v1", transport=httpx.MockTransport(respond)
    )
    original_client = driver._client
    driver._client = lambda _connection: client

    class Credential:
        connection_id = "local"
        auth_identity = "fixture"

        async def read(self):
            return {"api_key": "fixture"}

        async def refresh(self, payload):
            raise AssertionError("本测试不刷新凭据")

        @asynccontextmanager
        async def exclusive(self):
            yield

    try:
        # 非法 max_attempts 在校验边界如实报错，不静默回 1。
        bad = DriverConnectionDescriptor(
            "local", "local", "openai-compatible", "http://local.test/v1",
            "fixture", {"max_attempts": 0},
        )
        with pytest.raises(ValueError, match="max_attempts"):
            await driver.definition().open(bad, Credential())
        with pytest.raises(ValueError, match="max_attempts"):
            await driver.definition().open(
                replace(bad, config={"max_attempts": "2"}), Credential()
            )
        connection = await driver.definition().open(
            DriverConnectionDescriptor(
                "local", "local", "openai-compatible", "http://local.test/v1",
                "fixture", {"max_retries": 1, "max_attempts": 3},
            ),
            Credential(),
        )
        bound = connection.bind_chat(
            replace(BoundChatModelFake(object()).descriptor,
                    connection_id="local", driver_id="openai-compatible",
                    auth_identity="fixture", model="fixture"),
            {},
        )
        request = ModelRequest(({"role": "user", "content": "hi"},))
        # §6.3：生成调用恒为一次物理 attempt，未记账直调也无匿名重试后门；
        # 429 明确拒绝的重试预算只属 Models，driver 不得自行重发。
        with pytest.raises(Exception):
            await bound.complete(request)
        assert hits == 1
        # accounted 调用带 request_key：同样恒单次，重试预算只属 Models。
        hits = 0
        with pytest.raises(Exception):
            await bound.complete(replace(request, request_key="accounted"))
        assert hits == 1
        await connection.aclose()
    finally:
        driver._client = original_client
        await client.aclose()


def test_retry_budget_maps_legacy_and_rejects_invalid():
    """Models 边界集中解析：max_attempts 显式优先，max_retries 迁移为 N+1。"""
    from plugins.models.state import _retry_budget

    assert _retry_budget({}) == 1
    assert _retry_budget({"max_retries": 0}) == 1
    assert _retry_budget({"max_retries": 3}) == 4
    assert _retry_budget({"max_attempts": 2}) == 2
    assert _retry_budget({"max_attempts": 2, "max_retries": 9}) == 2
    for invalid in ({"max_attempts": 0}, {"max_attempts": "3"}, {"max_retries": -1}):
        with pytest.raises(ValueError):
            _retry_budget(invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("delta_event", [
    'data: {"type":"response.output_text.delta","delta":"part"}\n\n',
    'data: {"type":"response.reasoning_summary_text.delta","delta":"part"}\n\n',
    'data: {"type":"response.function_call_arguments.delta","item_id":"i","delta":"{\\""}\n\n',
])
async def test_partial_stream_failure_settles_durable_evidence_as_uncertain(
    tmp_path, observe, delta_event,
):
    """真实 codex driver → Models：协议层观察到 text/tool/reasoning 增量后
    response.incomplete(context_length_exceeded) 的结算必须携带部分输出证据；
    key_recovery 判 uncertain——不缩减、不重试、resume 不得重付。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import ContextLengthError
    from plugins.codex.driver import definition
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    requests = []

    async def respond(request):
        requests.append(request)
        # 协议层先交付一类真实增量，再以 failed 报容量失败。
        payload = (
            delta_event
            + 'data: {"type":"response.failed","response":{'
            + '"status":"failed","error":{"code":"context_length_exceeded",'
            + '"message":"too long"}}}\n\n'
        )
        return web.Response(text=payload, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/responses", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        async def read(self):
            return {"driver": "codex", "api_key": "k", "access_token": "k",
                    "account_id": "test", "expires_at": "2099-01-01T00:00:00+00:00"}

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id="codex",
        connection_id="test-connection", model="fixture",
    )
    driver = await definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", "codex",
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        cast(Any, Credential()),
    )
    try:
        bound = _BoundChat(descriptor, driver.bind_chat(descriptor, {}), store)
        async def preview(_delta):
            return None
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            on_delta=preview if observe else None,
            request_key="partial-key",
        )
        with pytest.raises(ContextLengthError) as failure:
            await bound.complete(request)
        # driver 证据透传到异常对象：协议层确实观察到了部分输出。
        assert failure.value.response_delta_seen
        records = store.calls_for_key("partial-key")
        assert len(records) == 1
        assert records[0]["failure"] == "ContextLengthError"
        assert records[0]["partial_response"] == 1, "部分输出证据必须耐久入账"
        assert records[0]["next_attempt_at"] is None, "部分输出后不得安排自动重试"
        assert bound.key_recovery("partial-key") == "uncertain"
        # 同 key 终结：再次调用不得发送第二个 provider 请求。
        with pytest.raises(Exception):
            await bound.complete(request)
        assert len(requests) == 1
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_stream_failure_without_deltas_is_uncertain(tmp_path):
    """HTTP 200 流内 response.failed(context_length_exceeded) 即使零 delta
    也不证明请求未被处理——send_evidence 缺失，key_recovery 判 uncertain，
    不得自动重试/缩减/换 key。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import ContextLengthError
    from plugins.codex.driver import definition
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    async def respond(request):
        return web.Response(
            text='data: {"type":"response.failed","response":{'
                 '"status":"failed","error":{"code":"context_length_exceeded",'
                 '"message":"too long"}}}\n\n',
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/responses", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        async def read(self):
            return {"driver": "codex", "api_key": "k", "access_token": "k",
                    "account_id": "test", "expires_at": "2099-01-01T00:00:00+00:00"}

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id="codex",
        connection_id="test-connection", model="fixture",
    )
    driver = await definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", "codex",
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        cast(Any, Credential()),
    )
    try:
        bound = _BoundChat(descriptor, driver.bind_chat(descriptor, {}), store)
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            request_key="clean-key",
        )
        with pytest.raises(ContextLengthError) as failure:
            await bound.complete(request)
        assert not getattr(failure.value, "response_delta_seen", False)
        records = store.calls_for_key("clean-key")
        assert records[0]["partial_response"] == 0
        assert records[0]["send_evidence"] is None, "流内失败不携带发送证据"
        assert records[0]["next_attempt_at"] is None
        assert bound.key_recovery("clean-key") == "uncertain"
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_http_status_rejection_is_provable_send_evidence(tmp_path):
    """正例：provider 在流开始前以 HTTP 错误状态明确拒绝——send_evidence
    落账为 rejected，ContextLengthError 判 rejected 保留有界缩减资格。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import ContextLengthError
    from plugins.codex.driver import definition
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    async def respond(request):
        return web.Response(
            status=400, text='{"error":{"code":"context_length_exceeded"}}'
        )

    app = web.Application()
    app.router.add_post("/v1/responses", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        async def read(self):
            return {"driver": "codex", "api_key": "k", "access_token": "k",
                    "account_id": "test", "expires_at": "2099-01-01T00:00:00+00:00"}

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id="codex",
        connection_id="test-connection", model="fixture",
    )
    driver = await definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", "codex",
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        cast(Any, Credential()),
    )
    try:
        bound = _BoundChat(descriptor, driver.bind_chat(descriptor, {}), store)
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            request_key="http-rejected-key",
        )
        with pytest.raises(ContextLengthError) as failure:
            await bound.complete(request)
        assert failure.value.send_evidence == "rejected"
        records = store.calls_for_key("http-rejected-key")
        assert records[0]["send_evidence"] == "rejected"
        assert bound.key_recovery("http-rejected-key") == "rejected"
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_connect_failure_is_unsent_evidence(tmp_path):
    """正例：连接建立失败可证明请求未发出——send_evidence=unsent 落账，
    key_recovery 判 answered（可证明失败允许真实 resume 开新准备）。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import TransportError
    from plugins.codex.driver import definition
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    # 绑定后立即关闭的端口：TCP connect 确定性失败。
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        async def read(self):
            return {"driver": "codex", "api_key": "k", "access_token": "k",
                    "account_id": "test", "expires_at": "2099-01-01T00:00:00+00:00"}

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id="codex",
        connection_id="test-connection", model="fixture",
    )
    driver = await definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", "codex",
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        cast(Any, Credential()),
    )
    try:
        bound = _BoundChat(descriptor, driver.bind_chat(descriptor, {}), store)
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            request_key="unsent-key",
        )
        with pytest.raises(TransportError):
            await bound.complete(request)
        records = store.calls_for_key("unsent-key")
        assert records[0]["send_evidence"] == "unsent"
        assert bound.key_recovery("unsent-key") == "answered"
    finally:
        await driver.aclose()


@pytest.mark.asyncio
async def test_zero_delta_stream_eof_never_replays_same_request(tmp_path):
    """协调者复现：真实 openai driver + HTTP200 SSE 无 delta 直接 EOF——
    provider 已接收请求，零 delta 不证明未处理。requests 与耐久 attempt
    都必须保持 1，无自动重试计划，key_recovery 判 uncertain。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import TransportError
    from plugins.openai_compatible.driver import definition
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    requests = []

    async def respond(request):
        requests.append(request)
        return web.Response(
            text=": accepted, processing\n\n", content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        async def read(self):
            return {"api_key": "k", "access_token": "k"}

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id="openai-compatible",
        connection_id="test-connection", model="fixture",
    )
    driver = await definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", "openai-compatible",
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        cast(Any, Credential()),
    )
    try:
        bound = _BoundChat(
            descriptor, driver.bind_chat(descriptor, {}), store, max_attempts=2
        )

        async def preview(_delta):
            return None

        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            on_delta=preview,
            request_key="zero-delta-key",
        )
        with pytest.raises(TransportError):
            await bound.complete(request)
        assert len(requests) == 1, "HTTP200_ACCEPTED_ZERO_DELTA_DISCONNECT 不得重发"
        records = store.calls_for_key("zero-delta-key")
        assert len(records) == 1, "耐久账目不得出现第二个 attempt"
        assert records[0]["send_evidence"] is None
        assert records[0]["next_attempt_at"] is None
        assert bound.key_recovery("zero-delta-key") == "uncertain"
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503, 504])
@pytest.mark.parametrize("driver_name,path", [
    ("openai_compatible", "/v1/chat/completions"),
    ("opencode_go", "/v1/chat/completions"),
    ("codex", "/v1/responses"),
])
async def test_http_5xx_never_proves_request_unprocessed(
    tmp_path, driver_name, path, status,
):
    """三 driver 一致：5xx/网关错误不能证明后端未接收或未处理——不授
    send_evidence、不自动重试、key_recovery 判 uncertain。错误正文即使
    包含 context_length 文案也不得提升为安全容量拒绝。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import ModelError, TransportError
    from importlib import import_module
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    requests = []
    driver_id = driver_name.replace("_", "-")

    async def respond(request):
        requests.append(request)
        # 5xx 正文故意携带容量文案：状态与证据优先于诊断文案。
        return web.Response(
            status=status,
            text='{"error":{"message":"upstream context_length_exceeded"}}',
        )

    app = web.Application()
    app.router.add_post(path, respond)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        async def read(self):
            return {"driver": driver_name, "api_key": "k", "access_token": "k",
                    "account_id": "test", "expires_at": "2099-01-01T00:00:00+00:00"}

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id=driver_id,
        connection_id="test-connection", model="fixture",
    )
    driver = await import_module(f"plugins.{driver_name}.driver").definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", driver_id,
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        Credential(),
    )
    try:
        bound = _BoundChat(
            descriptor, driver.bind_chat(descriptor, {}), store, max_attempts=2
        )
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            request_key="gateway-key",
        )
        with pytest.raises(ModelError) as exc_info:
            await bound.complete(request)
        # status-first：5xx 正文携带 context_length 文案仍归 TransportError，
        # 不得提升为可证明的容量拒绝。
        assert type(exc_info.value) is TransportError
        assert len(requests) == 1, "5xx 不得自动重发同一请求"
        records = store.calls_for_key("gateway-key")
        assert len(records) == 1
        assert records[0]["send_evidence"] is None
        assert records[0]["next_attempt_at"] is None
        assert bound.key_recovery("gateway-key") == "uncertain"
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["recover", "budget_one", "always_401", "refresh_fails"])
async def test_codex_401_rotates_credential_without_hidden_second_post(tmp_path, mode):
    """真实 codex driver → Models：401 明确拒绝后凭据轮换属 auth owner，
    driver 在同一 complete 内不得再发模型 POST。预算>=2 时 Models 链发起
    第二 attempt：两条耐久记录、第二次 POST 用轮换后的凭据；预算 1、
    二次 401、轮换失败分别如实终结，无隐藏第三请求。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from agent.plugin_composition.models import AuthenticationError
    from plugins.codex.driver import definition
    from plugins.models.state import _BoundChat
    from plugins.models.store import ModelsStore
    from tests.model_plugin_fakes import BoundChatModelFake

    posts = []
    refreshes = []
    sse_ok = (
        'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
        'data: {"type":"response.completed","response":{}}\n\n'
    )

    async def respond(request):
        posts.append(request.headers.get("Authorization"))
        if mode == "always_401" or len(posts) == 1:
            return web.Response(status=401, text='{"error":{"message":"bad token"}}')
        return web.Response(text=sse_ok, content_type="text/event-stream")

    async def token(request):
        refreshes.append(await request.json())
        if mode == "refresh_fails":
            return web.Response(status=500, text="refresh down")
        return web.json_response({
            "access_token": f"rotated-{len(refreshes)}",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
        })

    app = web.Application()
    app.router.add_post("/v1/responses", respond)
    app.router.add_post("/oauth/token", token)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    class Credential:
        connection_id = "test-connection"
        auth_identity = "test"

        def __init__(self):
            self.current = {
                "driver": "codex",
                "access_token": "initial-token",
                "refresh_token": "refresh-token",
                "account_id": "test",
                "expires_at": "2099-01-01T00:00:00+00:00",
                "auth_base": f"http://127.0.0.1:{port}",
                "api_base": f"http://127.0.0.1:{port}/v1",
            }

        async def read(self):
            return dict(self.current)

        async def refresh(self, payload):
            self.current = dict(payload)

        @asynccontextmanager
        async def exclusive(self):
            yield

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    descriptor = replace(
        BoundChatModelFake(object()).descriptor, driver_id="codex",
        connection_id="test-connection", model="fixture",
    )
    driver = await definition().open(
        DriverConnectionDescriptor(
            "test-connection", "local", "codex",
            f"http://127.0.0.1:{port}/v1", "test", {},
        ),
        Credential(),
    )
    try:
        bound = _BoundChat(
            descriptor, driver.bind_chat(descriptor, {}), store,
            max_attempts=1 if mode == "budget_one" else 2,
        )
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            request_key="auth-key",
        )
        if mode == "recover":
            response = await bound.complete(request)
            assert response.content == "ok"
            # 两个模型 POST 各自由一条耐久 attempt 记账；第二请求使用轮换凭据。
            assert posts == ["Bearer initial-token", "Bearer rotated-1"]
            records = store.calls_for_key("auth-key")
            assert len(records) == 2
            assert records[0]["failure"] == "AuthenticationError"
            assert records[0]["send_evidence"] == "rejected"
            assert records[1]["state"] == "success"
            return
        with pytest.raises(AuthenticationError):
            await bound.complete(request)
        if mode == "budget_one":
            # 预算 1：driver 不得自行发第二模型请求。
            assert posts == ["Bearer initial-token"]
            assert len(refreshes) == 1, "凭据轮换仍属 auth owner 的正常能力"
        elif mode == "always_401":
            # 预算 2：第二 attempt 遭拒后终结，无隐藏第三请求。
            assert posts == ["Bearer initial-token", "Bearer rotated-1"]
        elif mode == "refresh_fails":
            # 轮换失败如实终结：单 POST，无可调度重试。
            assert posts == ["Bearer initial-token"]
        records = store.calls_for_key("auth-key")
        assert records[-1]["failure"] == "AuthenticationError"
        assert records[-1]["send_evidence"] == "rejected"
        assert records[-1]["next_attempt_at"] is None
        assert bound.key_recovery("auth-key") == "answered"
    finally:
        await driver.aclose()
        await runner.cleanup()
