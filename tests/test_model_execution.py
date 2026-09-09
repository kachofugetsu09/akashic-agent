"""真实 ModelsState execution 的代际、重试和 task 绑定合同。"""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from agent.config_models import Config
from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CapabilitySources,
    CHAT_MODELS,
    ModelCapabilities,
    ModelKind,
    ModelRole,
    ModelRequest,
    SetDefaultModel,
)
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from bootstrap import tools as bootstrap
from bootstrap.init_workspace import init_workspace
from core.net.http import SharedHttpResources


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
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        control = RuntimeModelControl(core.plugin_manager.snapshot_store)
        await control.apply(
            AddConnection(
                0, "local", "Local", "openai-compatible",
                f"http://127.0.0.1:{port}/v1", "fixture", {"api_key": "fixture"},
            )
        )
        capabilities = ModelCapabilities(
            context_window=32000,
            max_output_tokens=1024,
            supports_tool_calls=True,
            supported_reasoning_efforts=("low", "high"),
        )
        await control.apply(
            AddModel(1, "first", "local", ModelKind.CHAT, "first", capabilities, CapabilitySources())
        )
        await control.apply(SetDefaultModel(2, ModelRole.DEFAULT, "first"))

        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            models_service = context.require(CHAT_MODELS)
            async with models_service.execution() as first_execution:
                first_descriptor = first_execution.chat(ModelRole.AGENT).descriptor
                assert first_descriptor.model_id == "first"
                response = await first_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "first"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )
                assert response.content == "ok"
                assert calls[-1]["model"] == "first"

                async with models_service.execution() as nested:
                    assert nested is first_execution
                    assert nested.chat(ModelRole.AGENT).descriptor == first_descriptor

                await control.apply(
                    AddModel(3, "second", "local", ModelKind.CHAT, "second", capabilities, CapabilitySources())
                )
                await control.apply(SetDefaultModel(4, ModelRole.DEFAULT, "second"))

                # 已打开的 execution 固定旧 descriptor；默认切换只影响下一次 execution。
                async with models_service.execution() as still_first:
                    assert still_first is first_execution
                    assert still_first.chat(ModelRole.AGENT).descriptor == first_descriptor
                await first_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "retry"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )
                assert calls[-1]["model"] == "first"
                assert first_execution.chat(ModelRole.AGENT).descriptor == first_descriptor

                lease_count = snapshot.lease_count

                async def child_execution() -> None:
                    with pytest.raises(RuntimeError, match="不能由子 task 继承"):
                        async with models_service.execution():
                            raise AssertionError("子 task 不应取得 model execution")

                await asyncio.create_task(child_execution())
                assert snapshot.lease_count == lease_count
                assert len(calls) == 2
                assert transports[0] is transports[1]

            with pytest.raises(RuntimeError, match="连接已关闭"):
                await first_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(messages=({"role": "user", "content": "closed"},))
                )

            async with models_service.execution() as second_execution:
                second_descriptor = second_execution.chat(ModelRole.AGENT).descriptor
                assert second_descriptor.model_id == "second"
                assert second_descriptor.binding_id != first_descriptor.binding_id
                await second_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "second"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )

        assert [call["model"] for call in calls] == ["first", "first", "second"]
        assert transports[2] is not transports[0]
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()
        await runner.cleanup()


async def _noop_delta() -> None:
    """为真实流式 provider 请求提供最小 delta consumer。"""


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("driver_name", ["openai_compatible", "opencode_go", "codex"])
async def test_driver_reuses_socket_and_reads_rotated_credentials(streaming, driver_name, caplog):
    """同一连接的两次请求复用 socket，并使用各自读取到的凭据。"""
    from dataclasses import replace
    from agent.plugin_composition import DriverConnectionDescriptor
    from importlib import import_module
    definition = import_module(f"plugins.{driver_name}.driver").definition
    driver_id = driver_name.replace("_", "-")
    from tests.model_plugin_fakes import BoundChatModelFake

    seen = []
    caplog.set_level("DEBUG", logger="core.net.http")

    async def complete(request):
        seen.append((request.transport, request.headers["Authorization"]))
        assert "Cookie" not in request.headers
        if driver_name == "codex":
            return web.Response(
                text='data: {"type":"response.output_text.delta","delta":"ok"}\n\ndata: {"type":"response.completed","response":{}}\n\n',
                content_type="text/event-stream",
            )
        if streaming:
            return web.Response(
                text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
                content_type="text/event-stream",
            )
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
    descriptor = replace(BoundChatModelFake(object()).descriptor, driver_id=driver_id)
    driver = await definition().open(
        DriverConnectionDescriptor("test-connection", "local", driver_id,
                                   f"http://127.0.0.1:{port}/v1", "test", {}),
        credential,
    )
    try:
        bound = driver.bind_chat(descriptor, {})
        request = ModelRequest(
            messages=({"role": "user", "content": "hello"},),
            on_delta=(lambda _delta: _noop_delta()) if streaming else None,
        )
        assert (await bound.complete(request)).content == "ok"
        credential.token = "second"
        assert (await bound.complete(request)).content == "ok"
        # 负载可让尾流超过 10 ms；只有明确记录放弃复用时才允许新 socket。
        if seen[0][0] is not seen[1][0]:
            assert "尾流未结束，关闭连接" in caplog.text
        assert [item[1] for item in seen] == ["Bearer first", "Bearer second"]
        await driver.aclose()
        with pytest.raises(RuntimeError, match="连接已关闭"):
            await bound.complete(request)
    finally:
        await driver.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_done_does_not_wait_for_a_stalled_http_tail():
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
            result = await _consume_stream(response, lambda _delta: _noop_delta())
        assert result.content == "ok"
        assert waiting.is_set() and cancelled.is_set()
    finally:
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_driver_scope_closes_all_connections_on_failure(failure):
    """部分绑定失败或取消仍释放全部已打开连接，包括关闭自身报错的情况。"""
    from agent.plugin_composition import DriverConnection
    from plugins.models.state import _driver_scope

    closed = []

    async def first():
        closed.append("first")

    async def second():
        closed.append("second")
        raise LookupError("close failed")

    def unused(*_args):
        raise AssertionError("此例只检查生命周期")

    with pytest.raises(LookupError, match="close failed"):
        async with _driver_scope() as opened:
            opened["first"] = DriverConnection(unused, unused, close=first)
            opened["second"] = DriverConnection(unused, unused, close=second)
            raise failure()
    assert closed == ["second", "first"]


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
