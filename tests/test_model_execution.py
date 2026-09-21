"""真实 ModelsState execution 的代际、重试和 task 绑定合同。"""

from __future__ import annotations

import asyncio
import json
import socket
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from agent.config_models import Config
from agent.plugin_composition import (
    CHAT_MODELS,
    ModelRequest,
)
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from bootstrap import tools as bootstrap
from bootstrap.init_workspace import init_workspace
from core.net.http import SharedHttpResources


async def _model_command(control: RuntimeModelControl, payload: dict[str, object]) -> dict[str, object]:
    """Configure the installed Models owner through its public RPC boundary."""

    result = await control.invoke_rpc("models/command", payload)
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
        control = RuntimeModelControl(core.plugin_manager.snapshot_store)
        await _model_command(control, {
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
        await _model_command(control, {
            "type": "add_model",
            "expected_revision": 1,
            "model_id": "first",
            "connection_id": "local",
            "kind": "chat",
            "model": "first",
            "capabilities": capabilities,
            "capability_sources": {},
        })
        await _model_command(control, {
            "type": "set_default",
            "expected_revision": 2,
            "role": "default",
            "model_id": "first",
        })

        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            models_service = context.require(CHAT_MODELS)
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

                await _model_command(control, {
                    "type": "add_model",
                    "expected_revision": 3,
                    "model_id": "second",
                    "connection_id": "local",
                    "kind": "chat",
                    "model": "second",
                    "capabilities": capabilities,
                    "capability_sources": {},
                })
                await _model_command(control, {
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
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()
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
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
async def test_driver_close_finishes_before_releasing_lease_after_repeated_cancel(
    scoped, close_fails,
):
    """重复取消不能截断关闭，关闭失败也必须在归还租约前报告。"""
    from agent.plugin_composition import DriverConnection
    from plugins.models.state import _driver_scope

    entered = asyncio.Event()
    closing = asyncio.Event()
    finish_close = asyncio.Event()
    events = []

    async def close():
        closing.set()
        await finish_close.wait()
        events.append("closed")
        if close_fails:
            raise LookupError("close failed")

    def unused(*_args):
        raise AssertionError("此例只检查生命周期")

    driver = DriverConnection(unused, unused, close=close)

    async def run():
        try:
            if scoped:
                async with _driver_scope() as opened:
                    opened["test"] = driver
                    entered.set()
                    await asyncio.Event().wait()
            else:
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    await driver.aclose()
        finally:
            events.append("lease released")

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    await closing.wait()
    task.cancel()
    # 调度屏障让第二次取消先送达，再允许底层关闭完成。
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    await barrier.wait()
    assert not task.done()
    assert events == []
    finish_close.set()
    with pytest.raises(LookupError if close_fails else asyncio.CancelledError):
        await task
    assert events == ["closed", "lease released"]


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
        Credential(),
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
        Credential(),
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
        Credential(),
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
        Credential(),
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
        Credential(),
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
