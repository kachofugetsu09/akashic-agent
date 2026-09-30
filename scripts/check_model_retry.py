"""用本地 HTTP 对端和临时 Models 账本验证安全重试，不访问真实 provider。"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import threading
from unittest.mock import patch


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """为真实连接 probe 返回本地模型目录，不替换生产 probe。"""
        assert self.path.endswith("/models"), self.path
        body = json.dumps({"data": [{"id": "scenario"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """消费本地场景响应，并记录真实收到的 POST 次数。"""
        self.rfile.read(int(self.headers["Content-Length"]))
        status, delay = self.server.replies.popleft()
        self.server.received.append(status)
        value = (
            {"choices": [{"message": {"role": "assistant", "content": "local-result"},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
            if status == 200 else {"error": {"message": "scenario failure"}}
        )
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if delay is not None:
            self.send_header("Retry-After", str(delay))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


class Credential:
    connection_id = "scenario"
    auth_identity = "scenario"

    async def read(self) -> dict[str, str]:
        return {"api_key": "local-scenario"}


async def run(args: argparse.Namespace) -> dict:
    """逐项检查 provider 发送次数、耐久结算和重新打开账本后的行为。"""
    sys.path.insert(0, str(args.source))
    import httpx
    from agent.plugin_composition import (
        BoundModelDescriptor, CapabilitySources, ModelCapabilities, ModelError,
        ModelRequest, ModelUnavailableError,
    )
    from core.net.http import HttpClient
    from plugins.models.state import _BoundChat, _retry_budget
    from plugins.models.store import ModelsStore
    from plugins.openai_compatible import driver
    from agent.plugin_composition import CHAT_MODELS, MODEL_DRIVERS, CompositionRoot, ModelKind
    from plugins.models.settings import (
        MODEL_SETTINGS, AddConnection, AddModel, CreateConnectionWithModel, SetDefaultModel,
    )
    from plugins.models.state import ModelsState

    descriptor = BoundModelDescriptor(
        binding_id="scenario", plugin_snapshot_id="scenario", model_revision=0,
        model_id="scenario", connection_id="scenario", driver_id="openai-compatible",
        driver_contract_version="scenario", auth_identity="scenario", model="scenario",
        role="default", reasoning_effort=None, capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(), capability_digest="scenario",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.replies = deque()
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    http = HttpClient(lambda: httpx.AsyncClient(base_url=endpoint, trust_env=False))
    physical = driver._BoundChat(
        driver._ConnectionConfig(endpoint, 1, 1, 0, False),
        Credential(), descriptor, driver._ModelConfig(None, 16), http,
    )
    report = {"source": str(args.source), "checks": []}
    with tempfile.TemporaryDirectory(prefix="model-retry-check-") as temporary:
        root = Path(temporary)

        async def public_configuration():
            """从公开设置持久化空连接配置，再通过真实插件注册和 execution 绑定。"""
            composition = CompositionRoot("retry-public-scenario")
            store = ModelsStore(root / "public.db", root / "backups")
            states = []

            async def models(ctx):
                store.initialize()
                state = ModelsState(store, context=ctx)
                states.append(state)
                await ctx.effect(lambda: store.close, label="scenario-store")
                await ctx.provide(MODEL_DRIVERS, state.drivers)
                await ctx.provide(CHAT_MODELS, state.chat_models)
                await ctx.provide(MODEL_SETTINGS, state.settings)

            async def provider(ctx):
                await ctx.require(MODEL_DRIVERS).register(ctx, driver.definition())

            await composition.mount(models, name="models")
            try:
                await composition.mount(provider, name="openai-compatible", inject=(MODEL_DRIVERS,))
                state = states[0]
                server.replies = deque([(200, None)])
                server.received = []
                receipt = await state.settings.apply(CreateConnectionWithModel(
                    AddConnection(
                        expected_revision=0, connection_id="public", name="Local scenario",
                        driver_id=driver.definition().driver_id, endpoint=endpoint,
                        auth_identity="public", credential={"api_key": "local-scenario"},
                    ),
                    AddModel(
                        expected_revision=0, model_id="public", connection_id="public",
                        kind=ModelKind.CHAT, model="scenario",
                        capabilities=ModelCapabilities(context_window=8192, supports_tool_calls=True),
                        capability_sources=CapabilitySources(context_window="fixture"),
                    ),
                ))
                await state.settings.apply(SetDefaultModel(
                    expected_revision=receipt.revision, role="default", model_id="public",
                ))
                assert server.received == [200], "配置试算没有执行一次真实 POST"
                snapshot = store.read_snapshot()
                assert snapshot is not None and not snapshot.connections["public"].driver_config
                server.replies = deque([(429, 0), (200, None)])
                server.received = []
                async with state.chat_models.execution() as execution:
                    try:
                        response = await execution.chat("default").complete(ModelRequest(
                            [{"role": "user", "content": "scenario"}], request_key="public",
                        ))
                    except ModelError:
                        assert args.baseline
                    else:
                        assert not args.baseline and response.content == "local-result"
                expected = 1 if args.baseline else 2
                assert len(server.received) == expected
                assert len(store.calls_for_key("public")) == expected
                report["checks"].append({"case": "public-config-and-plugin-binding", "posts": expected})
            finally:
                await composition.dispose()

        async def case(name, config, statuses, count, success):
            """真实 HTTP 结果进入生产 driver，再观察 Models 的账本和回放。"""
            server.replies = deque((status, 0) for status in statuses)
            server.received = []
            store = ModelsStore(root / f"{name}.db", root / "backups")
            store.initialize()
            request = ModelRequest([{"role": "user", "content": "scenario"}], request_key=name)
            bound = _BoundChat(descriptor, physical, store, max_attempts=_retry_budget(config))
            try:
                try:
                    response = await bound.complete(request)
                except ModelError:
                    assert not success, name
                else:
                    assert success and response.content == "local-result", name
                records = store.calls_for_key(name)
                assert len(records) == count and len(server.received) == count, name
                assert records[-1]["state"] == ("success" if success else "error"), records
                if statuses[0] == 429:
                    assert records[0]["send_evidence"] == "rejected", records
                if statuses[0] >= 500:
                    assert records[0]["send_evidence"] is None, records
                    assert records[0]["next_attempt_at"] is None, records
            finally:
                store.close()
            # 关掉 owner 再打开同一真实数据库；既有成功回放，终结失败不新建额度。
            reopened = ModelsStore(root / f"{name}.db", root / "backups")
            reopened.initialize()
            try:
                replay = _BoundChat(descriptor, physical, reopened, max_attempts=_retry_budget(config))
                try:
                    result = await replay.complete(request)
                except ModelUnavailableError:
                    assert not success, name
                else:
                    assert success and result.content == "local-result", name
                assert len(server.received) == count, f"{name}: 重新开库导致重复 POST"
                assert len(reopened.calls_for_key(name)) == count
            finally:
                reopened.close()
            report["checks"].append({"case": name, "posts": count})

        try:
            # 1. 同一场景先在主线复现默认单次失败，再在候选核对失败后成功。
            await public_configuration()
            await case("default", {}, [429, 200], 1 if args.baseline else 2, not args.baseline)
            if args.baseline:
                return report
            await case("default-exhausted", {}, [429] * 4, 3, False)
            await case("explicit-one", {"max_attempts": 1}, [429, 200], 1, False)
            await case("legacy-zero", {"max_retries": 0}, [429, 200], 1, False)
            await case("legacy-one", {"max_retries": 1}, [429, 200], 2, True)
            await case("explicit-precedence", {"max_attempts": 1, "max_retries": 3},
                       [429, 200], 1, False)
            await case("uncertain-5xx", {}, [503, 200], 1, False)
            await case("auth-no-retry", {}, [401, 200], 1, False)

            # 2. 无 key 调用仍只尝试一次；连接未建立的真实错误使用有界额度。
            server.replies = deque([(429, 0), (200, None)])
            server.received = []
            store = ModelsStore(root / "anonymous.db", root / "backups")
            store.initialize()
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=_retry_budget({}))
                try:
                    await bound.complete(ModelRequest([{"role": "user", "content": "scenario"}]))
                except ModelError:
                    pass
                else:
                    raise AssertionError("无 key 的调用自动重试")
                assert len(server.received) == 1
                with sqlite3.connect(store.path) as connection:
                    assert connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 1
            finally:
                store.close()
            report["checks"].append({"case": "anonymous-single-attempt", "posts": 1})

            # 绑定但不监听的本地 socket 确定性返回连接拒绝，不访问外部网络。
            with socket.socket() as refused:
                refused.bind(("127.0.0.1", 0))
                refused_endpoint = f"http://127.0.0.1:{refused.getsockname()[1]}"
                refused_http = HttpClient(lambda: httpx.AsyncClient(
                    base_url=refused_endpoint, trust_env=False,
                ))
                refused_driver = driver._BoundChat(
                    driver._ConnectionConfig(refused_endpoint, 1, 1, 0, False),
                    Credential(), descriptor, driver._ModelConfig(None, 16), refused_http,
                )
                store = ModelsStore(root / "unsent.db", root / "backups")
                store.initialize()
                try:
                    bound = _BoundChat(descriptor, refused_driver, store, max_attempts=_retry_budget({}))
                    try:
                        await bound.complete(ModelRequest([], request_key="unsent"))
                    except ModelError:
                        pass
                    else:
                        raise AssertionError("未监听的 socket 返回了成功")
                    records = store.calls_for_key("unsent")
                    assert len(records) == 3
                    assert all(record["send_evidence"] == "unsent" for record in records)
                    assert records[-1]["next_attempt_at"] is None
                finally:
                    store.close()
                    await refused_http.aclose()
            report["checks"].append({"case": "real-connect-refused", "attempts": 3, "posts": 0})

            # 3. 在已提交的退避记录处取消，重新开库后按剩余额度继续。
            server.replies = deque([(429, 3600), (200, None)])
            server.received = []
            store = ModelsStore(root / "cancel.db", root / "backups")
            store.initialize()
            request = ModelRequest([{"role": "user", "content": "scenario"}], request_key="cancel")
            bound = _BoundChat(descriptor, physical, store, max_attempts=_retry_budget({}))
            backoff = asyncio.Event()
            original_sleep = asyncio.sleep

            async def wait_backoff(delay):
                if delay > 3000:
                    backoff.set()
                await original_sleep(delay)

            try:
                with patch("plugins.models.state.asyncio.sleep", wait_backoff):
                    task = asyncio.create_task(bound.complete(request))
                    await asyncio.wait_for(backoff.wait(), 5)
                    record = store.calls_for_key("cancel")[0]
                    assert record["next_attempt_at"] is not None and record["state"] == "error"
                    assert len(server.received) == 1
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError("退避取消未传播")
            finally:
                store.close()
            resumed_store = ModelsStore(root / "cancel.db", root / "backups")
            resumed_store.initialize()
            try:
                resumed = _BoundChat(descriptor, physical, resumed_store, max_attempts=_retry_budget({}))
                # 确定性推进耐久退避的墙钟；网络请求仍穿过真实 HTTP/driver。
                with patch("plugins.models.state.time.time", return_value=record["next_attempt_at"] + 1):
                    assert (await resumed.complete(request)).content == "local-result"
                assert len(server.received) == 2
                assert len(resumed_store.calls_for_key("cancel")) == 2
            finally:
                resumed_store.close()
            report["checks"].append({"case": "cancel-backoff-reopen", "posts": 2})
        finally:
            await http.aclose()
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive(), "场景服务没有排空"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
