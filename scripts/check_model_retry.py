"""用本地 HTTP 对端和临时 Models 账本验证模型生成恢复，不访问真实 provider。"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import threading


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
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        status, delay = self.server.replies.popleft()
        self.server.received.append(status)
        if ':streamGenerateContent' in self.path or ':generateContent' in self.path:
            part = {"text": "local-result"}
            if delay == "gemini-invalid":
                part = {"text": 123}
            value = {"candidates": [{"content": {"role": "model", "parts": [part]},
                                      "finishReason": "STOP"}]}
            if delay == "gemini-length":
                value["candidates"][0]["content"]["parts"] = []
                value["candidates"][0]["finishReason"] = "MAX_TOKENS"
            if delay in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "SAFETY"}:
                value["candidates"][0]["finishReason"] = delay
                # 即使候选含看似完整的工具参数，失败也不能交出可执行响应。
                value["candidates"][0]["content"]["parts"] = [{"functionCall": {
                    "name": "write_file", "args": {"path": "must-not-exist", "content": "unsafe"}}}]
            streaming = ':streamGenerateContent' in self.path
            body = (("data: " + json.dumps(value) + "\n\n") if streaming else json.dumps(value)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status in (429, 503):
                self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(body)
            return
        value = (
            {"choices": [{"message": {"role": "assistant", "content": "local-result"},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
            if status == 200 else {"error": {"code": "scenario_error", "message": "scenario failure local-scenario"}}
        )
        if status == 400:
            value = {"detail": "scenario failure " + "x" * 475 + " local-scenario"}
        streaming = status == 200 and request.get("stream")
        if streaming:
            chunks: list[dict[str, object]] = [
                {"choices": [{"delta": {"reasoning_content": "local-thinking"}}]},
                {"choices": [{"delta": {"content": "" if delay == "empty" else "local-result"}}]},
            ]
            if delay != "stream-error":
                chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                               "usage": value["usage"]})
            body = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                    + ("" if delay == "stream-error" else "data: [DONE]\n\n")).encode()
        else:
            body = ("scenario failure local-scenario".encode() if status == 502 else json.dumps(value).encode())
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        # 声明更多字节后关闭连接，制造真实的 HTTP 流中断。
        self.send_header("Content-Length", str(len(body) + (100 if delay == "stream-error" else 0)))
        if delay is not None and not streaming:
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

    async def refresh(self, payload: Mapping[str, str]) -> None:
        raise AssertionError("本地固定凭据不支持刷新")

    def exclusive(self) -> AbstractAsyncContextManager[None]:
        raise AssertionError("本地固定凭据不支持认证轮换")


class GeminiCredential(Credential):
    async def read(self) -> dict[str, str]:
        return {"access_token": "local-scenario"}


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
    from plugins.models.store import ModelsStore, _request_digest
    from plugins.openai_compatible import driver
    from agent.plugin_composition import CompositionRoot
    from plugins.models.contract import (
    CHAT_MODELS,
    MODEL_DRIVERS,
)
    from plugins.models.settings import (
        MODEL_SETTINGS, AddConnection, AddModel, CreateConnectionWithModel, SetDefaultModel,
    )
    from plugins.models.state import ModelsState
    from agent.plugin_composition.tasks import Tasks
    from plugins.reply.status import ReplyState
    from plugins.sources.session import SourceSession
    from plugins.content.plugin import check_text
    from agent.plugin_contracts import ContentPart, Control, Input
    from session.log import MessageLog, SessionAttributes
    from agent.plugin_composition.message_view import read_message_rows

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
    report = {"source": str(args.source), "checks": [], "retry_previews": [], "errors": [], "failure_messages": []}

    async def complete_with_preview(bound, request, on_attempt=None, on_retry=None):
        """通过真实 Task、Reply 预览、Models 和 HTTP，观察同一草稿的尝试切换。"""
        tasks, status = Tasks(), ReplyState()
        snapshots = []

        async def operation(task):
            with status.open(task, "scenario", "conversation") as preview:
                with preview("scenario-output") as publish:
                    async def delta(value):
                        await publish(value)
                        draft = status.snapshot("scenario")[0].preview
                        snapshots.append(draft)
                        if value.get("retry_status"):
                            assert draft is not None
                            assert draft.retry_status == value["retry_status"]
                            assert "local-scenario" not in draft.retry_status
                            report["retry_previews"].append(asdict(draft))
                            if on_retry is not None:
                                on_retry()
                        if "call_record_id" in value:
                            assert not draft.retry_status, "下一次尝试没有清除旧错误提示"
                        if on_attempt is not None and "call_record_id" in value:
                            on_attempt()

                    response = await bound.complete(replace(request, on_delta=delta))
                    draft = status.snapshot("scenario")[0].preview
                    assert draft.message_id == "scenario-output"
                    assert draft.text == (response.content or "")
                    assert draft.thinking == (response.thinking or "")
                    assert draft.call_record_id == response.call_record_id
                    return response

        try:
            task = await tasks.admit("scenario", lambda slot: slot.start(operation))
            try:
                return await task.join()
            except asyncio.CancelledError:
                task.cancel()
                try:
                    await task.join()
                except asyncio.CancelledError:
                    pass
                raise
        finally:
            await tasks.close()
            assert not status.snapshot("scenario"), snapshots
            status.close()

    with tempfile.TemporaryDirectory(prefix="model-retry-check-") as temporary:
        root = Path(temporary)

        async def save_visible_failure(name, error):
            """通过真实来源 owner 追加 failure，重开消息库核对原输入与完整原因。"""
            path = root / f"{name}-messages.db"
            log, tasks = MessageLog(path), Tasks()
            log.ensure_session("scenario", SessionAttributes())
            reader = log.reader("scenario")
            source = SourceSession(
                reader=reader,
                inputs=log.writer("scenario", author="user", source="conversation",
                                  body_types=(Input,), content={"text": check_text}),
                controls=log.writer("scenario", author="system", source="conversation",
                                    body_types=(Control,), content={}),
                tasks=tasks,
            )
            try:
                original = await source.accept("scenario-input", Input((ContentPart("text", "scenario"),)))
                await source.record_failure(error)
                rows = reader.snapshot()
                assert rows[0] == original and len(rows) == 2
                assert rows[1].body.reason == str(error)
                report["failure_messages"].append((await read_message_rows(reader.read_page(), display_only=True))[1])
            finally:
                await tasks.close()
                log.close()
            reopened = MessageLog(path)
            try:
                assert reopened.reader("scenario").snapshot() == rows
            finally:
                reopened.close()

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
                        kind='chat', model="scenario",
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
                    except (RuntimeError, TimeoutError) as _model_error:
                        if not (ModelError.matches(_model_error)):
                            raise
                        assert args.baseline
                    else:
                        assert not args.baseline and response.content == "local-result"
                expected = 1 if args.baseline else 2
                assert len(server.received) == expected
                assert len(store.calls_for_key("public")) == expected
                report["checks"].append({"case": "public-config-and-plugin-binding", "posts": expected})
            finally:
                await composition.dispose()

        async def gemini_protocol():
            """真实原生 HTTP 响应必须区分暂时故障、协议失败和长度上限。"""
            from plugins.gemini.driver import _Chat
            native = replace(descriptor, driver_id="gemini")
            async with httpx.AsyncClient(base_url=endpoint + '/', trust_env=False) as client:
                physical = _Chat(client, GeminiCredential(), native)
                for name, replies in (
                    ("gemini-invalid", [(200, "gemini-invalid"), (200, None)]),
                    ("gemini-length", [(200, "gemini-length"), (200, None)]),
                    ("gemini-recover", [(503, None), (200, None)]),
                ):
                    server.replies, server.received = deque(replies), []
                    store = ModelsStore(root / f"{name}.db", root / "backups")
                    store.initialize()
                    try:
                        bound = _BoundChat(native, physical, store, max_attempts=None)
                        request = ModelRequest([{"role": "user", "content": "scenario"}], request_key=name)
                        try:
                            response = await asyncio.wait_for(complete_with_preview(bound, request), 10)
                        except (RuntimeError, TimeoutError) as error:
                            if not (ModelError.matches(error)):
                                raise
                            failure = ModelError.read(error)
                            assert failure is not None and name == "gemini-invalid" and not failure.retryable
                        else:
                            assert name != "gemini-invalid"
                            assert response.finish_reason == ("length" if name == "gemini-length" else "stop")
                            if name == "gemini-recover":
                                assert response.content == "local-result" and response.provider_metadata is not None
                                assert (await bound.complete(request)).provider_metadata == response.provider_metadata
                        expected = 2 if name == "gemini-recover" else 1
                        assert len(server.received) == expected and len(store.calls_for_key(name)) == expected
                        if name == "gemini-recover":
                            assert store.calls_for_key(name)[0]["next_attempt_at"] is not None
                        if name == "gemini-invalid":
                            assert store.calls_for_key(name)[0]["next_attempt_at"] is None
                        report["checks"].append({"case": name, "posts": expected})
                    finally:
                        store.close()

        async def gemini_malformed():
            """真实原生 HTTP 与账本覆盖两种响应、次数耗尽、取消和永久拒绝。"""
            from plugins.gemini.driver import _Chat
            native = replace(descriptor, driver_id="gemini")
            async with httpx.AsyncClient(base_url=endpoint + '/', trust_env=False) as client:
                physical = _Chat(client, GeminiCredential(), native)
                for name, reason, streaming, budget, cancel in (
                    ("malformed-json", "MALFORMED_FUNCTION_CALL", False, None, False),
                    ("malformed-sse", "MALFORMED_FUNCTION_CALL", True, None, False),
                    ("malformed-budget", "MALFORMED_FUNCTION_CALL", True, 2, False),
                    ("malformed-cancel", "MALFORMED_FUNCTION_CALL", True, None, True),
                    ("unexpected", "UNEXPECTED_TOOL_CALL", True, None, False),
                    ("safety", "SAFETY", True, None, False),
                    ("invalid-request", None, True, None, False),
                ):
                    server.replies = deque([(400 if reason is None else 200, reason)] * (budget or 1) + [(200, None)])
                    server.received = []
                    path = root / f"{name}.db"
                    store = ModelsStore(path, root / "backups")
                    store.initialize()
                    bound = _BoundChat(native, physical, store, max_attempts=budget)
                    request = ModelRequest([{"role": "user", "content": "scenario"}], request_key=name)
                    recovering = reason == "MALFORMED_FUNCTION_CALL" and budget is None and not cancel
                    try:
                        try:
                            if streaming:
                                def stop():
                                    task = asyncio.current_task()
                                    assert task is not None
                                    task.cancel()
                                response = await complete_with_preview(bound, request, on_retry=stop if cancel else None)
                            else:
                                response = await bound.complete(request)
                        except asyncio.CancelledError:
                            assert cancel
                        except (RuntimeError, TimeoutError) as error:
                            if not (ModelError.matches(error)):
                                raise
                            assert not recovering and not cancel
                            failure = ModelError.read(error)
                            assert failure is not None and failure.retryable == (reason == "MALFORMED_FUNCTION_CALL")
                            if budget is not None:
                                assert "自动重试次数已用完" in str(error)
                        else:
                            assert recovering and response.content == "local-result" and not response.tool_calls
                            assert (await bound.complete(request)).content == response.content
                        records = store.calls_for_key(name)
                        expected = 2 if recovering else budget or 1
                        assert len(records) == len(server.received) == expected
                        first = records[0]
                        assert first["state"] == "error" and first["response"] is None
                        assert (first["next_attempt_at"] is not None) == (recovering or cancel or budget == 2)
                        if budget is not None:
                            assert records[-1]["next_attempt_at"] is None
                        assert first["send_evidence"] == ("rejected" if reason is None else None)
                        assert first["usage"] is None
                        store.close()
                        store = ModelsStore(path, root / "backups")
                        store.initialize()
                        assert store.calls_for_key(name) == records
                        report["checks"].append({"case": name, "posts": expected})
                    finally:
                        store.close()

        async def case(name, config, statuses, count, success, retry_after: int | str = 0):
            """真实 HTTP 结果进入生产 driver，再观察 Models 的账本和回放。"""
            server.replies = deque((status, retry_after) for status in statuses)
            server.received = []
            store = ModelsStore(root / f"{name}.db", root / "backups")
            store.initialize()
            request = ModelRequest([{"role": "user", "content": "scenario"}], request_key=name)
            bound = _BoundChat(descriptor, physical, store, max_attempts=_retry_budget(config))
            try:
                try:
                    response = await complete_with_preview(bound, request)
                except (RuntimeError, TimeoutError) as error:
                    if not (ModelError.matches(error)):
                        raise
                    assert not success, name
                    assert "scenario failure" in str(error)
                    if statuses[0] not in (400, 502):
                        assert "scenario_error" in str(error)
                    if statuses[0] == 400:
                        assert "local-" not in str(error), "截断泄漏了部分凭据"
                    assert "local-scenario" not in str(error)
                    assert f"已尝试 {count}" in str(error)
                    report["errors"].append({"case": name, "message": str(error)})
                    await save_visible_failure(name, error)
                else:
                    assert success and response.content == "local-result", name
                records = store.calls_for_key(name)
                assert len(records) == count and len(server.received) == count, name
                assert records[-1]["state"] == ("success" if success else "error"), records
                if statuses[0] == 429:
                    assert records[0]["send_evidence"] == "rejected", records
                    if retry_after == 3600:
                        assert records[0]["next_attempt_at"] is None
                        assert "至少再等待" in records[0]["failure"]
                    elif isinstance(retry_after, str) and "," in retry_after:
                        expected = parsedate_to_datetime(retry_after).timestamp()
                        assert records[0]["next_attempt_at"] == expected
                        assert records[1]["started_at"] >= datetime.fromtimestamp(expected, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                if statuses[0] >= 500:
                    assert records[0]["send_evidence"] is None, records
                    assert records[0]["next_attempt_at"] is not None, records
            finally:
                store.close()
            # 关掉 owner 再打开同一真实数据库；既有成功回放，终结失败不新建额度。
            reopened = ModelsStore(root / f"{name}.db", root / "backups")
            reopened.initialize()
            try:
                replay = _BoundChat(descriptor, physical, reopened, max_attempts=_retry_budget(config))
                try:
                    result = await replay.complete(request)
                except (RuntimeError, TimeoutError) as _model_error:
                    if not (ModelError.matches(_model_error, ModelUnavailableError)):
                        raise
                    assert not success, name
                else:
                    assert success and result.content == "local-result", name
                assert len(server.received) == count, f"{name}: 重新开库导致重复 POST"
                assert len(reopened.calls_for_key(name)) == count
                assert reopened.calls_for_key(name) == records, "回放改变了既有调用记录"
            finally:
                reopened.close()
            report["checks"].append({"case": name, "posts": count})

        try:
            if args.gemini_only:
                await gemini_protocol()
                await gemini_malformed()
                return report
            # 1. 同一场景先在主线复现默认单次失败，再在候选核对失败后成功。
            await public_configuration()
            if not args.baseline:
                await gemini_protocol()
                await gemini_malformed()
            await case("default", {}, [429, 200], 1 if args.baseline else 2, not args.baseline)
            if args.baseline:
                return report
            await case("explicit-exhausted", {"max_attempts": 6}, [429] * 7, 6, False)
            await case("default-past-six", {}, [429] * 7 + [200], 8, True)
            await case("explicit-one", {"max_attempts": 1}, [429, 200], 1, False)
            await case("legacy-zero", {"max_retries": 0}, [429, 200], 1, False)
            await case("legacy-one", {"max_retries": 1}, [429, 200], 2, True)
            await case("explicit-precedence", {"max_attempts": 1, "max_retries": 3},
                       [429, 200], 1, False)
            await case("server-recovery", {}, [503, 200], 2, True)
            await case("gateway-recovery", {}, [502, 200], 2, True)
            await case("long-diagnostic-redaction", {}, [400, 200], 1, False)
            await case("auth-no-retry", {}, [401, 200], 1, False)
            await case("quota-no-retry", {}, [402, 200], 1, False)
            await case("invalid-retry-after", {}, [429, 200], 2, True, retry_after="NaN")
            await case("date-retry-after", {}, [429, 200], 2, True, retry_after=
                       format_datetime(datetime.fromtimestamp(time.time() + 2, timezone.utc), usegmt=True))

            # HTTP 200 已产生思考和正文后断流：撤销旧草稿，原模型步恢复成功。
            server.replies = deque([(200, "stream-error"), (200, None)])
            server.received = []
            store = ModelsStore(root / "stream-error.db", root / "backups")
            store.initialize()
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=3)
                await complete_with_preview(bound, ModelRequest([], request_key="stream-error"))
                records = store.calls_for_key("stream-error")
                assert len(records) == 2 and server.received == [200, 200]
                assert records[0]["partial_response"] and records[0]["state"] == "error"
                assert records[0]["usage"] is None and records[0]["send_evidence"] is None
                assert records[0]["next_attempt_at"] is not None
                report["checks"].append({"case": "reply-stream-error-recovered", "posts": 2})
            finally:
                store.close()

            # 完整 HTTP 空生成仍消耗实际 attempt，已知 usage 保留，再生成完整正文。
            server.replies = deque([(200, "empty"), (200, None)])
            server.received = []
            store = ModelsStore(root / "empty.db", root / "backups")
            store.initialize()
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=None)
                await complete_with_preview(bound, ModelRequest([], request_key="empty"))
                records = store.calls_for_key("empty")
                assert len(records) == 2 and records[0]["state"] == "error"
                assert records[0]["usage"] is not None and records[0]["partial_response"]
            finally:
                store.close()
            report["checks"].append({"case": "empty-generation-recovered", "posts": 2})

            # 回调即使抛出 ModelTimeoutError，也不是 provider 故障，不能自动重复请求。
            from agent.plugin_composition.models import ModelTimeoutError
            server.replies = deque([(200, None), (200, None)])
            server.received = []
            store = ModelsStore(root / "callback.db", root / "backups")
            store.initialize()
            async def rejected_preview(value):
                if value.get("thinking_delta"):
                    raise ModelTimeoutError("preview callback failed").exception()
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=None)
                try:
                    await bound.complete(ModelRequest([], request_key="callback", on_delta=rejected_preview))
                except (RuntimeError, TimeoutError) as error:
                    if not (ModelError.matches(error, ModelTimeoutError)):
                        raise
                    assert "preview callback failed" in str(error)
                else:
                    raise AssertionError("回调失败被伪装成功")
                assert len(server.received) == 1
                assert store.calls_for_key("callback")[0]["next_attempt_at"] is None
            finally:
                store.close()
            report["checks"].append({"case": "callback-failure-no-retry", "posts": 1})

            # 已确认旧 owner 死亡后恢复生成；同纪元身份不明的 owner 仍阻断。
            path = root / "orphan.db"
            store = ModelsStore(path, root / "backups")
            store.initialize()
            request = ModelRequest([], request_key="orphan")
            store.resume_call(descriptor, _request_digest(request), request_key="orphan",
                              owner_id=f"{store.host_epoch}:old-process:old-root:old-attempt", max_attempts=6)
            before = store.calls_for_key("orphan")[0]
            store.close()
            reopened = ModelsStore(path, root / "backups")
            reopened.initialize()
            server.replies = deque([(200, None)])
            server.received = []
            try:
                bound = _BoundChat(descriptor, physical, reopened, max_attempts=None)
                await complete_with_preview(bound, request)
                records = reopened.calls_for_key("orphan")
                assert len(records) == 2 and records[0]["state"] == "error"
                assert records[0]["usage"] is None and records[0]["response"] is None
                assert all(records[0][key] == before[key] for key in ("id", "request_digest", "binding", "attempt", "started_at", "owner_id"))
                unknown = ModelRequest([], request_key="unknown-owner")
                reopened.resume_call(descriptor, _request_digest(unknown), request_key="unknown-owner",
                    owner_id=f"{reopened.host_epoch}:another-process:another-root:attempt", max_attempts=6)
                try:
                    await bound.complete(unknown)
                except (RuntimeError, TimeoutError) as error:
                    if not (ModelError.matches(error, ModelUnavailableError)):
                        raise
                    assert "无法确认" in str(error)
                else:
                    raise AssertionError("身份不明的 owner 被接管")
                assert server.received == [200]
            finally:
                reopened.close()
            report["checks"].append({"case": "confirmed-orphan-recovered", "posts": 1})

            # 首次连接确实被拒绝，第二个 attempt 发布时才让同一端口开始监听。
            recovery = ThreadingHTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
            recovery.server_bind()
            recovery.replies = deque([(200, None)])
            recovery.received = []
            recovery_thread = threading.Thread(target=recovery.serve_forever, daemon=True)
            recovery_endpoint = f"http://127.0.0.1:{recovery.server_port}"
            recovery_http = HttpClient(lambda: httpx.AsyncClient(
                base_url=recovery_endpoint, trust_env=False,
            ))
            recovery_driver = driver._BoundChat(
                driver._ConnectionConfig(recovery_endpoint, 1, 1, 0, False),
                Credential(), descriptor, driver._ModelConfig(None, 16), recovery_http,
            )
            store = ModelsStore(root / "connect-recovery.db", root / "backups")
            store.initialize()

            def listen_on_retry():
                if len(store.calls_for_key("connect-recovery")) == 2:
                    recovery.server_activate()
                    recovery_thread.start()

            try:
                bound = _BoundChat(descriptor, recovery_driver, store, max_attempts=3)
                response = await complete_with_preview(
                    bound, ModelRequest([], request_key="connect-recovery"), listen_on_retry,
                )
                records = store.calls_for_key("connect-recovery")
                assert len(records) == 2 and recovery.received == [200]
                assert records[0]["send_evidence"] == "unsent"
                assert records[0]["state"] == "error" and records[1]["state"] == "success"
                assert records[0]["id"] != records[1]["id"] == response.call_record_id
                report["checks"].append({"case": "reply-connect-recovery", "attempts": 2, "posts": 1})
            finally:
                store.close()
                await recovery_http.aclose()
                if recovery_thread.is_alive():
                    await asyncio.to_thread(recovery.shutdown)
                    recovery_thread.join(timeout=5)
                recovery.server_close()
                assert not recovery_thread.is_alive()

            # 2. 无 key 调用仍只尝试一次；连接未建立的真实错误使用有界额度。
            server.replies = deque([(429, 0), (200, None)])
            server.received = []
            store = ModelsStore(root / "anonymous.db", root / "backups")
            store.initialize()
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=_retry_budget({}))
                try:
                    await bound.complete(ModelRequest([{"role": "user", "content": "scenario"}]))
                except (RuntimeError, TimeoutError) as _model_error:
                    if not (ModelError.matches(_model_error)):
                        raise
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
                    bound = _BoundChat(descriptor, refused_driver, store, max_attempts=6)
                    try:
                        await bound.complete(ModelRequest([], request_key="unsent"))
                    except (RuntimeError, TimeoutError) as error:
                        if not (ModelError.matches(error)):
                            raise
                        assert "ConnectError" in str(error) and "已尝试 6/6" in str(error)
                        assert "自动重试次数已用完" in str(error)
                        report["errors"].append({"case": "real-connect-refused", "message": str(error)})
                        await save_visible_failure("real-connect-refused", error)
                    else:
                        raise AssertionError("未监听的 socket 返回了成功")
                    records = store.calls_for_key("unsent")
                    assert len(records) == 6
                    for index, record in enumerate(records[:-1]):
                        finished = datetime.fromisoformat(record["finished_at"]).replace(tzinfo=timezone.utc).timestamp()
                        base = min(20, 2 * 2 ** index)
                        assert base * 0.9 <= record["next_attempt_at"] - finished <= base * 1.1 + 1
                    assert all(record["send_evidence"] == "unsent" for record in records)
                    assert records[-1]["next_attempt_at"] is None
                finally:
                    store.close()
                    await refused_http.aclose()
            report["checks"].append({"case": "real-connect-refused", "attempts": 6, "posts": 0})

            # 3. 真实等待期间取消，重新开库按同一允许时间和剩余额度继续。
            server.replies = deque([(429, 2), (200, None)])
            server.received = []
            store = ModelsStore(root / "cancel.db", root / "backups")
            store.initialize()
            request = ModelRequest([{"role": "user", "content": "scenario"}], request_key="cancel")
            bound = _BoundChat(descriptor, physical, store, max_attempts=_retry_budget({}))
            backoff = asyncio.Event()
            try:
                task = asyncio.create_task(complete_with_preview(bound, request, on_retry=backoff.set))
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
                assert (await complete_with_preview(resumed, request)).content == "local-result"
                records = resumed_store.calls_for_key("cancel")
                assert len(server.received) == 2 and len(records) == 2
                assert records[0] == record, "恢复改写了原失败或等待期限"
            finally:
                resumed_store.close()
            report["checks"].append({"case": "cancel-backoff-reopen", "posts": 2})

            # SQLite 真实拒写回执时，保留 provider 错误并公开保存失败，不重发。
            server.replies = deque([(429, 0), (200, None)])
            server.received = []
            store = ModelsStore(root / "receipt-failure.db", root / "backups")
            store.initialize()
            with sqlite3.connect(store.path) as connection:
                connection.execute("CREATE TRIGGER reject_receipt BEFORE UPDATE OF state ON model_calls BEGIN SELECT RAISE(ABORT, 'receipt write rejected'); END")
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=6)
                try:
                    await bound.complete(ModelRequest([], request_key="receipt-failure"))
                except (RuntimeError, TimeoutError) as error:
                    if not (ModelError.matches(error)):
                        raise
                    assert "HTTP 429" in str(error) and "回执保存失败" in str(error)
                    assert "自动重试已停止" in str(error)
                    await save_visible_failure("receipt-failure", error)
                else:
                    raise AssertionError("回执保存失败被报告为成功")
                assert server.received == [429]
                records = store.calls_for_key("receipt-failure")
                assert len(records) == 1 and records[0]["state"] == "started"
            finally:
                store.close()
            report["checks"].append({"case": "receipt-failure-visible", "posts": 1})

            # Retry-After 很长时保持可取消等待，不提前请求、不终结成要求新输入。
            server.replies = deque([(429, 3600), (200, None)])
            server.received = []
            store = ModelsStore(root / "long-wait.db", root / "backups")
            store.initialize()
            waiting = asyncio.Event()
            try:
                bound = _BoundChat(descriptor, physical, store, max_attempts=None)
                request = ModelRequest([], request_key="long-wait")
                task = asyncio.create_task(complete_with_preview(bound, request, on_retry=waiting.set))
                await asyncio.wait_for(waiting.wait(), 5)
                records = store.calls_for_key("long-wait")
                assert records[0]["next_attempt_at"] > time.time() + 3500
                assert bound.key_recovery("long-wait") == "open"
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                assert server.received == [429]
                assert store.calls_for_key("long-wait") == records
            finally:
                store.close()
            report["checks"].append({"case": "long-retry-after-cancellable", "posts": 1})

            # 旧退避的时间不会因经过 90 秒变成终态；重开保留原请求与失败记录。
            store = ModelsStore(root / "expired.db", root / "backups")
            store.initialize()
            request = ModelRequest([], request_key="expired")
            call_id = store.resume_call(descriptor, _request_digest(request), request_key="expired",
                                        owner_id="fixture", max_attempts=6)
            store.finish_call(call_id, usage=None, failure="TransportError: old connection failure",
                              send_evidence="unsent", next_attempt_at=time.time() - 1)
            with sqlite3.connect(store.path) as connection:
                connection.execute("UPDATE model_calls SET started_at=datetime('now','-91 seconds') WHERE id=?", (call_id,))
            before = store.calls_for_key("expired")
            store.close()
            reopened = ModelsStore(root / "expired.db", root / "backups")
            server.replies = deque([(200, None)])
            server.received = []
            try:
                bound = _BoundChat(descriptor, physical, reopened, max_attempts=None)
                assert bound.key_recovery("expired") == "open"
                assert (await complete_with_preview(bound, request)).content == "local-result"
                assert reopened.calls_for_key("expired")[0] == before[0]
                assert server.received == [200]
            finally:
                reopened.close()
            report["checks"].append({"case": "old-backoff-reopen", "posts": 1})

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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--gemini-only", action="store_true", help="仅验证 Gemini HTTP 与恢复边界")
    parser.add_argument("--output", type=Path, help="保存真实错误与预览供浏览器重放")
    args = parser.parse_args()
    args.source = args.source.resolve()
    report = json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
