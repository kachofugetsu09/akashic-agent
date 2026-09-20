"""§10.2 决定性反例：与红测试互补，逐项固定不可退化的行为。

并发用 Event 协调；持久证据全部来自一次性测试数据库，不猜内存态。
"""

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelRequest,
    ModelUnavailableError,
)
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from session.log import MessageLog
from session.message import Control, Input, Output
from tests import test_message_react as react_fixtures


def _descriptor() -> BoundModelDescriptor:
    return BoundModelDescriptor(
        binding_id="bound",
        plugin_snapshot_id="snapshot",
        model_revision=0,
        model_id="model",
        connection_id="connection",
        driver_id="driver",
        driver_contract_version="1",
        auth_identity="identity",
        model="model",
        role="agent",
        reasoning_effort=None,
        capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(),
        capability_digest="digest",
    )


@pytest.mark.asyncio
async def test_unrelated_same_source_result_does_not_supersede_draft(tmp_path):
    """§10.2-9：无关 ToolResult/Output 抬高 head 时复用同一响应原子提交。"""
    calls = 0
    injected = asyncio.Event()

    async def complete(request):
        nonlocal calls
        calls += 1
        if calls == 1 and not injected.is_set():
            # 在 provider 返回前提交一条同来源、非 Input/Control 的事实抬高 head。
            writer = log.writer(
                "s", author="probe", source="conversation",
                body_types=(Output,), content={},
            )
            writer.append("probe-output", Output((), "continue"))
            writer.expire()
            injected.set()
        return LLMResponse("final answer")

    async def invoke(key, arguments):
        del key, arguments
        raise AssertionError("tool must not run")

    async with react_fixtures.runtime(
        tmp_path, complete, invoke
    ) as (conversation, log, _store, run):
        await conversation.accept("u1", Input(()))
        result = await (await conversation.start(run)).join()
        assert result.body.finish == "complete"
        snapshot = log.reader("s").snapshot()
        assert [m.message_id for m in snapshot[:2]] == ["u1", "probe-output"]
        assert not any(
            isinstance(m.body, Control) for m in snapshot
        ), "无关事实抬高 head 不能升级为 failure 或中断"
        assert calls == 1


@pytest.mark.asyncio
async def test_dead_owner_started_call_settles_as_orphan_then_explicit_retry(tmp_path):
    """§10.2-10/13：同 key 的 started 记录属死 owner 时明确结算，不永久等待。"""
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    calls = 0

    class Driver:
        max_tool_schemas = None

        async def complete(self, request):
            nonlocal calls
            del request
            calls += 1
            return LLMResponse("retried answer")

    request = ModelRequest((), request_key="stable-key")
    descriptor = _descriptor()
    # 模拟上一进程遗留的 started 记录：owner 不属于任何活 attempt。
    orphan_id = store.resume_call(
        descriptor, request, request_key="stable-key", owner_id="dead-process"
    )
    bound = _BoundChat(descriptor, Driver(), store)
    with pytest.raises(ModelUnavailableError, match="不确定"):
        await bound.complete(request)
    assert calls == 0, "孤儿证据未结算前不得发起新的付费请求"
    orphan = store.read_call(orphan_id)
    assert orphan["state"] == "error" and "orphaned" in orphan["failure"]

    # 显式重试是新的真实 attempt，不复用孤儿结果。
    response = await bound.complete(request)
    assert response.content == "retried answer"
    assert calls == 1
    rows = store.calls_for_key("stable-key")
    assert sorted(row["attempt"] for row in rows) == [0, 1]


@pytest.mark.asyncio
async def test_live_attempt_is_not_treated_as_orphan(tmp_path):
    """§10.2-6/10：同进程活 attempt 不被误判成孤儿，新代际登记为新 attempt。"""
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Driver:
        max_tool_schemas = None

        async def complete(self, request):
            nonlocal calls
            del request
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return LLMResponse(f"answer {calls}")

    bound = _BoundChat(_descriptor(), Driver(), store)
    request = ModelRequest((), request_key="same-premise")
    first = asyncio.create_task(bound.complete(request))
    await asyncio.wait_for(entered.wait(), 1)
    # 第二个同 key 调用发现的是活 attempt，必须另起 attempt 而非结算或重放。
    second = asyncio.create_task(bound.complete(request))
    release.set()
    await asyncio.gather(first, second)
    assert calls == 2
    rows = store.calls_for_key("same-premise")
    assert sorted(row["attempt"] for row in rows) == [0, 1]
    assert all(row["state"] == "success" for row in rows)


@pytest.mark.asyncio
async def test_settlement_store_failure_propagates_without_hidden_retry(tmp_path):
    """§10.2-5：结算保存失败如实上抛，进程不重发请求也不假装成功。"""
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    calls = 0

    class Driver:
        max_tool_schemas = None

        async def complete(self, request):
            nonlocal calls
            del request
            calls += 1
            return LLMResponse("unrecorded")

    bound = _BoundChat(_descriptor(), Driver(), store)

    def broken(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("settlement store unavailable")

    store.finish_call = broken  # type: ignore[method-assign]
    with pytest.raises(OSError, match="unavailable"):
        await bound.complete(ModelRequest(()))
    assert calls == 1


@pytest.mark.asyncio
async def test_dead_listener_evicted_once_and_late_unsubscribe_is_safe(tmp_path):
    """§10.2-13：确认死亡的 listener 只移除一次，迟到注销幂等。"""
    log = MessageLog(tmp_path / "state.db")
    reader = log.reader("s")

    class DeadLoop:
        def call_soon_threadsafe(self, callback, *args):
            del callback, args
            raise RuntimeError("loop is closed")

    poison = asyncio.Event()
    with log._lock:
        log._listeners[poison] = DeadLoop()
    inputs = log.writer(
        "s", author="test", source="conversation", body_types=(Input,), content={},
    )
    with pytest.raises(RuntimeError, match="loop is closed"):
        inputs.append("m1", Input(()))
    with log._lock:
        assert poison not in log._listeners
        # 迟到注销对已驱逐 listener 必须无害。
        log._listeners.pop(poison, None)
    inputs.append("m2", Input(()))
    assert reader.get("m2") is not None
    log.close()


@pytest.mark.asyncio
async def test_failure_control_stalls_lane_until_explicit_resume(tmp_path):
    """§10.2-12：持久 failure 之后同位置不再驱动程序；显式 resume 才继续。"""
    async def complete(request):
        del request
        return LLMResponse("answer")

    async def invoke(key, arguments):
        del key, arguments
        raise AssertionError("tool must not run")

    async with react_fixtures.runtime(
        tmp_path, complete, invoke
    ) as (conversation, log, _store, run):
        await conversation.accept("u1", Input(()))
        await conversation.record_failure(RuntimeError("program failed durably"))
        # failure 已持久停摆：同一 prefix 不再接纳新程序运行。
        assert await conversation.start(run) is None
        controls = [
            m for m in log.reader("s").snapshot()
            if isinstance(m.body, Control)
        ]
        assert controls and controls[-1].body.action == "failure"
