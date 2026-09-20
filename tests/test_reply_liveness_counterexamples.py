"""§10.2 决定性反例：与红测试互补，逐项固定不可退化的行为。

并发用 Event 协调；持久证据全部来自一次性测试数据库，不猜内存态。
"""

import asyncio
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
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
    """§10.2-6/10：同进程活 attempt 不被误判成孤儿，并发同 key 调用合并到原 attempt。"""
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
            entered.set()
            await release.wait()
            return LLMResponse(f"answer {calls}")

    bound = _BoundChat(_descriptor(), Driver(), store)
    request = ModelRequest((), request_key="same-premise")
    first = asyncio.create_task(bound.complete(request))
    await asyncio.wait_for(entered.wait(), 1)
    # 第二个同 key 调用发现的是活 attempt，必须合并等待而非另付一次外部请求。
    second = asyncio.create_task(bound.complete(request))
    release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert calls == 1
    assert second_result.call_record_id == first_result.call_record_id
    rows = store.calls_for_key("same-premise")
    assert [row["attempt"] for row in rows] == [0]
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
    """§10.2-13：确认死亡的 listener 移除且不污染提交；未确认死亡的保留，迟到注销幂等。"""
    log = MessageLog(tmp_path / "state.db")
    reader = log.reader("s")

    class DeadLoop:
        def is_closed(self) -> bool:
            return True

        def call_soon_threadsafe(self, callback, *args):
            del callback, args
            raise RuntimeError("loop is closed")

    class UnverifiedLoop:
        def call_soon_threadsafe(self, callback, *args):
            del callback, args
            raise RuntimeError("transient notify failure")

    poison = asyncio.Event()
    unverified = asyncio.Event()
    with log._lock:
        log._listeners[poison] = DeadLoop()
        log._listeners[unverified] = UnverifiedLoop()
    inputs = log.writer(
        "s", author="test", source="conversation", body_types=(Input,), content={},
    )
    # 已提交事务返回原结果；确认死亡的移除，无法确认死亡的保留待核对。
    message = inputs.append("m1", Input(()))
    assert message.message_id == "m1"
    with log._lock:
        assert poison not in log._listeners
        assert unverified in log._listeners
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


_KILLED_CHILD = """
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, {repo!r})

from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities,
    ModelRequest,
)
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore


class Driver:
    max_tool_schemas = None

    async def complete(self, request):
        await asyncio.Event().wait()


descriptor = BoundModelDescriptor(
    binding_id="bound", plugin_snapshot_id="snapshot", model_revision=0,
    model_id="model", connection_id="connection", driver_id="driver",
    driver_contract_version="1", auth_identity="identity", model="model",
    role="agent", reasoning_effort=None, capabilities=ModelCapabilities(),
    capability_sources=CapabilitySources(), capability_digest="digest",
)
store = ModelsStore(Path(sys.argv[1]), Path(sys.argv[2]))
store.initialize()
asyncio.run(
    _BoundChat(descriptor, Driver(), store).complete(
        ModelRequest((), request_key="killed-key")
    )
)
"""


@pytest.mark.asyncio
async def test_sigkilled_provider_process_leaves_settled_orphan_without_replay(tmp_path):
    """§10.2-10 真实强杀：子进程死在 provider 窗口，started 记录按死 owner 结算。"""
    repo = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_KILLED_CHILD).format(repo=repo),
         str(tmp_path / "models.db"), str(tmp_path / "backups")],
    )
    try:
        store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
        # 等子进程的 started 记录真实落库再 SIGKILL，不猜时间。
        for _ in range(200):
            try:
                probe = ModelsStore(tmp_path / "models.db", tmp_path / "backups", writable=False)
                rows = probe.calls_for_key("killed-key")
            except Exception:
                rows = ()
            if rows:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("子进程未在被杀前持久化 started 记录")
        child.kill()
        assert child.wait() != 0

        # 本进程重新初始化宿主纪元：旧 owner 的纪元/进程 token 都失去存活证据。
        store.initialize()
        calls = 0

        class Driver:
            max_tool_schemas = None

            async def complete(self, request):
                nonlocal calls
                del request
                calls += 1
                return LLMResponse("explicit retry")

        bound = _BoundChat(_descriptor(), Driver(), store)
        with pytest.raises(ModelUnavailableError, match="不确定"):
            await bound.complete(ModelRequest((), request_key="killed-key"))
        assert calls == 0, "孤儿未结算前不得重发付费请求"
        orphan = store.calls_for_key("killed-key")[0]
        assert orphan["state"] == "error" and "orphaned" in orphan["failure"]

        response = await bound.complete(ModelRequest((), request_key="killed-key"))
        assert response.content == "explicit retry" and calls == 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
