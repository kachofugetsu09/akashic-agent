from __future__ import annotations

import asyncio
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import agent.plugins.manager as plugin_manager_module
from agent.control.models import TurnItem, TurnItemKind, TurnRequest, TurnStatus
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.control.scoped_turn import TurnAcceptedReceipt
from agent.control.timer import TimerReceipt, TimerStatus
from agent.lifecycle.composition import (
    CONTEXT_PREPARED_EVENT,
    run_composition_lifecycle,
)
from agent.lifecycle.types import BeforeTurnCtx
from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from plugins.content.plugin import CONTENT_SOURCE, CONTENT_WAKE
from plugins.content.store import ContentStore
from plugins.drift.plugin import DRIFT_PROPOSALS, DRIFT_WAKE
from session.manager import SessionManager
from session.store import SessionStore


class _TimerHandle:
    def __init__(self, deadline: datetime) -> None:
        self.deadline = deadline
        self.future: asyncio.Future[TimerReceipt] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def id(self) -> str:
        return "timer:wake:e2e"

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self.future)

    async def cancel(self) -> TimerReceipt:
        if not self.future.done():
            self.future.set_result(
                TimerReceipt(
                    self.id, self.deadline, datetime.now(UTC), TimerStatus.CANCELLED
                )
            )
        return await self.future

    async def cleanup(self) -> None:
        _ = await self.cancel()

    def fire(self) -> None:
        self.future.set_result(
            TimerReceipt(self.id, self.deadline, datetime.now(UTC), TimerStatus.FIRED)
        )


class _Timer:
    def __init__(self) -> None:
        self.handles: list[_TimerHandle] = []

    def schedule(self, deadline: datetime) -> _TimerHandle:
        handle = _TimerHandle(deadline)
        self.handles.append(handle)
        return handle


async def _eventually(predicate) -> None:
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


def _copy_plugins(tmp_path: Path) -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    paths = []
    for name in ("content", "drift", "wake"):
        target = tmp_path / "plugins" / name
        shutil.copytree(root / "plugins" / name, target)
        paths.append(target)
    return paths


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "decision_name",
        "decision_arguments",
        "decision_status",
        "expected_content_state",
        "expected_body",
    ),
    [
        (
            "share_content",
            {"message": "fixture share body"},
            "success",
            "settled",
            "fixture share body",
        ),
        (
            "skip_content",
            {"reason": "fixture candidate is irrelevant"},
            "success",
            "abandoned",
            None,
        ),
        (None, {}, None, "deferred", None),
        ("share_content", {"message": "   "}, "error", "deferred", None),
        ("skip_content", {"reason": "\t"}, "error", "deferred", None),
    ],
)
async def test_real_wake_plugin_uses_typed_decision_not_model_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision_name: str | None,
    decision_arguments: dict[str, object],
    decision_status: str | None,
    expected_content_state: str,
    expected_body: str | None,
) -> None:
    timer = _Timer()
    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", lambda: timer)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    wake_data = workspace / "plugin-data" / "wake-builtin"
    wake_data.mkdir(parents=True)
    (wake_data / "config.local.toml").write_text(
        """
[delivery]
channel = "recording"
recipient = "recipient:one"
session_id = "recipient-session"
""".strip() + "\n",
        encoding="utf-8",
    )
    store = SessionStore(workspace / "sessions.db")
    sessions = SessionManager(workspace)
    provider_calls: list[str] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        turn_id = request.metadata["turnId"]
        assert isinstance(turn_id, str)
        ctx = BeforeTurnCtx(
            session_key=request.thread_id,
            channel=str(request.metadata["channel"]),
            chat_id=str(request.metadata["chatId"]),
            content=request.input,
            timestamp=datetime.now(UTC),
            retrieved_memory_block="",
            retrieval_trace_raw=None,
            history_messages=(),
            turn_id=turn_id,
        )
        await run_composition_lifecycle(CONTEXT_PREPARED_EVENT, ctx)
        items = (
            []
            if decision_name is None
            else [
                TurnItem(
                    TurnItemKind.TOOL_CALL,
                    f"item:{decision_name}",
                    {
                        "callId": f"call:{decision_name}",
                        "name": decision_name,
                        "status": decision_status,
                        "arguments": decision_arguments,
                        "resultPreview": '{"recorded":true}',
                    },
                )
            ]
        )
        return ControlExecutionResult(
            response="过滤，不推送（事故诱饵，绝不能发给用户）",
            items=items,
        )

    async def deliver(request, provider_started):
        provider_started(
            DurableBindingAttempt(
                request.logical_delivery_id,
                "snapshot:recording",
                "generation:recording",
                "binding:recording",
            )
        )
        provider_calls.append(request.body)
        return ChannelDeliveryReceipt(
            request.logical_delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider:recording",),
        )

    conversation = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=_copy_plugins(tmp_path),
        event_bus=EventBus(),
        workspace=workspace,
        session_manager=sessions,
        tool_registry=ToolRegistry(),
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    manager.bind_durable_delivery_sender(deliver)
    await manager.load_all()
    if decision_status == "error":
        registry = manager.current_snapshot.tool_registry
        assert registry is not None and decision_name is not None
        lease = manager.snapshot_store.lease()
        snapshot_token = bind_runtime_snapshot(lease)
        try:
            registry.set_context(
                origin_channel="wake",
                origin_session_key="wake:default",
                turn_id="turn:boundary",
            )
            with pytest.raises(ValueError, match="必须是非空字符串"):
                await registry.execute(
                    decision_name,
                    decision_arguments,
                    raise_errors=True,
                )
        finally:
            reset_runtime_snapshot(snapshot_token)
            await lease.release()
    root = manager.current_snapshot.composition_root
    assert root is not None
    source = root.context.require(CONTENT_SOURCE).bind("fitbit-e2e")
    _ = source.submit(
        "poll:e2e",
        (
            {
                "item_id": "sleep:e2e",
                "revision": "1",
                "payload": {"kind": "sleep"},
                "not_before": datetime.now(UTC),
                "requires_ack": False,
            },
        ),
    )
    ledger = DurableDeliveryStore(
        workspace / "runtime" / "deliveries" / "settlements.sqlite"
    )
    content_store = ContentStore(
        workspace / "plugin-data" / "content-builtin" / "content.sqlite3"
    )
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: len(timer.handles) == 1)
        timer.handles[0].fire()
        await _eventually(
            lambda: content_store.state_counts() == {expected_content_state: 1}
        )

        turns = store.list_turns("wake:default")
        assert len(turns) == 1
        accepted = TurnAcceptedReceipt("wake:default", turns[0].id)
        delivery = PluginDurableDeliveries(ledger, None, None, recover_started=False)
        view = delivery.lookup(accepted)
        messages = sessions.control_store.fetch_session_messages("recipient-session")
        if expected_body is None:
            assert provider_calls == []
            assert messages == []
            assert view is None
        else:
            assert provider_calls == [expected_body]
            assert len(messages) == 1
            assert messages[0]["content"] == expected_body
            assert messages[0]["control_turn_id"] == turns[0].id
            assert view is not None and view.state == "settled"
        assert all("事故诱饵" not in message["content"] for message in messages)
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()
        await conversation.shutdown()
        sessions.close()
        store.close()


@pytest.mark.asyncio
async def test_wake_candidate_has_zero_timer_turn_and_formal_domain_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _Timer()
    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", lambda: timer)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    executions: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        executions.append(request)
        return ControlExecutionResult(response="unexpected")

    conversation = ConversationRuntime(store, execute)
    plugin_dirs = _copy_plugins(tmp_path)
    manager = PluginManager(
        plugin_dirs=plugin_dirs,
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    await manager.load_all()
    root = manager.current_snapshot.composition_root
    assert root is not None
    now = datetime.now(UTC)
    source = root.context.require(CONTENT_SOURCE).bind("candidate-source")
    _ = source.submit(
        "batch:candidate",
        (
            {
                "item_id": "content:candidate",
                "revision": "1",
                "payload": {"kind": "candidate"},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )
    _ = root.context.require(DRIFT_PROPOSALS).propose(
        "drift:candidate",
        "1",
        {},
        now,
        next_due=now + timedelta(minutes=5),
    )
    before_content = root.context.require(CONTENT_WAKE).snapshot(now)
    before_drift = root.context.require(DRIFT_WAKE).snapshot(now)
    wake_dir = next(path for path in plugin_dirs if path.name == "wake")
    with (wake_dir / "plugin.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# candidate wake revision\n")
    try:
        candidate = await manager.prepare_candidate("wake")
        assert candidate is not None
        assert timer.handles == []
        assert executions == []
        assert root.context.require(CONTENT_WAKE).snapshot(now) == before_content
        assert root.context.require(DRIFT_WAKE).snapshot(now) == before_drift
        await manager.discard_prepared("wake")
    finally:
        await manager.terminate_all()
        await conversation.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_real_root_selected_content_runs_one_scoped_react_and_not_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _Timer()
    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", lambda: timer)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    prepared: list[BeforeTurnCtx] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        turn_id = request.metadata["turnId"]
        assert isinstance(turn_id, str)
        ctx = BeforeTurnCtx(
            session_key=request.thread_id,
            channel=str(request.metadata["channel"]),
            chat_id=str(request.metadata["chatId"]),
            content=request.input,
            timestamp=datetime.now(UTC),
            retrieved_memory_block="",
            retrieval_trace_raw=None,
            history_messages=(),
            turn_id=turn_id,
        )
        await run_composition_lifecycle(CONTEXT_PREPARED_EVENT, ctx)
        prepared.append(ctx)
        return ControlExecutionResult(response="" if ctx.abort else "wake response")

    conversation = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=_copy_plugins(tmp_path),
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    await manager.load_all()
    root = manager.current_snapshot.composition_root
    assert root is not None
    content = root.context.require(CONTENT_SOURCE).bind("e2e-source")
    now = datetime.now(UTC)
    _ = content.submit(
        "batch:1",
        (
            {
                "item_id": "content:1",
                "revision": "1",
                "payload": {"kind": "fitbit", "preprocess_score": 0.9},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )
    _ = root.context.require(DRIFT_PROPOSALS).propose(
        "drift:1",
        "1",
        {"prompt": "reflect"},
        now,
        next_due=now + timedelta(minutes=5),
    )
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: len(timer.handles) == 1)
        timer.handles[0].fire()
        await _eventually(
            lambda: bool(store.list_turns("wake:default"))
            and store.list_turns("wake:default")[0].status is TurnStatus.COMPLETED
        )
        wake_content = root.context.require(CONTENT_WAKE)
        await _eventually(lambda: wake_content.selected() == ())

        turns = store.list_turns("wake:default")
        assert len(turns) == 1
        assert turns[0].final_response == "wake response"
        assert prepared[0].abort is False
        assert '"owner":"content"' in prepared[0].extra_hints[0]
        assert root.context.require(DRIFT_WAKE).selected() == ()
        drift_snapshot = root.context.require(DRIFT_WAKE).snapshot(datetime.now(UTC))
        assert len(cast(Sequence[object], drift_snapshot["proposals"])) == 1
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()
        await conversation.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_real_root_both_decline_is_quiet_but_keeps_control_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _Timer()
    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", lambda: timer)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    provider_calls = 0
    prepared: list[BeforeTurnCtx] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        nonlocal provider_calls
        turn_id = request.metadata["turnId"]
        assert isinstance(turn_id, str)
        ctx = BeforeTurnCtx(
            session_key=request.thread_id,
            channel=str(request.metadata["channel"]),
            chat_id=str(request.metadata["chatId"]),
            content=request.input,
            timestamp=datetime.now(UTC),
            retrieved_memory_block="",
            retrieval_trace_raw=None,
            history_messages=(),
            turn_id=turn_id,
        )
        await run_composition_lifecycle(CONTEXT_PREPARED_EVENT, ctx)
        prepared.append(ctx)
        if not ctx.abort:
            provider_calls += 1
            return ControlExecutionResult(response="unexpected")
        return ControlExecutionResult(response=ctx.abort_reply)

    conversation = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=_copy_plugins(tmp_path),
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    await manager.load_all()
    root = manager.current_snapshot.composition_root
    assert root is not None
    now = datetime.now(UTC)
    content = root.context.require(CONTENT_SOURCE).bind("quiet-source")
    _ = content.submit(
        "batch:quiet",
        (
            {
                "item_id": "content:quiet",
                "revision": "1",
                "payload": {"wake_action": "decline"},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )
    _ = root.context.require(DRIFT_PROPOSALS).propose(
        "drift:quiet",
        "1",
        {"wake_action": "decline"},
        now,
        next_due=now + timedelta(minutes=5),
    )
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: len(timer.handles) == 1)
        timer.handles[0].fire()
        await _eventually(
            lambda: bool(store.list_turns("wake:default"))
            and store.list_turns("wake:default")[0].status is TurnStatus.COMPLETED
        )

        turns = store.list_turns("wake:default")
        assert len(turns) == 1
        assert turns[0].input == "Check durable Wake duties."
        assert turns[0].final_response == ""
        assert [item.kind.value for item in turns[0].items] == [
            "userMessage",
            "assistantMessage",
        ]
        assert provider_calls == 0
        assert prepared[0].abort is True and prepared[0].abort_reply == ""
        assert root.context.require(CONTENT_WAKE).selected() == ()
        assert root.context.require(DRIFT_WAKE).selected() == ()
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()
        await conversation.shutdown()
        store.close()
