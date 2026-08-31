import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.control.context import running_turn_id
from agent.core.passive_turn import (
    _collect_current_akashic_push_media,
    _persistence_from_metadata,
)
from agent.core.runtime_support import SessionLike, TurnRunResult
from agent.looping.core import AgentLoop, _supports_stream_events
from agent.looping.interrupt import ActiveTurnState
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.types import BeforeReasoningCtx, BeforeTurnCtx
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from agent.context import ContextBuilder
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)
from agent.looping.session_lane import SessionLaneRegistry
from agent.persona import reset_veda
from agent.plugin_composition import LLMResponse, ToolCall
from plugins.compaction.engine import (
    CommittedContextUnit,
    ContextPayloadSegments,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.web_fetch import WebFetchTool
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from bus.events_lifecycle import TurnCommitted
from core.error_context import current_session_key
from bootstrap.wiring import wire_turn_lifecycle
from plugins.compaction.runtime import CompactionProjection
from session.store import CompactionHead
from tests.provider_fakes import ProviderContextBudgetStub
from tests.model_plugin_fakes import (
    BoundChatModelFake,
    bind_test_model_snapshot,
    build_test_chat_models,
    build_test_model_store,
)
from tests.compaction_fakes import install_test_projection


class _NoopTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "ok"


class _Provider(ProviderContextBudgetStub):
    async def chat(self, **kwargs):
        return LLMResponse(content="ok", tool_calls=[])


class _FakeMemoryEngine:
    def read_self(self) -> str:
        return ""

    def get_memory_context(self) -> str:
        return ""

    def has_long_term_memory(self) -> bool:
        return False


class _MandatoryCompactionRuntime:
    async def projection(self, session, *, prefix, current_anchor, pending):
        messages = getattr(session, "messages", [])
        history = (
            [dict(message) for message in messages]
            if isinstance(messages, list)
            else []
        )
        units: tuple[CommittedContextUnit, ...] = ()
        if history:
            ids = tuple(f"pipeline-message-{index}" for index in range(len(history)))
            units = (
                CommittedContextUnit(
                    source_from_seq=0,
                    consolidated_through_seq=len(history) - 1,
                    source_message_ids=ids,
                    messages=tuple(history),
                    message_refs=tuple(
                        (message_id, index) for index, message_id in enumerate(ids)
                    ),
                ),
            )
        return CompactionProjection(
            segments=ContextPayloadSegments(
                prefix=tuple(prefix),
                committed_units=units,
                current_anchor=tuple(current_anchor),
                pending=tuple(pending),
            ),
            active=None,
            head=CompactionHead(
                session_key=str(getattr(session, "key", "pipeline-session")),
                parent_generation=0,
                next_generation=1,
            ),
        )

    async def recover_pending(self, session):
        return None

    async def commit_checkpoint(self, *args, **kwargs):
        raise AssertionError("test compaction gate unexpectedly attempted a commit")


class _TestOutboundPort:
    async def dispatch(self, _outbound: object) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(
            delivery_id="test-delivery",
            status=ChannelDeliveryStatus.DELIVERED,
        )


def test_stream_events_support_realtime_private_channels():
    assert _supports_stream_events("telegram", "123")
    assert not _supports_stream_events("telegram", "-1001")
    assert not _supports_stream_events("telegram", "@alice")
    assert _supports_stream_events("akashic", "shared-chat")
    assert not _supports_stream_events("web", "retired-chat")
    assert not _supports_stream_events("mobile", "retired-chat")
    assert not _supports_stream_events("qq", "123")
    assert not _supports_stream_events("cli", "direct")


def test_akashic_push_media_is_collected_only_for_the_current_session():
    media: list[str] = []
    _collect_current_akashic_push_media(
        media,
        {
            "target_channel": "akashic",
            "target_chat_id": "chat",
            "image": "artifact:image",
        },
        channel="akashic",
        chat_id="chat",
    )
    _collect_current_akashic_push_media(
        media,
        {
            "target_channel": "telegram",
            "target_chat_id": "chat",
            "file": "artifact:file",
        },
        channel="akashic",
        chat_id="chat",
    )

    assert media == ["artifact:image"]


def test_stream_event_sink_respects_suppression_flag():
    loop = object.__new__(AgentLoop)
    loop._event_bus = EventBus()
    msg = InboundMessage(
        channel="telegram",
        sender="u",
        chat_id="123",
        content="hello",
        metadata={"suppress_stream_events": True},
    )

    assert AgentLoop._build_stream_event_sink(loop, msg) is None


@pytest.mark.asyncio
async def test_process_direct_accepts_generic_effect_metadata():
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = None
    loop._process = AsyncMock(
        return_value=OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )

    result = await AgentLoop.process_direct_message(
        loop,
        content="天气",
        session_key="scheduler:job",
        channel="telegram",
        chat_id="123",
        metadata={
            "omit_user_turn": True,
            "effects": {"post_commit": "suppress"},
            "disabled_prompt_sections": ["memory"],
        },
        disabled_tools=["message_push"],
    )

    msg = loop._process.await_args.args[0]
    assert result.content == "ok"
    assert msg.metadata == {
        "omit_user_turn": True,
        "effects": {"post_commit": "suppress"},
        "disabled_prompt_sections": ["memory"],
        "suppress_stream_events": True,
        "disabled_tools": ["message_push"],
    }
    assert loop._process.await_args.kwargs["dispatch_outbound"] is False


@pytest.mark.asyncio
async def test_process_direct_in_memory_metadata_has_no_history_or_persistence():
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = None
    loop._process = AsyncMock(
        return_value=OutboundMessage(
            channel="scheduler",
            chat_id="job-1",
            content="ok",
        )
    )

    await AgentLoop.process_direct_message(
        loop,
        content="天气",
        session_key="scheduler:job-1",
        channel="scheduler",
        chat_id="job-1",
        metadata={
            "omit_user_turn": True,
            "omit_assistant_turn": True,
            "skip_session_history": True,
            "effects": {"post_commit": "suppress"},
        },
    )

    msg = loop._process.await_args.args[0]
    assert msg.metadata == {
        "omit_user_turn": True,
        "omit_assistant_turn": True,
        "skip_session_history": True,
        "effects": {"post_commit": "suppress"},
        "suppress_stream_events": True,
    }
    persistence = _persistence_from_metadata(msg.metadata)
    assert persistence.persist_user is False
    assert persistence.persist_assistant is False


@pytest.mark.asyncio
async def test_process_direct_runs_concurrently_with_another_session():
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = None
    events: list[str] = []
    passive_started = asyncio.Event()
    direct_started = asyncio.Event()
    release_passive = asyncio.Event()

    async def _process(
        msg: InboundMessage,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        events.append(f"start:{key}")
        if key == "cli:1":
            passive_started.set()
            await release_passive.wait()
        else:
            direct_started.set()
        events.append(f"end:{key}")
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=key,
        )

    loop._process = _process
    passive_msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="hello",
    )
    passive_task = asyncio.create_task(
        AgentLoop._process_with_runtime_admission(loop, passive_msg)
    )
    await passive_started.wait()
    direct_task = asyncio.create_task(
        AgentLoop.process_direct(
            loop,
            content="天气",
            session_key="scheduler:job",
            channel="telegram",
            chat_id="123",
        )
    )
    await asyncio.wait_for(direct_started.wait(), timeout=1)

    assert events == ["start:cli:1", "start:scheduler:job", "end:scheduler:job"]
    assert not passive_task.done()
    release_passive.set()

    await asyncio.gather(passive_task, direct_task)

    assert events == [
        "start:cli:1",
        "start:scheduler:job",
        "end:scheduler:job",
        "end:cli:1",
    ]
    assert loop._session_lanes._states == {}


@pytest.mark.asyncio
async def test_process_direct_waits_for_the_same_session_lane():
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = None
    events: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def _process(
        msg: InboundMessage,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        events.append(f"start:{key}:{msg.content}")
        if msg.content == "hello":
            first_started.set()
            await release_first.wait()
        events.append(f"end:{key}:{msg.content}")
        return OutboundMessage(msg.channel, msg.chat_id, key)

    loop._process = _process
    passive_msg = InboundMessage("cli", "u", "1", "hello")
    passive_task = asyncio.create_task(
        AgentLoop._process_with_runtime_admission(loop, passive_msg)
    )
    await first_started.wait()
    direct_task = asyncio.create_task(
        AgentLoop.process_direct(
            loop,
            content="second",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
        )
    )

    await asyncio.sleep(0.01)
    assert events == ["start:cli:1:hello"]
    assert not direct_task.done()
    release_first.set()
    await asyncio.gather(passive_task, direct_task)

    assert events == [
        "start:cli:1:hello",
        "end:cli:1:hello",
        "start:cli:1:second",
        "end:cli:1:second",
    ]
    assert loop._session_lanes._states == {}


@pytest.mark.asyncio
async def test_process_direct_waits_for_explicit_busy_session_lane():
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = None
    events: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def _process(
        msg: InboundMessage,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        _ = busy_session_key, dispatch_outbound
        key = session_key or msg.session_key
        events.append(f"start:{key}")
        if key == "cli:1":
            first_started.set()
            await release_first.wait()
        events.append(f"end:{key}")
        return OutboundMessage(msg.channel, msg.chat_id, key)

    loop._process = _process
    passive_task = asyncio.create_task(
        AgentLoop._process_with_runtime_admission(
            loop,
            InboundMessage("cli", "u", "1", "hello"),
        )
    )
    await first_started.wait()
    direct_task = asyncio.create_task(
        AgentLoop.process_direct(
            loop,
            content="scheduled",
            session_key="scheduler:job",
            busy_session_key="cli:1",
            channel="cli",
            chat_id="1",
        )
    )

    await asyncio.sleep(0.01)
    assert events == ["start:cli:1"]
    assert not direct_task.done()
    release_first.set()
    await asyncio.gather(passive_task, direct_task)

    assert events == [
        "start:cli:1",
        "end:cli:1",
        "start:scheduler:job",
        "end:scheduler:job",
    ]
    assert loop._session_lanes._states == {}


@pytest.mark.asyncio
async def test_cancelled_session_lane_waiter_does_not_block_reentry():
    lanes = SessionLaneRegistry()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def hold_first() -> None:
        async with lanes.hold("programmatic:one"):
            first_entered.set()
            await release_first.wait()

    async def wait_for_same_lane() -> None:
        async with lanes.hold("programmatic:one"):
            raise AssertionError("cancelled waiter entered the lane")

    first = asyncio.create_task(hold_first())
    await first_entered.wait()
    waiter = asyncio.create_task(wait_for_same_lane())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_first.set()
    await first
    assert lanes._states == {}
    async with lanes.hold("programmatic:one"):
        assert list(lanes._states) == ["programmatic:one"]
    assert lanes._states == {}


@pytest.mark.asyncio
async def test_process_uses_busy_session_key_for_processing_state(tmp_path: Path):
    loop = _make_loop(tmp_path)
    state = MagicMock()
    loop._processing_state = state  # type: ignore[attr-defined]
    loop._react = AsyncMock(  # type: ignore[method-assign]
        return_value=OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="天气",
    )

    async with bind_test_model_snapshot(_Provider()):
        outbound = await loop._process(
            msg,
            session_key="scheduler:job",
            busy_session_key="telegram:123",
            dispatch_outbound=False,
        )

    assert outbound.content == "ok"
    state.enter.assert_called_once_with("telegram:123")
    state.exit.assert_called_once_with("telegram:123")
    loop._react.assert_awaited_once()  # type: ignore[attr-defined]
    call = loop._react.await_args  # type: ignore[attr-defined]
    assert call.args == (msg, "scheduler:job")
    assert call.kwargs["chat_models"] is not None
    assert call.kwargs["model_id"] is None
    assert call.kwargs["reasoning_effort"] is None
    assert call.kwargs["dispatch_outbound"] is False
    assert call.kwargs["command_admitted"] is True


@pytest.mark.asyncio
async def test_process_restores_session_context(tmp_path: Path):
    loop = _make_loop(tmp_path)
    loop._react = AsyncMock(  # type: ignore[method-assign]
        return_value=OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="天气",
    )
    token = current_session_key.set("outer-session")
    try:
        async with bind_test_model_snapshot(_Provider()):
            await loop._process(msg, dispatch_outbound=False)
        assert current_session_key.get() == "outer-session"
    finally:
        current_session_key.reset(token)


@pytest.mark.asyncio
async def test_process_restores_session_context_after_core_failure(tmp_path: Path):
    loop = _make_loop(tmp_path)
    state = MagicMock()
    loop._processing_state = state  # type: ignore[attr-defined]
    loop._react = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("core failed")
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="天气",
    )
    token = current_session_key.set("outer-session")
    try:
        with pytest.raises(RuntimeError, match="core failed"):
            async with bind_test_model_snapshot(_Provider()):
                await loop._process(msg, dispatch_outbound=False)
        assert current_session_key.get() == "outer-session"
        state.enter.assert_called_once_with("telegram:123")
        state.exit.assert_called_once_with("telegram:123")
    finally:
        current_session_key.reset(token)


@pytest.mark.asyncio
async def test_process_does_not_run_removed_web_fetch_spill_cleanup(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    loop = _make_loop(tmp_path)
    loop.tools.register(WebFetchTool(requester=cast(Any, object())))
    loop._react = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("provider failed")
    )
    msg = InboundMessage(
        channel="web",
        sender="user",
        chat_id="desktop-chat",
        content="继续",
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        async with bind_test_model_snapshot(_Provider()):
            await loop._process(msg, dispatch_outbound=False)

    assert "web_fetch_cleanup" not in caplog.text


def _make_loop(tmp_path: Path) -> AgentLoop:
    _ = reset_veda(tmp_path)
    tools = ToolRegistry()
    tools.register(_NoopTool())
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            context=ContextBuilder(tmp_path),
            outbound_port=cast(Any, _TestOutboundPort()),
        ),
        AgentLoopConfig(),
    )
    loop.session_manager.get_or_create.return_value = SimpleNamespace(metadata={})
    loop._runtime_snapshot_store = build_test_model_store(_Provider())
    return loop


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_phase", ("before_turn", "before_reasoning"))
async def test_runtime_abort_never_enters_model_execution(
    tmp_path: Path,
    abort_phase: str,
) -> None:
    loop = _make_loop(tmp_path)
    session = MagicMock()
    session.key = "cli:abort"
    session.metadata = {}
    session.messages = []
    session.get_history.return_value = []
    loop.session_manager.get_or_create.return_value = session

    class _RejectingChatModels:
        calls = 0

        def execution(self, **_selection: object) -> object:
            self.calls += 1
            raise AssertionError("abort must not enter model execution")

    chat_models = _RejectingChatModels()
    loop._runtime_snapshot_store = build_test_model_store(
        _Provider(),
        chat_models=chat_models,
    )

    async def abort(ctx: object) -> object:
        ctx.abort = True  # type: ignore[attr-defined]
        ctx.abort_reply = f"{abort_phase} stopped"  # type: ignore[attr-defined]
        return ctx

    event_type = BeforeTurnCtx if abort_phase == "before_turn" else BeforeReasoningCtx
    loop._event_bus.on(event_type, abort)

    result = await loop._process_with_runtime_admission(
        InboundMessage("cli", "user", "abort", "hello"),
        dispatch_outbound=False,
    )

    assert result.content == f"{abort_phase} stopped"
    assert chat_models.calls == 0


@pytest.mark.asyncio
async def test_runtime_admission_runs_normal_tool_loop_through_chat_models(
    tmp_path: Path,
) -> None:
    """真实准入链只从 CHAT_MODELS 取得一次 Turn-local 模型绑定。"""

    class _ToolProvider(ProviderContextBudgetStub):
        def __init__(self) -> None:
            self.responses = [
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall("call-1", "noop", {})],
                ),
                LLMResponse(content="done", tool_calls=[]),
            ]
            self.calls = 0

        async def chat(self, **_kwargs: object) -> LLMResponse:
            self.calls += 1
            return self.responses.pop(0)

    loop = _make_loop(tmp_path)
    session = MagicMock()
    session.key = "cli:normal"
    session.created_at = datetime(2026, 8, 29, tzinfo=UTC)
    session.metadata = {}
    session.messages = []
    session.get_history.return_value = []
    session.add_message.side_effect = lambda role, content, **kwargs: {
        "role": role,
        "content": content,
        **kwargs,
    }
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)
    provider = _ToolProvider()
    chat_models = build_test_chat_models(provider)
    loop._runtime_snapshot_store = build_test_model_store(
        provider,
        chat_models=chat_models,
    )

    result = await loop._process_with_runtime_admission(
        InboundMessage("cli", "user", "normal", "use noop"),
        dispatch_outbound=False,
    )

    assert result.content == "done"
    assert provider.calls == 2
    assert chat_models.execution_calls == 1  # type: ignore[attr-defined]


def test_agent_loop_fanouts_turn_committed_from_passive_turn(tmp_path: Path):
    loop = _make_loop(tmp_path)
    turn_events: list[TurnCommitted] = []
    loop._event_bus.on(TurnCommitted, lambda event: turn_events.append(event))
    session = MagicMock()
    session.key = "cli:1"
    session.messages = []
    session.metadata = {}
    session.get_history = MagicMock(return_value=[])

    def add_message(role: str, content: str, **kwargs: object) -> dict[str, object]:
        message = {"role": role, "content": content, **kwargs}
        session.messages.append(message)
        return message

    session.add_message = MagicMock(side_effect=add_message)
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)
    loop._reasoner.run_turn = AsyncMock(
        return_value=TurnRunResult(
            reply="ok",
            tool_chain=[
                {
                    "text": "",
                    "calls": [
                        {
                            "name": "noop",
                            "arguments": {"x": 1},
                            "result": "done",
                        }
                    ],
                }
            ],
            context_retry={
                "react_stats": {
                    "iteration_count": 1,
                    "turn_input_sum_tokens": 100,
                }
            },
        )
    )

    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")

    async def _process_and_drain() -> None:
        provider = _Provider()
        await loop._react(
            msg,
            msg.session_key,
            chat_models=build_test_chat_models(provider),
        )
        await loop._event_bus.drain()
        await loop._event_bus.aclose()

    asyncio.run(_process_and_drain())

    assert turn_events
    turn_event = turn_events[0]
    assert turn_event.session_key == "cli:1"
    assert turn_event.persisted_user_message == "hello"
    assert turn_event.assistant_response == "ok"
    assert turn_event.tool_chain_raw[0]["calls"][0]["name"] == "noop"
    assert turn_event.react_stats["iteration_count"] == 1
    assert turn_event.react_stats["turn_input_sum_tokens"] == 100


@pytest.mark.asyncio
async def test_agent_loop_afterstep_fires_with_turn_lifecycle_wiring(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session_key = "cli:123"
    loop._active_turn_states[session_key] = ActiveTurnState(session_key=session_key)
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(loop._event_bus),
        active_turn_states=loop.active_turn_states,
    )
    msg = InboundMessage(channel="cli", sender="u", chat_id="123", content="你好")
    session = SimpleNamespace(
        key=session_key,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        messages=[],
        metadata={},
        last_consolidated=0,
        get_history=MagicMock(return_value=[]),
        history_units=MagicMock(return_value=[]),
        add_message=MagicMock(),
    )
    loop.session_manager.get_or_create.return_value = session

    provider = _Provider()
    await loop._reasoner.run_turn(
        msg=msg,
        session=cast(SessionLike, session),
        agent_model=BoundChatModelFake(provider),
        fallback_model=BoundChatModelFake(provider),
        base_history=[],
    )

    state = loop._active_turn_states[session_key]
    assert state.partial_reply == "ok"
    assert state.tools_used == []
    assert state.tool_chain_partial == []
