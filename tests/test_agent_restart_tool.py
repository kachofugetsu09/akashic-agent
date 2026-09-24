from __future__ import annotations

import asyncio
import json
import secrets
import shutil
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.config_input import save_config

from agent.plugin_composition import CompositionRoot, Context, ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugins.manager import PluginManager
from agent.restart import RESTART_GATE, RestartGate, RestartRejectedError
from bus.event_bus import EventBus
from agent.control.frame_book import FrameBook
from bootstrap.app_server import build_control_service
from bootstrap.tools import CoreRuntime
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.control.connection import NdjsonConnection
from plugins.message_push.restart import PendingRestart, RestartTool
from plugins.tools.api import (
    CallSource,
    ContentPart,
    MessageReply,
    durable_call_key,
)
from plugins.content.plugin import check_text
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from plugins.programmatic.control import AdmitParams, PROGRAMMATIC, SendParams
from session.log import MessageLog, MessageReader
from session.artifact_store import ArtifactStore
from session.message import (
    CallRef, ContentReferences, Input, Message, Output, ToolCall, ToolResult, freeze_json,
)

STARTUP_PROBE_EMIT = ServiceKey[Callable[[], Awaitable[None]]]("test.startup_probe.emit")


class _FixtureTransport:
    """Only the connection identity is consumed by programmatic control."""

    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id


def _copy_plugin_sources(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            root / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )


def _write_restart_provider(
    root: Path, *, reload_fixture: bool = False, shared_event: bool = False,
    state_root: Path | None = None,
) -> None:
    provider = root / "restart_provider"
    provider.mkdir()
    if reload_fixture and state_root is None:
        raise ValueError("reload fixture needs state root")
    fixture_imports = (
        "import asyncio\n"
        "from pathlib import Path\n"
        "from agent.plugin_composition import RUNTIME_STARTING\n"
        "from plugins.delivery.api import FINAL_OUTPUT_DELIVERY\n"
        if reload_fixture or shared_event else ""
    )
    inject = "(FINAL_OUTPUT_DELIVERY,)" if reload_fixture else "()"
    fixture_setup = (
        "    await ctx.on(RUNTIME_STARTING, lambda _event: None)\n"
        if shared_event else ""
    ) + (f"""
    delivery = ctx.require(FINAL_OUTPUT_DELIVERY)

    class Waiter:
        async def wait(self, reader, turn):
            return None

    waiter = Waiter()
    delivery.register("reload-probe", waiter)
    await ctx.effect(
        lambda: lambda: delivery.unregister("reload-probe", waiter),
        label="reload-probe-delivery",
    )

    def prepared(_event):
        marker = Path({str(state_root)!r}, "await-prepare")
        if marker.exists():
            generation_id = ctx.runtime.generation_id
            asyncio.get_running_loop().call_soon(
                lambda: Path({str(state_root)!r}, "prepared").write_text(generation_id),
            )

    await ctx.on(RUNTIME_STARTING, prepared)
""" if reload_fixture else "")
    (provider / "plugin.py").write_text(
        f"""
from contextlib import asynccontextmanager
from types import SimpleNamespace
from agent.plugin_composition import CHAT_MODELS, ServiceKey
{fixture_imports}
from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities,
    ToolCall,
)
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore

api_version = 3
name = "restart_provider"
version = "1.0.0"
inject = {inject}

async def apply(ctx):
{fixture_setup}
    calls = []
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()

    class Driver:
        max_tool_schemas = None

        def estimate_context_tokens(self, messages, tools):
            return 10

        async def complete(self, request):
            calls.append(request)
            if len(calls) in (1, 4):
                return LLMResponse(None, [ToolCall(
                    "search-call", "tool_search", {{"query": "agent_restart"}},
                )])
            if len(calls) in (2, 5):
                return LLMResponse(None, [ToolCall(
                    "restart-call", "agent_restart", {{"reason": "fixture"}},
                )])
            return LLMResponse("final fixture reply")

    descriptor = BoundModelDescriptor(
        binding_id="fixture-model", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture",
        driver_contract_version="1", auth_identity="fixture", model="fixture",
        role="agent", reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="fixture",
    )
    model = _BoundChat(descriptor, Driver(), store)

    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            yield SimpleNamespace(chat=lambda role: model)

    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
    await ctx.provide(ServiceKey("fixture.calls"), calls)
"""
    )


def _write_blocking_sender(root: Path, state_root: Path, *, reject_first: bool = False) -> None:
    sender = root / "fixture_sender"
    sender.mkdir()
    (sender / "plugin.py").write_text(
        f"""
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from agent.plugin_composition import ServiceKey
from plugins.delivery.api import Receipt
from plugins.delivery.senders import DELIVERY_SENDERS

api_version = 3
name = "fixture_sender"
version = "1.0.0"
inject = (DELIVERY_SENDERS,)
STATE_ROOT = {str(state_root)!r}
REJECT_FIRST = {reject_first!r}

async def apply(ctx):
    Path(STATE_ROOT).mkdir(parents=True, exist_ok=True)

    class Sender:
        idempotent = True

        async def send(self, key, address, message):
            count_path = Path(STATE_ROOT, "send-count")
            count = int(count_path.read_text()) if count_path.exists() else 0
            count += 1
            count_path.write_text(str(count))
            if REJECT_FIRST and count == 1:
                Path(STATE_ROOT, "rejected").write_text(message.message_id)
                return Receipt(status="rejected", provider_ids=("fixture",), error="fixture rejection")
            Path(STATE_ROOT, "started").write_text("1")
            while not Path(STATE_ROOT, "release").exists():
                await asyncio.sleep(0.01)
            Path(STATE_ROOT, "calls").write_text(message.message_id)
            return Receipt(status="delivered", provider_ids=("fixture",))

        async def query(self, key, address):
            return None

    @asynccontextmanager
    async def open():
        yield Sender()

    await ctx.require(DELIVERY_SENDERS).register(
        ctx, name="test", idempotent=True, open=open,
    )
"""
    )


def _write_startup_probe(root: Path, run_id: str, state_root: Path) -> None:
    # 每次 boot 的输入归 fixture 所有；重启不靠改插件源码替换 stable。
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / "run-id").write_text(run_id)
    probe = root / "startup_probe"
    probe.mkdir()
    (probe / "plugin.py").write_text(
        f"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.control_frames import CONTROL_FRAMES
from agent.plugin_composition.messages import MESSAGE_WRITERS
from plugins.delivery.api import FINAL_OUTPUT_DELIVERY
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.message import CallRef, Input, Output, ToolCall, ToolResult

api_version = 3
name = "startup_probe"
version = "1.0.0"
inject = (MESSAGE_WRITERS, BINDINGS, TOOLS, ALL_TOOLS, FINAL_OUTPUT_DELIVERY, TURN_PROJECTION, CONTROL_FRAMES)
STATE_ROOT = {str(state_root)!r}


class Waiter:
    async def wait(self, reader, turn):
        Path(STATE_ROOT).mkdir(parents=True, exist_ok=True)
        Path(STATE_ROOT, "delivered-" + reader.session_id.replace(":", "_")).write_text("1")
        return None


async def apply(ctx):
    delivery = ctx.require(FINAL_OUTPUT_DELIVERY)
    waiter = Waiter()
    delivery.register("startup-probe", waiter)
    await ctx.effect(
        lambda: lambda: delivery.unregister("startup-probe", waiter),
        label="startup-probe-delivery",
    )
    writers = ctx.require(MESSAGE_WRITERS)
    tools = ctx.require(TOOLS)
    bindings = ctx.require(BINDINGS)

    def append_after_prepare():
        run_id = Path(STATE_ROOT, "run-id").read_text()
        binding = tools.bind(ctx.require(ALL_TOOLS)().select("agent_restart"), bindings)
        session = "startup-probe:" + run_id
        inputs = writers.bind(
            ctx, author="user", source="startup-probe", body_types=(Input,), content={{}},
        )(session)
        outputs = writers.bind(
            ctx, author="assistant", source="startup-probe", body_types=(Output,),
            content={{}}, check_call=lambda call: None,
        )(session)
        inputs.append("startup-input-" + run_id, Input(()))
        call = outputs.append(
            "startup-call-" + run_id,
            Output((ToolCall(binding, {{"reason": "startup"}}),), "continue"),
        )
        results = writers.bind(
            ctx, author="tool", source="startup-probe", body_types=(ToolResult,), content={{}},
        )(session, call_ref=CallRef(call.message_id, 0))
        def append_result():
            results.append(
                "startup-result-" + run_id,
                ToolResult(CallRef(call.message_id, 0), "success", ()),
            )
            outputs.append("startup-final-" + run_id, Output((), "complete"))

        asyncio.get_running_loop().call_soon(append_result)

    async def emit():
        async with ctx.runtime_scope():
            append_after_prepare()

    await ctx.provide(ServiceKey("test.startup_probe.emit"), emit)
"""
    )


@asynccontextmanager
async def _restart_application(
    tmp_path: Path, gate: RestartGate, *, channel: bool,
    source_tag: str | None = None, reject_first: bool = False,
    startup_run: str | None = None, reload_probe: bool = False,
    message_log: MessageLog | None = None,
):
    sources = tmp_path / ("plugins" if source_tag is None else f"plugins-{source_tag}")
    names = (
        "sources",
        "content",
        "assets",
        "commands",
        "standard_tools",
        "context",
        "tools",
        "conversation",
        "react",
        "turn_projection",
        "reply",
        "reply_program",
        "tool_search",
        "delivery",
        "message_push",
    )
    names += ("delivery_policy",) if channel else ("programmatic",)
    _copy_plugin_sources(sources, names)
    _write_restart_provider(
        sources,
        reload_fixture=reload_probe,
        state_root=tmp_path / "reload-state" if reload_probe else None,
    )
    if startup_run is not None:
        _write_startup_probe(sources, startup_run, tmp_path / "startup-state")
    if channel:
        _write_blocking_sender(
            sources, tmp_path / "sender-state", reject_first=reject_first,
        )
    owns_log = message_log is None
    log = MessageLog(tmp_path / "sessions.db") if message_log is None else message_log
    artifact_store = ArtifactStore(tmp_path / "sessions.db")
    context_config = tmp_path / "workspace/plugin-data/context-builtin"
    context_config.parent.mkdir(parents=True, exist_ok=True)
    save_config(context_config, {"prompt_sources": {"skills": "standard_tools"}})
    host = PluginManager(
        [sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache", message_log=log,
        restart_gate=gate,
        channel_attachment_store=ChannelAttachmentArtifactStore(
            workspace=tmp_path / "workspace", metadata_store=artifact_store
        ),
    )
    try:
        await host.load_all()
        await host.start_runtime()
        yield log, host
    finally:
        await host.terminate_all()
        artifact_store.close()
        if owns_log:
            log.close()


def _restart_context(root: CompositionRoot) -> Context:
    """Get the active restart child that owns the registered tool."""
    for fiber in root._fibers.values():
        if (
            fiber.name == "restart"
            and fiber.runtime is not None
            and fiber.runtime.plugin_id == "message_push"
        ):
            return fiber.context
    raise AssertionError("live Root 缺少 message_push restart owner")


@asynccontextmanager
async def _live_root(host: PluginManager):
    """Read the one formal Root used by local plugin lifecycle tests."""
    root = host.live_root
    assert root is not None
    yield root


def _source(message_id: str = "call-a", reason: str = "reload") -> CallSource:
    call_ref = CallRef(message_id, 0)
    message = Message(
        message_id,
        "session-a",
        0,
        datetime.now(timezone.utc),
        "agent",
        "conversation",
        Output((ToolCall("binding-a", {"reason": reason}),), "continue"),
    )
    return CallSource(call_ref, (message,))


def _commit_recorder(commits: list[str], committed: asyncio.Event):
    def record(request_id: str) -> None:
        commits.append(request_id)
        committed.set()

    return record


def _pending(source: CallSource) -> PendingRestart:
    arguments = freeze_json({"reason": "reload"})
    assert isinstance(arguments, Mapping)
    return PendingRestart(
        source.call_ref,
        durable_call_key(source.call_ref),
        cast(Mapping[str, object], arguments),
    )


@pytest.mark.asyncio
async def test_restart_tool_binds_invoke_to_prepared_durable_call() -> None:
    tool = RestartTool(
        RestartGate(boot_id="fixture-boot", supervised=True, commit=lambda _: None), FrameBook(),
    )
    source = _source()
    rejected = await tool.prepare({"reason": "reload", "extra": True}, source)
    assert isinstance(rejected, str) and "只能包含 reason" in rejected
    assert tool._prepared is None
    prepared = await tool.prepare({"reason": " reload "}, source)
    assert isinstance(prepared, Mapping)
    pending = tool._prepared
    assert pending is not None
    assert prepared == {"reason": "reload"}
    assert pending.effect_key == durable_call_key(source.call_ref)
    assert pending.arguments == prepared
    assert tool.idempotent is False

    result = await tool.invoke(pending.effect_key, prepared)
    assert result.outcome == "success"
    assert result.parts[0].value == "已安排在本轮最终回复送达后重启。"

    with pytest.raises(RestartRejectedError, match="durable key"):
        await tool.invoke("message:[\"other\",0]", prepared)
    with pytest.raises(RestartRejectedError, match="参数"):
        await tool.invoke(pending.effect_key, {"reason": "other"})


@pytest.mark.asyncio
async def test_restart_prepare_is_idempotent_only_for_same_call_and_arguments() -> None:
    tool = RestartTool(
        RestartGate(boot_id="fixture-boot", supervised=True, commit=lambda _: None), FrameBook(),
    )
    source = _source()
    await tool.prepare({"reason": "reload"}, source)
    first = tool._prepared
    assert first is not None

    await tool.prepare({"reason": "reload"}, source)
    assert tool._prepared is first

    with pytest.raises(RestartRejectedError, match="参数不一致"):
        await tool.prepare({"reason": "different"}, source)
    with pytest.raises(RestartRejectedError, match="多个 restart"):
        await tool.prepare({"reason": "reload"}, _source("call-b"))


@pytest.mark.asyncio
async def test_restart_requires_prepare_and_query_is_unknown() -> None:
    tool = RestartTool(
        RestartGate(boot_id="fixture-boot", supervised=True, commit=lambda _: None), FrameBook(),
    )
    assert tool.idempotent is False
    assert await tool.query("message:[\"call-a\",0]") is None
    with pytest.raises(RestartRejectedError, match="prepare"):
        await tool.invoke("message:[\"call-a\",0]", {"reason": "reload"})


@pytest.mark.asyncio
async def test_unmanaged_runtime_does_not_register_restart_tool(tmp_path: Path) -> None:
    gate = RestartGate(boot_id="fixture-boot", supervised=False)
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(tmp_path, gate, channel=False) as (_log, host):
        root = host.live_root
        assert root is not None
        names = {ref.name for ref in root.context.require(ALL_TOOLS)().refs}
    assert "agent_restart" not in names


@pytest.mark.asyncio
async def test_restart_watcher_ignores_old_result_and_reads_new_after_prepare(
    tmp_path: Path,
) -> None:
    """新 watcher 只处理准备后的真实 ToolResult，旧结果不再触发。"""
    first_commits: list[str] = []
    first_committed = asyncio.Event()
    first_gate = RestartGate(
        boot_id="first-boot", supervised=True,
        commit=_commit_recorder(first_commits, first_committed),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(
        tmp_path, first_gate, channel=False, source_tag="baseline-first", startup_run="first",
    ) as (_log, host):
        root = host.live_root
        assert root is not None
        await root.context.require(STARTUP_PROBE_EMIT)()
        await asyncio.wait_for(first_committed.wait(), 2)
    assert len(first_commits) == 1

    second_commits: list[str] = []
    second_committed = asyncio.Event()
    second_gate = RestartGate(
        boot_id="second-boot", supervised=True,
        commit=_commit_recorder(second_commits, second_committed),
    )
    async with _restart_application(
        tmp_path, second_gate, channel=False, source_tag="baseline-second", startup_run="second",
    ) as (_log, host):
        root = host.live_root
        assert root is not None
        assert not second_commits
        await root.context.require(STARTUP_PROBE_EMIT)()
        await asyncio.wait_for(second_committed.wait(), 2)
        assert (tmp_path / "startup-state" / "delivered-startup-probe_second").exists()
    assert len(second_commits) == 1


@pytest.mark.asyncio
async def test_real_channel_restart_waits_for_cleanup_and_delivery_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 Reply/Delivery 链路不能在 cleanup 与首次选路之间误 abort。"""
    commits: list[str] = []
    committed = asyncio.Event()
    gate = RestartGate(
        boot_id="fixture-boot", supervised=True,
        commit=_commit_recorder(commits, committed),
        drain_timeout_s=2.0,
    )
    cleanup_blocked = asyncio.Event()
    cleanup_release = asyncio.Event()

    @asynccontextmanager
    async def controlled_cleanup(*_args, task, drain):
        try:
            yield
        finally:
            cleanup_blocked.set()
            await cleanup_release.wait()

    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(tmp_path, gate, channel=True) as (log, host):
        async with _live_root(host) as root:
            context = root.context
            execute = context.require(ServiceKey("reply.execute.v1"))
            monkeypatch.setitem(execute.keywords, "cleanup", controlled_cleanup)
            accept = context.require(CHANNEL_INPUT)
            await accept(
                "test:room", "input-1",
                ChannelInboundMessage(
                    "test", "user", "room", "restart now", datetime.now(timezone.utc), {},
                ),
            )
            await asyncio.wait_for(cleanup_blocked.wait(), 2)
            rows = log.reader("test:room").snapshot()
            assert rows and isinstance(rows[0].body, Input)
            assert not commits

            sender_state = tmp_path / "sender-state"

            async def wait_for_file(name: str) -> None:
                while not (sender_state / name).exists():
                    await asyncio.sleep(0.01)

            cleanup_release.set()
            await asyncio.wait_for(wait_for_file("started"), 2)
            assert not commits
            (sender_state / "release").write_text("1")
            await asyncio.wait_for(wait_for_file("calls"), 2)
            await asyncio.wait_for(committed.wait(), 2)
            rows = log.reader("test:room").snapshot()
            assert any(
                isinstance(row.body, Output) and row.body.finish == "complete"
                for row in rows
            ), (rows, root.receipt().incidents)
            assert len(commits) == 1
            assert (sender_state / "calls").read_text() == rows[-1].message_id


@pytest.mark.asyncio
async def test_real_channel_restart_reopens_after_rejected_delivery(
    tmp_path: Path,
) -> None:
    """首个真实发送被拒后必须释放 gate，下一次请求仍可提交。"""
    commits: list[str] = []
    committed = asyncio.Event()
    gate = RestartGate(
        boot_id="fixture-boot", supervised=True,
        commit=_commit_recorder(commits, committed),
        drain_timeout_s=2.0,
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(
        tmp_path, gate, channel=True, reject_first=True,
    ) as (log, host):
        async with _live_root(host) as root:
            accept = root.context.require(CHANNEL_INPUT)
            message = lambda input_id, text: ChannelInboundMessage(
                "test", "user", "room", text, datetime.now(timezone.utc), {},
            )
            await accept("test:room", "input-1", message("input-1", "reject once"))
            sender_state = tmp_path / "sender-state"

            async def wait_for_file(name: str) -> None:
                while not (sender_state / name).exists():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_file("rejected"), 2)

            async def wait_for_gate_drain() -> None:
                await gate.wait_until_open()
                while gate.permit_count:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_gate_drain(), 2)
            assert gate.permit_count == 0
            assert not commits

            await accept("test:room", "input-2", message("input-2", "retry now"))
            await asyncio.wait_for(wait_for_file("started"), 2)
            assert not commits
            (sender_state / "release").write_text("1")
            await asyncio.wait_for(wait_for_file("calls"), 2)
            await asyncio.wait_for(committed.wait(), 2)
            assert len(commits) == 1
            assert gate.permit_count == 0
            assert (sender_state / "calls").read_text()


@pytest.mark.asyncio
async def test_manager_reload_hands_late_tool_result_to_new_watcher(
    tmp_path: Path,
) -> None:
    """Manager 热重载后，旧 writer 的迟到结果只由新 watcher 提交一次。"""
    commits: list[str] = []
    committed = asyncio.Event()
    gate = RestartGate(
        boot_id="fixture-boot", supervised=True,
        commit=_commit_recorder(commits, committed),
        drain_timeout_s=2.0,
    )
    log = MessageLog(tmp_path / "sessions.db")
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with _restart_application(
            tmp_path, gate, channel=False, source_tag="reload-first", reload_probe=True,
            message_log=log,
        ) as (_old_log, host):
            session = "reload-probe:late"
            message_writer = log.writer(
                session, author="user", source="reload-probe", body_types=(Input,), content={},
            )
            output_writer = log.writer(
                session, author="assistant", source="reload-probe", body_types=(Output,),
                content={}, check_call=lambda call: None,
            )
            message_writer.append("late-input", Input(()))
            async with _live_root(host) as root:
                ctx = _restart_context(root)
                async with ctx.runtime_scope():
                    binding = ctx.require(TOOLS).bind(
                        ctx.require(ALL_TOOLS)().select("agent_restart"),
                        ctx.require(BINDINGS),
                    )
                    descriptor = log.read_binding(binding)
            metadata = descriptor["metadata"]
            assert isinstance(metadata, Mapping)
            tool_descriptor = metadata["tool"]
            assert isinstance(tool_descriptor, Mapping)
            assert tool_descriptor["name"] == "agent_restart"
            assert tool_descriptor["owner"] == "message_push"
            call = output_writer.append(
                "late-call", Output((ToolCall(binding, {"reason": "reload"}),), "continue"),
            )
            result_writer = log.writer(
                session, author="tool", source="reload-probe", body_types=(ToolResult,), content={},
                call_ref=CallRef(call.message_id, 0),
            )
            final_output = Output((), "complete")
            await host.terminate_all()
            async with _restart_application(
                tmp_path, gate, channel=False, source_tag="reload-second", reload_probe=True,
                message_log=log,
            ):
                assert any(
                    task.get_name() == "plugin-task:agent-restart-watcher"
                    for task in asyncio.all_tasks()
                )
                result_writer.append(
                    "late-result", ToolResult(CallRef(call.message_id, 0), "success", ()),
                )
                output_writer.append("late-final", final_output)
                rows = log.reader(session).snapshot()
                assert [row.seq for row in rows] == [0, 1, 2, 3]
                assert all(row.source == "reload-probe" for row in rows)
                assert isinstance(rows[2].body, ToolResult)
                assert isinstance(rows[3].body, Output)
                assert rows[3].body.finish == "complete"
                await asyncio.wait_for(committed.wait(), 2)
                assert len(commits) == 1
                assert gate.permit_count == 0
    finally:
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("supervised", [True, False], ids=["supervised", "unmanaged"])
async def test_restart_provider_update_preserves_live_root_and_restart_owner(
    tmp_path: Path, supervised: bool,
) -> None:
    """Replacing one provider keeps the restart owner and formal Root stable."""
    gate = RestartGate(
        boot_id="fixture-boot", supervised=supervised,
        commit=(lambda _request_id: None) if supervised else None,
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(tmp_path, gate, channel=False) as (_log, host):
        root = host.live_root
        provider = host.generation("restart_provider")
        restart_owner = host.generation("message_push")
        assert root is not None and provider is not None and restart_owner is not None
        selected = host._selection.read()
        watchers = {
            task for task in asyncio.all_tasks()
            if task.get_name() == "plugin-task:agent-restart-watcher"
        }
        tool_names = {ref.name for ref in root.context.require(ALL_TOOLS)().refs}
        assert ("agent_restart" in tool_names) is supervised
        assert bool(watchers) is supervised
        assert root.context.require(RESTART_GATE) is gate

        source = tmp_path / "plugins/restart_provider/plugin.py"
        code = source.read_text(encoding="utf-8")
        assert code.count('version = "1.0.0"') == 1
        source.write_text(code.replace('version = "1.0.0"', 'version = "1.0.1"'), encoding="utf-8")
        changed = await host.reconcile_changed()

        replacement = host.generation("restart_provider")
        assert any(
            item.get("plugin_id") == "restart_provider"
            and item.get("publication_state") == "active"
            for item in changed
        ), changed
        assert replacement is not None and replacement is not provider
        assert replacement.archive_ref != provider.archive_ref
        assert host._selection.read() != selected
        assert host.live_root is root
        assert host.generation("message_push") is restart_owner
        assert root.context.require(RESTART_GATE) is gate
        assert {ref.name for ref in root.context.require(ALL_TOOLS)().refs} == tool_names
        assert {
            task for task in asyncio.all_tasks()
            if task.get_name() == "plugin-task:agent-restart-watcher"
        } == watchers

@pytest.mark.asyncio
async def test_real_programmatic_restart_waits_for_frame_writer_drain_before_commit(
    tmp_path: Path,
) -> None:
    """归档工具成功落盘后，live programmatic provider 必须等待真实 frame drain。"""
    commits: list[str] = []
    committed = asyncio.Event()
    gate = RestartGate(
        boot_id="fixture-boot", supervised=True,
        commit=_commit_recorder(commits, committed),
        drain_timeout_s=2.0,
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(tmp_path, gate, channel=False) as (log, host):
        session = "programmatic:restart"
        frames = host._control_frames  # type: ignore[attr-defined]
        # build_control_service only reads these CoreRuntime fields in this fixture.
        core = cast(CoreRuntime, SimpleNamespace(
            plugin_manager=host,
            workspace=tmp_path / "workspace",
            message_log=log,
            control_frames=frames,
            channel_attachment_store=SimpleNamespace(resolve_refs=lambda _ids: ()),
        ))
        service = build_control_service(core)
        endpoint = "\0i750-" + secrets.token_hex(8)
        connection_done = asyncio.Event()
        drain_entered = asyncio.Event()
        drain_release = asyncio.Event()
        writer_state: dict[str, bytes] = {}

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            original_write = writer.write
            original_drain = writer.drain

            def write(payload: bytes) -> None:
                writer_state["payload"] = payload
                original_write(payload)

            async def drain() -> None:
                payload = writer_state.get("payload", b"")
                if (
                    b'"kind":"output"' in payload
                    and b'"finish":"complete"' in payload
                    and not drain_release.is_set()
                ):
                    drain_entered.set()
                    await drain_release.wait()
                await original_drain()

            writer.write = write  # type: ignore[method-assign]
            writer.drain = drain  # type: ignore[method-assign]
            connection = NdjsonConnection(
                reader,
                writer,
                service,
                max_message_bytes=2 * 1024 * 1024,
                max_pending_requests=128,
                outbound_queue_size=512,
                control_frames=frames,
            )
            try:
                await connection.run()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionResetError, BrokenPipeError):
                    # 断言失败时客户端可能带着未读帧关闭；仍需完成连接清理。
                    pass
                finally:
                    connection_done.set()

        server = await asyncio.start_unix_server(accept, path=endpoint)
        request_id = 0

        async def request(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
            method: str,
            params: dict[str, object],
        ) -> dict[str, object]:
            nonlocal request_id
            request_id += 1
            current = request_id
            writer.write((json.dumps({
                "jsonrpc": "2.0", "id": current, "method": method, "params": params,
            }) + "\n").encode())
            await writer.drain()
            while True:
                line = await reader.readline()
                assert line
                frame = json.loads(line)
                if frame.get("id") == current:
                    return frame

        reader, writer = await asyncio.open_unix_connection(endpoint)
        try:
            assert (await request(reader, writer, "initialize", {
                "protocolVersion": "2.0", "clientInfo": {"name": "fixture", "version": "1"},
            })) ["result"]
            writer.write(b'{"jsonrpc":"2.0","method":"initialized"}\n')
            await writer.drain()
            admit = await request(reader, writer, "programmatic/session/admit", {
                "session_id": session,
            })
            assert admit["result"]["session_id"] == session  # type: ignore[index]
            follow = await request(reader, writer, "session/follow", {
                "session_id": session, "subscription_id": "restart", "after_seq": -1,
            })
            assert follow["result"]["session_id"] == session  # type: ignore[index]
            send = await request(reader, writer, "programmatic/message/send", {
                "session_id": session, "message_id": "input", "text": "restart now",
            })
            assert send["result"]["message_id"] == "input"  # type: ignore[index]

            try:
                await asyncio.wait_for(drain_entered.wait(), 5)
            except TimeoutError:
                pytest.fail(f"writer={writer_state!r}; rows={log.reader(session).snapshot()!r}")
            assert not commits, "最终输出仍被真实 writer 阻塞，重启不得提交"
            drain_release.set()

            final_event = False
            while not final_event:
                line = await asyncio.wait_for(reader.readline(), 5)
                assert line
                frame = json.loads(line)
                event = frame.get("params", {}).get("event", {})
                for item in event.get("items", []):
                    body = item.get("body", {})
                    if body.get("kind") == "output" and body.get("finish") == "complete":
                        final_event = True
                        break

            page = await request(reader, writer, "message/read", {
                "session_id": session, "after_seq": -1, "limit": 50,
            })
            items = page["result"]["items"]  # type: ignore[index]
            assert items[-1]["body"]["finish"] == "complete"  # type: ignore[index]
            await asyncio.wait_for(committed.wait(), 2)
            assert len(commits) == 1
        finally:
            drain_release.set()
            writer.close()
            await writer.wait_closed()
            server.close()
            await server.wait_closed()
            await asyncio.wait_for(connection_done.wait(), 2)
            await service.shutdown()


@pytest.mark.asyncio
async def test_programmatic_restart_watcher_aborts_preclaim_after_disconnect(
    tmp_path: Path,
) -> None:
    """真实 watcher 必须消费断线异常并结束精确 pre-claim。"""
    commits: list[str] = []
    gate = RestartGate(
        boot_id="fixture-boot", supervised=True,
        commit=commits.append,
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(tmp_path, gate, channel=False) as (log, host):
        session = "programmatic:disconnect"
        frames = host._control_frames  # type: ignore[attr-defined]
        root = host.live_root
        assert root is not None
        watcher = _loaded_restart_watcher(root)
        original_wait = cast(
            Callable[..., Awaitable[None]],
            getattr(watcher, "_wait_for_request"),
        )
        watcher_entered = asyncio.Event()
        watcher_released = asyncio.Event()
        watcher_caught = asyncio.Event()
        caught_errors: list[ConnectionError] = []

        async def hold_before_wait(*args: object) -> None:
            watcher_entered.set()
            await watcher_released.wait()
            try:
                await original_wait(*args)
            except ConnectionError as error:
                caught_errors.append(error)
                watcher_caught.set()
                raise

        setattr(watcher, "_wait_for_request", hold_before_wait)
        async with _live_root(host) as root:
            generation = host.generation("programmatic")
            assert generation is not None and generation.fiber is not None
            context = generation.fiber.context
            async with context.runtime_scope():
                api = context.require(PROGRAMMATIC)
                await api.call("programmatic/session/admit", AdmitParams(session_id=session))
                await api.call(
                    "programmatic/message/send",
                    SendParams(session_id=session, message_id="input", text="restart now"),
                    _FixtureTransport("fixture-connection"),
                )
        try:
            await asyncio.wait_for(watcher_entered.wait(), 5)
            rows = log.reader(session).snapshot()
            result = next(
                row.body
                for row in rows
                if isinstance(row.body, ToolResult)
                and row.body.outcome == "success"
                and _is_restart_fixture_result(log.reader(session), row.body)
            )
            claim = frames.claim_for(session, result.call_ref)
            assert claim is not None
            error = ConnectionError("client disconnected")
            frames.fail_connection("fixture-connection", error)
            assert frames.claim_for(session, result.call_ref) is claim
            watcher_released.set()
            await asyncio.wait_for(watcher_caught.wait(), 2)
            assert caught_errors == [error]
            assert frames.claim_for(session, result.call_ref) is None
            assert gate.accepting
            assert commits == []
        finally:
            setattr(watcher, "_wait_for_request", original_wait)


def _loaded_restart_watcher(root: object) -> object:
    """从正式 Root 的 runtime.started listener 取得动态加载的 watcher。"""
    events = getattr(root, "_events")
    listeners = cast(
        Mapping[object, Iterable[object]],
        getattr(events, "_listeners"),
    )
    for key, entries in listeners.items():
        if getattr(key, "name", None) != "runtime.started":
            continue
        for entry in entries:
            callback = getattr(entry, "callback", None)
            owner = getattr(callback, "__self__", None)
            runtime = getattr(getattr(entry, "owner", None), "runtime", None)
            if (
                getattr(runtime, "plugin_id", None) == "message_push"
                and owner is not None
                and callable(getattr(owner, "_wait_for_request", None))
            ):
                return owner
    raise AssertionError("正式 Root 没有动态 message_push restart watcher")


def _is_restart_fixture_result(reader: MessageReader, result: ToolResult) -> bool:
    """确认 ToolResult 真正引用 fixture 的 agent_restart ToolCall。"""
    call_message = reader.get(result.call_ref.message_id)
    if call_message is None or not isinstance(call_message.body, Output):
        return False
    if result.call_ref.part_index >= len(call_message.body.parts):
        return False
    call = call_message.body.parts[result.call_ref.part_index]
    return isinstance(call, ToolCall) and call.arguments.get("reason") == "fixture"


@pytest.mark.asyncio
async def test_programmatic_restart_rejection_keeps_other_gate_request_and_aborts_claim(
    tmp_path: Path,
) -> None:
    """watcher 的重复 gate prepare 只清理自己的 pre-claim。"""

    class SettingsHoldingGate(RestartGate):
        def prepare(self, request_id: str) -> None:
            if request_id.startswith("restart_"):
                super().prepare("settings-request")
            super().prepare(request_id)

    commits: list[str] = []
    gate = SettingsHoldingGate(
        boot_id="fixture-boot", supervised=True,
        commit=commits.append,
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    async with _restart_application(tmp_path, gate, channel=False) as (log, host):
        session = "programmatic:settings"
        frames = host._control_frames  # type: ignore[attr-defined]
        async with _live_root(host) as root:
            generation = host.generation("programmatic")
            assert generation is not None and generation.fiber is not None
            context = generation.fiber.context
            async with context.runtime_scope():
                api = context.require(PROGRAMMATIC)
                await api.call("programmatic/session/admit", AdmitParams(session_id=session))
                await api.call(
                    "programmatic/message/send",
                    SendParams(session_id=session, message_id="input", text="restart now"),
                    _FixtureTransport("fixture-connection"),
                )

        call_ref: CallRef | None = None
        async with asyncio.timeout(5):
            async for message in log.reader(session).follow():
                rows = log.reader(session).snapshot()
                for row in rows:
                    if isinstance(row.body, ToolResult) and row.body.outcome == "success":
                        call_message = log.reader(session).get(row.body.call_ref.message_id)
                        if (
                            call_message is not None
                            and isinstance(call_message.body, Output)
                            and row.body.call_ref.part_index < len(call_message.body.parts)
                            and isinstance(
                                call_message.body.parts[row.body.call_ref.part_index], ToolCall,
                            )
                            and call_message.body.parts[row.body.call_ref.part_index].arguments.get(
                                "reason"
                            ) == "fixture"
                        ):
                            call_ref = row.body.call_ref
                            break
                if call_ref is not None:
                    break
        assert call_ref is not None
        resolved_call_ref = call_ref

        async def wait_for_claim_abort() -> None:
            while frames.claim_for(session, resolved_call_ref) is not None:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_claim_abort(), 2)
        assert not gate.accepting
        with pytest.raises(RestartRejectedError, match="已有重启请求"):
            gate.prepare("another-settings-request")
        assert commits == []
        gate.abort("settings-request")
