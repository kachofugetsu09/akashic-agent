from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.plugin_composition import CompositionOverlay, Context
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugins.generation import PluginGeneration
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot, lease_runtime_snapshot
from agent.restart import RESTART_GATE, RestartGate, RestartRejectedError
from bus.event_bus import EventBus
from agent.control.frame_book import FrameBook
from bootstrap.app_server import build_control_service
from bootstrap.tools import CoreRuntime
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.control.connection import NdjsonConnection
from plugins.message_push.restart import PendingRestart, RestartTool
from agent.plugin_contracts.tool_api import (
    CallSource,
    ContentPart,
    Denied,
    durable_call_key,
)
from plugins.tools.api import MessageReply
from plugins.content.plugin import check_text
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from plugins.programmatic.control import AdmitParams, PROGRAMMATIC, SendParams
from session.log import MessageLog, MessageReader
from session.artifact_store import ArtifactStore
from session.message import (
    CallRef, ContentReferences, Input, Message, Output, ToolCall, ToolResult, freeze_json,
)

import agent.plugins.manager as plugin_manager_module


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
    ModelRole, ToolCall,
)
from plugins.models.projection import MODEL_CALLS
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore

api_version = 3
name = "restart_provider"
version = "1.0.0"
inject = {inject}

async def apply(ctx, config):
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
        role=ModelRole.AGENT, reasoning_effort=None,
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
from agent.plugin_contracts.delivery import DELIVERY_SENDERS

api_version = 3
name = "fixture_sender"
version = "1.0.0"
inject = (DELIVERY_SENDERS,)
STATE_ROOT = {str(state_root)!r}
REJECT_FIRST = {reject_first!r}

async def apply(ctx, config):
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
    probe = root / "startup_probe"
    probe.mkdir()
    (probe / "plugin.py").write_text(
        f"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from agent.plugin_composition import RUNTIME_STARTING
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_WRITERS
from plugins.delivery.api import FINAL_OUTPUT_DELIVERY
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS
from session.message import CallRef, Input, Output, ToolCall, ToolResult

api_version = 3
name = "startup_probe"
version = "1.0.0"
inject = (MESSAGE_WRITERS, BINDINGS, TOOLS, ALL_TOOLS, FINAL_OUTPUT_DELIVERY)
RUN_ID = {run_id!r}
STATE_ROOT = {str(state_root)!r}


class Waiter:
    async def wait(self, reader, turn):
        Path(STATE_ROOT).mkdir(parents=True, exist_ok=True)
        Path(STATE_ROOT, "delivered-" + reader.session_id.replace(":", "_")).write_text("1")
        return None


async def apply(ctx, config):
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

    def append_after_prepare(_event):
        binding = tools.bind(ctx.require(ALL_TOOLS)().select("agent_restart"), bindings)
        session = "startup-probe:" + RUN_ID
        inputs = writers.bind(
            ctx, author="user", source="startup-probe", body_types=(Input,), content={{}},
        )(session)
        outputs = writers.bind(
            ctx, author="assistant", source="startup-probe", body_types=(Output,),
            content={{}}, check_call=lambda call: None,
        )(session)
        inputs.append("startup-input-" + RUN_ID, Input(()))
        call = outputs.append(
            "startup-call-" + RUN_ID,
            Output((ToolCall(binding, {{"reason": "startup"}}),), "continue"),
        )
        results = writers.bind(
            ctx, author="tool", source="startup-probe", body_types=(ToolResult,), content={{}},
        )(session, call_ref=CallRef(call.message_id, 0))
        def append_result():
            results.append(
                "startup-result-" + RUN_ID,
                ToolResult(CallRef(call.message_id, 0), "success", ()),
            )
            outputs.append("startup-final-" + RUN_ID, Output((), "complete"))

        asyncio.get_running_loop().call_soon(append_result)

    await ctx.on(RUNTIME_STARTING, append_after_prepare)
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
        "context",
        "tools",
        "conversation",
        "react",
        "turn_projection",
        "reply",
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
    with pytest.raises(ValueError, match="只能包含 reason"):
        await tool.prepare({"reason": "reload", "extra": True}, source)
    prepared = await tool.prepare({"reason": " reload "}, source)
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
    async with _restart_application(tmp_path, gate, channel=False) as (_log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            names = {
                ref.name
                for ref in snapshot.composition_root.context.require(ALL_TOOLS)().refs
            }
    assert "agent_restart" not in names


@pytest.mark.asyncio
async def test_starting_baseline_ignores_old_result_and_reads_result_after_prepare(
    tmp_path: Path,
) -> None:
    """启动期间追加的真实 ToolResult 不能被异步基线吞掉。"""
    first_commits: list[str] = []
    first_committed = asyncio.Event()
    first_gate = RestartGate(
        boot_id="first-boot", supervised=True,
        commit=_commit_recorder(first_commits, first_committed),
    )
    async with _restart_application(
        tmp_path, first_gate, channel=False, source_tag="baseline-first", startup_run="first",
    ):
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
    ):
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

    import plugins.conversation.program as conversation_program

    monkeypatch.setattr(conversation_program, "shell_cleanup", controlled_cleanup)
    async with _restart_application(tmp_path, gate, channel=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            accept = context.require(CHANNEL_INPUT)
            await accept(
                "test:room", "input-1",
                ChannelInboundMessage(
                    "test", "user", "room", "restart now", datetime.now(timezone.utc), {},
                ),
            )
            await asyncio.wait_for(cleanup_blocked.wait(), 2)
            rows = log.reader("test:room").snapshot()
            assert any(
                isinstance(row.body, Output) and row.body.finish == "complete"
                for row in rows
            ), rows
            assert not commits
            assert not gate.accepting

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
    async with _restart_application(
        tmp_path, gate, channel=True, reject_first=True,
    ) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            accept = snapshot.composition_root.context.require(CHANNEL_INPUT)
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

        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            accept = snapshot.composition_root.context.require(CHANNEL_INPUT)
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
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                ctx = snapshot.composition_root.context
                async with ctx.runtime_scope():
                    binding = ctx.require(TOOLS).bind(
                        ctx.require(ALL_TOOLS)().select("agent_restart"),
                        ctx.require(BINDINGS),
                    )
                    descriptor = log.read_binding(binding)
                    generations = snapshot.generations
            metadata = descriptor["metadata"]
            assert isinstance(metadata, Mapping)
            tool_descriptor = metadata["tool"]
            assert isinstance(tool_descriptor, Mapping)
            assert tool_descriptor["name"] == "agent_restart"
            assert isinstance(tool_descriptor["owner"], str)
            assert tool_descriptor["owner"] in generations
            call = output_writer.append(
                "late-call", Output((ToolCall(binding, {"reason": "reload"}),), "continue"),
            )
            result_writer = log.writer(
                session, author="tool", source="reload-probe", body_types=(ToolResult,), content={},
                call_ref=CallRef(call.message_id, 0),
            )
            final_output = Output((), "complete")
            await host.terminate_all()
            state = tmp_path / "reload-state"
            (state / "await-prepare").parent.mkdir(parents=True, exist_ok=True)
            (state / "await-prepare").write_text("1")

            async def append_late_result() -> None:
                await asyncio.wait_for(_wait_for_path(state / "prepared"), 2)
                result_writer.append(
                    "late-result", ToolResult(CallRef(call.message_id, 0), "success", ()),
                )
                output_writer.append("late-final", final_output)

            late_task = asyncio.create_task(append_late_result())
            async with _restart_application(
                tmp_path, gate, channel=False, source_tag="reload-second", reload_probe=True,
                message_log=log,
            ):
                assert any(
                    task.get_name() == "plugin-task:agent-restart-watcher"
                    for task in asyncio.all_tasks()
                )
                await asyncio.wait_for(late_task, 2)
                assert (state / "prepared").read_text()
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


async def _wait_for_path(path: Path) -> None:
    while not path.exists():
        await asyncio.sleep(0.01)


async def _wait_for_admission_pause(snapshot: RuntimeSnapshot) -> None:
    while snapshot.accepting_leases:
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("supervised", [True, False], ids=["supervised", "unmanaged"])
async def test_restart_provider_candidate_preserves_formal_root_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervised: bool,
) -> None:
    """候选和正式 Root 按宿主 gate 保持相同的 restart 声明。"""
    sources = tmp_path / "plugins"
    _copy_plugin_sources(
        sources,
        (
            "sources",
            "content",
            "context",
            "tools",
            "conversation",
            "react",
            "turn_projection",
            "reply",
            "tool_search",
            "delivery",
            "programmatic",
            "message_push",
        ),
    )

    # 1. 只替换一个已安装 provider generation。
    generated = tmp_path / "generated"
    generated.mkdir()
    _write_restart_provider(generated, shared_event=True)
    provider_repo = tmp_path / "restart-provider"
    provider_repo.mkdir()
    provider_source = generated / "restart_provider" / "plugin.py"
    shutil.copy2(provider_source, provider_repo / "plugin.py")
    (provider_repo / "akashic.plugin.toml").write_text(
        "schema_version=1\n"
        "name='restart_provider'\n"
        "version='1.0.0'\n"
        "api_version=3\n"
        "entrypoint='plugin.py'\n",
    )
    for args in (
        ("git", "init", "-q"),
        ("git", "add", "."),
        ("git", "-c", "user.email=fixture@example.invalid", "-c", "user.name=fixture", "commit", "-qm", "stable"),
    ):
        subprocess.run(args, cwd=provider_repo, check=True)
    install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(provider_repo),
        marketplace="fixture",
        ref_name="HEAD",
        sparse_paths=[],
        plugins_home=tmp_path / "home",
    )

    commits: list[str] = []
    gate = RestartGate(
        boot_id="fixture-boot", supervised=supervised,
        commit=commits.append if supervised else None,
    )
    log = MessageLog(tmp_path / "sessions.db")
    artifact_store = ArtifactStore(tmp_path / "sessions.db")
    host = PluginManager(
        [sources],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
        message_log=log,
        restart_gate=gate,
        channel_attachment_store=ChannelAttachmentArtifactStore(
            workspace=tmp_path / "workspace", metadata_store=artifact_store
        ),
    )
    observed: dict[str, RuntimeSnapshot] = {}
    original_check = plugin_manager_module._validate_candidate_formal_snapshot_identity

    def capture_identity(
        generation: PluginGeneration, *, candidate: RuntimeSnapshot, formal: RuntimeSnapshot,
    ) -> None:
        observed["candidate"] = candidate
        observed["formal"] = formal
        original_check(generation, candidate=candidate, formal=formal)

    monkeypatch.setattr(
        plugin_manager_module,
        "_validate_candidate_formal_snapshot_identity",
        capture_identity,
    )
    old_task: asyncio.Task[object] | None = None
    promotion_task: asyncio.Task[dict[str, object]] | None = None
    release_authorize: asyncio.Event | None = None
    watcher_history: list[asyncio.Task[None]] = []
    watcher_started: asyncio.Queue[asyncio.Task[None]] = asyncio.Queue()
    spawn = Context.spawn

    async def observe_spawn(
        context: Context, coroutine: Coroutine[object, object, object], *, name: str,
    ) -> asyncio.Task[object]:
        """在真实任务创建完成时记录，避免忙轮询阻碍清理线程。"""
        task = await spawn(context, coroutine, name=name)
        if name == "agent-restart-watcher":
            watcher = cast(asyncio.Task[None], task)
            watcher_history.append(watcher)
            watcher_started.put_nowait(watcher)
        return task

    monkeypatch.setattr(Context, "spawn", observe_spawn)
    try:
        await host.load_all()
        runtime_runner = asyncio.create_task(host.run_runtime_services())
        old_watcher = await asyncio.wait_for(watcher_started.get(), 2) if supervised else None
        stable = host.current_snapshot
        assert stable is not None
        stable_generation_ids = {
            plugin_id: generation.generation_id
            for plugin_id, generation in stable.generations.items()
        }

        # 2. 通过真实 install_candidate 准备更新。
        (provider_repo / "plugin.py").write_text(
            (provider_repo / "plugin.py").read_text() + "\n# candidate revision\n",
        )
        for args in (
            ("git", "add", "."),
            ("git", "-c", "user.email=fixture@example.invalid", "-c", "user.name=fixture", "commit", "-qm", "candidate"),
        ):
            subprocess.run(args, cwd=provider_repo, check=True)
        installed, status = await host.install_candidate(
            source=str(provider_repo),
            marketplace="fixture",
            ref_name="HEAD",
            sparse_paths=[],
        )
        assert installed.plugin_name == "restart_provider"
        assert status["candidate_plugin_id"] == "restart_provider@fixture"
        latest = host.latest_snapshot
        assert latest is not None
        changed = [
            plugin_id
            for plugin_id, generation in latest.generations.items()
            if stable.generations[plugin_id].generation_id != generation.generation_id
        ]
        assert changed == ["restart_provider@fixture"]
        assert (
            latest.generations["message_push"].generation_id
            == stable_generation_ids["message_push"]
        )
        assert latest.composition_root is not None
        candidate_overlay = latest.composition_root
        assert isinstance(candidate_overlay, CompositionOverlay)
        expected_replaced = {"reply", "restart_provider@fixture"}
        if supervised:
            expected_replaced.add("message_push")
        assert expected_replaced <= candidate_overlay.replaced_plugin_ids
        assert expected_replaced <= candidate_overlay.candidate.active_plugin_ids()
        candidate_gate = candidate_overlay.context.require(RESTART_GATE)
        assert isinstance(candidate_gate, RestartGate)
        assert candidate_gate.supervised is supervised
        assert candidate_gate.execution_enabled is False
        expected_gate_error = "不允许重启效果" if supervised else "未由 supervisor 托管"
        with pytest.raises(RestartRejectedError, match=expected_gate_error):
            candidate_gate.prepare("candidate-request")
        with pytest.raises(RestartRejectedError, match="不允许重启效果"):
            await candidate_gate.commit("candidate-request")
        candidate_tool_names = {
            ref.name for ref in candidate_overlay.context.require(ALL_TOOLS)().refs
        }
        if supervised:
            assert "agent_restart" in candidate_tool_names
        else:
            assert "agent_restart" not in candidate_tool_names

        release_authorize = asyncio.Event()
        if supervised:
            # 3. 真实 ToolExecution 在授权处阻塞，证明 promotion 必须等待旧 lease。
            session = "reload-probe:real"
            input_writer = log.writer(
                session, author="user", source="reload-probe", body_types=(Input,), content={},
            )
            output_writer = log.writer(
                session, author="assistant", source="reload-probe", body_types=(Output,),
                content={}, check_call=lambda call: None,
            )
            input_writer.append("real-input", Input(()))
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                context = snapshot.composition_root.context
                async with context.runtime_scope():
                    binding = context.require(TOOLS).bind(
                        context.require(ALL_TOOLS)().select("agent_restart"),
                        context.require(BINDINGS),
                    )
            call = output_writer.append(
                "real-call", Output((ToolCall(binding, {"reason": "reload"}),), "continue"),
            )
            old_result_writer = log.writer(
                session, author="tool", source="reload-probe", body_types=(ToolResult,),
                content={"text": lambda _part: ContentReferences()},
                call_ref=CallRef(call.message_id, 0),
            )
            entered_authorize = asyncio.Event()

            async def authorize(_binding_id: str, _arguments: Mapping[str, object]) -> Mapping[str, object]:
                entered_authorize.set()
                await release_authorize.wait()
                raise Denied("old runtime drained before promotion")

            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                context = snapshot.composition_root.context
                async with context.runtime_scope():
                    execution = context.require(TOOLS).execution(authorize)
                    old_task = asyncio.create_task(
                        execution.execute_call(
                            MessageReply(
                                "real-old-result", CallRef(call.message_id, 0),
                                log.reader(session), old_result_writer, lambda: None,
                            )
                        ),
                        name="reload-probe:old-tool-execution",
                    )
                    await asyncio.wait_for(entered_authorize.wait(), 2)

        # 5. promotion 完成 candidate Overlay 到 formal Root 的切换。
        promotion_task = asyncio.create_task(
            host.switch_ready("restart_provider@fixture", update_id=installed.update_id),
        )
        if supervised:
            await asyncio.wait_for(_wait_for_admission_pause(stable), 2)
            assert not promotion_task.done()
            assert old_task is not None and not old_task.done()
            assert commits == []
            release_authorize.set()
            old_result = await asyncio.wait_for(old_task, 2)
            assert old_result.outcome == "denied"
            persisted_result = log.reader(session).get("real-old-result")
            assert persisted_result is not None
            assert persisted_result.body == ToolResult(
                CallRef(call.message_id, 0), "denied", old_result.parts,
            )
        promoted = await asyncio.wait_for(promotion_task, 5)
        assert promoted["publication_state"] == "promoted"
        candidate = observed["candidate"]
        formal = observed["formal"]
        assert candidate.snapshot_id == formal.snapshot_id

        candidate_topology = candidate.composition_topology
        formal_topology = formal.composition_topology
        assert candidate_topology is not None and formal_topology is not None
        assert candidate_topology.services == formal_topology.services
        assert candidate_topology.fibers == formal_topology.fibers
        assert candidate.composition_root is not None
        assert formal.composition_root is not None
        assert (
            candidate.composition_root.plugin_service_owners()
            == formal.composition_root.plugin_service_owners()
        )
        expected_listeners = {
            "serial:runtime.started:tools",
            "serial:runtime.stopping:tools",
            "emit:runtime.starting:reply",
            "emit:runtime.starting:restart_provider@fixture",
            "serial:runtime.started:reply",
            "serial:runtime.stopping:reply",
            "serial:runtime.started:programmatic",
            "serial:runtime.stopping:programmatic",
        }
        if supervised:
            expected_listeners |= {
                "emit:runtime.starting:restart",
                "serial:runtime.started:restart",
                "serial:runtime.stopping:restart",
            }
        assert set(candidate_topology.listeners) == expected_listeners
        assert formal_topology.listeners == candidate_topology.listeners
        assert commits == []
        assert gate.permit_count == 0
        if supervised:
            await asyncio.wait_for(watcher_started.get(), 2)
            new_watchers = [task for task in watcher_history if task is not old_watcher]
            assert new_watchers
            assert new_watchers[-1] is not old_watcher
            assert old_watcher is not None and old_watcher.done()

        candidate_tool = RestartTool(candidate_gate, FrameBook())
        with pytest.raises(RestartRejectedError, match="正式 supervisor"):
            await candidate_tool.prepare({"reason": "candidate"}, _source())
    finally:
        if release_authorize is not None:
            release_authorize.set()
        if promotion_task is not None and not promotion_task.done():
            promotion_task.cancel()
        if old_task is not None and not old_task.done():
            old_task.cancel()
        if promotion_task is not None:
            await asyncio.gather(promotion_task, return_exceptions=True)
        if old_task is not None:
            await asyncio.gather(old_task, return_exceptions=True)
        if "runtime_runner" in locals():
            runtime_runner.cancel()
            await asyncio.gather(runtime_runner, return_exceptions=True)
        await host.terminate_all()
        artifact_store.close()
        log.close()


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
        endpoint = tmp_path / "control.sock"
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

        server = await asyncio.start_unix_server(accept, path=str(endpoint))
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

        reader, writer = await asyncio.open_unix_connection(str(endpoint))
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

            async with asyncio.timeout(5):
                await drain_entered.wait()
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
    async with _restart_application(tmp_path, gate, channel=False) as (log, host):
        session = "programmatic:disconnect"
        frames = host._control_frames  # type: ignore[attr-defined]
        snapshot = host.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        watcher = _loaded_restart_watcher(snapshot.composition_root)
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
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
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
    async with _restart_application(tmp_path, gate, channel=False) as (log, host):
        session = "programmatic:settings"
        frames = host._control_frames  # type: ignore[attr-defined]
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
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
