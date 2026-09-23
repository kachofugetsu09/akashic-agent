import ast
import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.config_input import save_config
from agent.plugin_composition.tasks import RestartGate, Task, Tasks
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.log import MessageLog, MessageWriter, OwnerTransaction
from session.artifact_store import ArtifactStore
from plugins.content.plugin import check_text
from plugins.conversation.plugin import check_origin
from plugins.sources.plugin import SOURCES
from plugins.tools.api import MessageReply
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import ALL_TOOLS, TOOLS, open_tool
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


@dataclass
class ModelControl:
    entered: asyncio.Queue = field(default_factory=asyncio.Queue)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    main_calls: int = 0
    main_entered: asyncio.Queue = field(default_factory=asyncio.Queue)
    main_release: asyncio.Event = field(default_factory=asyncio.Event)
    main_tool: bool = False
    send_failure: str | None = None
    sent: asyncio.Queue = field(default_factory=asyncio.Queue)
    restart_gate: RestartGate | None = None
    conversation_context: object | None = None
    conversation_owner: tuple[object, object] | None = None
    reply_context: object | None = None
    reply_owner: tuple[object, object] | None = None
    report_owner_checks: int = 0


CONTROLS: dict[str, ModelControl] = {}


def _write_python_source(path: Path, source: str) -> None:
    """Validate dynamic fixture code before it reaches the plugin loader."""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source)


def _provider_context(host: PluginManager, plugin_id: str):
    generation = host.generation(plugin_id)
    assert generation is not None and generation.fiber is not None
    return generation.fiber.context


def _registered_source(host: PluginManager, name: str):
    context = _provider_context(host, "sources")
    with context.runtime_scope():
        matches = tuple(source for source in context.require(SOURCES).entries()
                        if source.name == name)
    assert len(matches) == 1
    return matches[0]


async def _capture_report_owners(host: PluginManager, control: ModelControl) -> None:
    from plugins.conversation.plugin import CONVERSATION_COMPLETE
    from plugins.reply.api import REPLY_PROGRAM

    conversation = _provider_context(host, "conversation")
    reply = _provider_context(host, "reply")
    async with conversation.runtime_scope():
        conversation_owner = (
            CONVERSATION_COMPLETE, conversation.require(CONVERSATION_COMPLETE),
        )
    async with reply.runtime_scope():
        reply_owner = (REPLY_PROGRAM, reply.require(REPLY_PROGRAM))
    control.conversation_context = conversation
    control.conversation_owner = conversation_owner
    control.reply_context = reply
    control.reply_owner = reply_owner


async def _bind_tool(host: PluginManager, name: str) -> str:
    tools_context = _provider_context(host, "tools")
    caller_context = _provider_context(host, "subagent")
    async with tools_context.runtime_scope():
        async with caller_context.runtime_scope():
            return caller_context.require(TOOLS).bind(
                caller_context.require(ALL_TOOLS)().select(name),
                caller_context.require(BINDINGS),
            )


def text_part(part: ContentPart | ToolCall) -> str:
    """Read a fixture text part after checking the persisted part shape."""
    assert isinstance(part, ContentPart)
    assert isinstance(part.value, str)
    return part.value


def mapping_part(part: ContentPart | ToolCall) -> Mapping[str, object]:
    """Read a fixture object part after checking its JSON shape."""
    assert isinstance(part, ContentPart)
    assert isinstance(part.value, Mapping)
    return part.value


@asynccontextmanager
async def application(
    tmp_path, *, background=False, block=False, block_main=False, main_tool=False,
    legacy_consumer=False,
):
    sources = tmp_path / "plugins"
    sources.mkdir()
    for name in ("tools", "content", "context", "assets", "standard_tools", "turn_projection", "sources"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in (
        "conversation",
        "commands",
        "react",
        "subagent",
        "reply",
        "reply_program",
        "tool_search",
        "delivery",
        "delivery_policy",
    ):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    if legacy_consumer:
        legacy = sources / "legacy_consumer"
        legacy.mkdir()
        _write_python_source(legacy / "plugin.py", '''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "legacy_consumer"
version = "1.0.0"
inject = (ServiceKey("conversation.v1"),)
async def apply(ctx):
    await ctx.provide(ServiceKey("legacy.consumer.loaded"), True)
''')
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log = MessageLog(workspace / "sessions.db")
    store = ArtifactStore(workspace / "sessions.db")
    context_config = workspace / "plugin-data/context-builtin"
    context_config.parent.mkdir(parents=True, exist_ok=True)
    save_config(context_config, {"prompt_sources": {"skills": "standard_tools"}})
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=store)
    initialize_plugin_workspace(workspace)
    provider = sources / "models_fixture"
    provider.mkdir()
    provider_source = '''
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from agent.plugin_composition import CHAT_MODELS
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.delivery.api import Receipt
import json
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ToolCall
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
api_version = 3
name = "models_fixture"
inject = (DELIVERY_SENDERS,)
version = "1.0.0"
async def apply(ctx):
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            if any(item.get("role") == "tool" for item in request.messages):
                return LLMResponse("child finished")
            return LLMResponse(None, [ToolCall("child-write", "write_file", {"path": "answer.txt", "content": "once"})])
    descriptor = BoundModelDescriptor(
        binding_id="fixture-model", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture", driver_contract_version="1",
        auth_identity="fixture", model="fixture", role="agent", reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=10000), capability_sources=CapabilitySources(), capability_digest="fixture")
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
    class Sender:
        idempotent = True
        async def send(self, key, address, message):
            control.sent.put_nowait((key, address, message))
            if control.send_failure == "raise":
                raise TimeoutError("sender connection lost")
            if control.send_failure == "failed":
                return Receipt(status="failed", error="sender connection lost")
            return Receipt(status="delivered", provider_ids=(key,))
        async def query(self, key, address):
            return None
    @asynccontextmanager
    async def sender():
        yield Sender()
    control = CONTROLS[CONTROL_PATH]
    await ctx.require(DELIVERY_SENDERS).register(ctx, name="test", idempotent=True, open=sender)
'''
    control = ModelControl(main_tool=main_tool)
    if not block_main:
        control.main_release.set()
    if not block:
        control.release.set()
    CONTROLS[str(tmp_path)] = control
    module = provider / "plugin.py"
    text = provider_source.replace("async def apply(ctx):", "from " + __name__ + " import CONTROLS\nasync def apply(ctx):")
    text = text.replace("CONTROL_PATH", repr(str(tmp_path)))
    text = text.replace("        async def complete(self, request):", "        async def complete(self, request):\n            control = CONTROLS[" + repr(str(tmp_path)) + "]\n            if '## 后台任务结果' in str(request.messages):\n                control.main_calls += 1\n                control.main_entered.put_nowait(request)\n                assert control.conversation_context is not None and control.conversation_owner is not None\n                control.conversation_context.require_runtime_owner(*control.conversation_owner)\n                assert control.reply_context is not None and control.reply_owner is not None\n                control.reply_context.require_runtime_owner(*control.reply_owner)\n                control.report_owner_checks += 1\n                await control.main_release.wait()\n                if control.main_tool and 'main-report.txt' not in str(request.messages[:-1]):\n                    return LLMResponse(None, [ToolCall('main-write', 'write_file', {'path': CONTROL_REPORT_PATH, 'content': 'main result'})])\n                summary = ('new provider result' if 'new provider result' in str(request.messages[-1])\n                           else 'cancelled' if 'cancelled' in str(request.messages[-1])\n                           else 'child finished')\n                return LLMResponse('main summary: ' + summary)\n            if '[human followup]' in str(request.messages):\n                return LLMResponse('human answer')\n            control.calls += 1\n            control.entered.put_nowait(request)\n            await control.release.wait()")
    text = text.replace("CONTROL_REPORT_PATH", repr(str(tmp_path / "workspace/main-report.txt")))
    _write_python_source(module, text)
    event_bus = EventBus()
    restart_gate = RestartGate(
        boot_id=f"subagent-test-{tmp_path.name}", supervised=True,
        commit=lambda _request_id: None,
    )
    restart_gate.prepare("hold-default-reply")
    control.restart_gate = restart_gate
    host = PluginManager(
        [sources], event_bus=event_bus, workspace=workspace,
        installed_cache_root=tmp_path / "cache", message_log=log,
        channel_attachment_store=artifacts, restart_gate=restart_gate,
    )
    tasks = Tasks()
    body_error: BaseException | None = None
    try:
        await host.load_all()
        await _capture_report_owners(host, control)
        binding = await _bind_tool(host, "spawn")
        subagent_context = _provider_context(host, "subagent")
        async with subagent_context.runtime_scope():
            bindings = subagent_context.require(BINDINGS)
        reader = log.reader("test:parent")
        inputs = log.writer(reader.session_id, author="user", source="fixture", body_types=(Input,),
                            content={"text": check_text, "channel.origin": check_origin})
        inputs.append("parent-input", Input((ContentPart("text", "complete an independent file task"),
            ContentPart("channel.origin", {"channel": "test", "chat_id": "parent", "sender": "user"}))))
        output = log.writer(reader.session_id, author="assistant", source="fixture", body_types=(Output,),
                            content={"text": check_text}, check_call=lambda call: None)
        output.append("parent-call", Output((ToolCall(binding,
            {"task": "Write a report in your task directory", "profile": "scripting", "run_in_background": background}),), "continue"))
        ref = CallRef("parent-call", 0)
        result_writer = log.writer(reader.session_id, author="tool", source="fixture", body_types=(ToolResult,),
                                  content={"text": check_text}, call_ref=ref)
        reply = MessageReply("parent-result", ref, reader, result_writer, lambda: None)
        async def authorize(binding, arguments):
            return {"allowed": True}
        execution = ToolExecution(
            log.owner("plugin:tools"), tasks, partial(open_tool, bindings), authorize,
            task_key="effects",
        )
        yield host, log, execution, reply
    except BaseException as error:
        body_error = error
        raise
    finally:
        CONTROLS.pop(str(tmp_path), None)
        cleanup_errors: list[BaseException] = []
        for cleanup in (tasks.close, host.terminate_all, event_bus.aclose):
            try:
                await cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
        for cleanup in (log.close, store.close):
            try:
                cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if body_error is None:
                raise BaseExceptionGroup("subagent fixture cleanup failed", cleanup_errors)
            raise BaseExceptionGroup(
                "subagent fixture body and cleanup failed", [body_error, *cleanup_errors],
            ) from None


@pytest.mark.asyncio
async def test_old_factory_consumer_stays_pending_without_affecting_live_peers(tmp_path):
    from agent.plugin_composition import CompositionError, FiberState, ServiceKey
    from plugins.conversation.plugin import CONVERSATION_COMPLETE

    async with application(tmp_path, legacy_consumer=True) as (host, _log, _execution, _reply):
        legacy = host.generation("legacy_consumer")
        assert legacy is not None and legacy.fiber is not None
        assert legacy.fiber.state is FiberState.PENDING
        assert legacy.fiber.missing_services == ("conversation.v1",)
        conversation = _provider_context(host, "conversation")
        async with conversation.runtime_scope():
            complete = conversation.require(CONVERSATION_COMPLETE)
            with pytest.raises(CompositionError) as caught:
                conversation.require(ServiceKey("conversation.v1"))
            assert caught.value.code == "INACTIVE_SERVICE"
            assert "conversation.v1" in str(caught.value)
        assert callable(complete)
        tools = _provider_context(host, "tools")
        async with tools.runtime_scope():
            catalog = tools.require(TOOLS)
        assert host.generation("tools") is not None
        assert host.generation("tools").state == "active"
        async with tools.runtime_scope():
            assert tools.require(TOOLS) is catalog


@pytest.mark.asyncio
async def test_report_drains_with_conversation_and_rejects_new_calls(tmp_path):
    from agent.plugin_composition import CompositionError
    from plugins.conversation.plugin import CONVERSATION_COMPLETE
    from session.log import SessionAttributes

    async with application(tmp_path) as (host, log, _execution, _reply):
        session_id = "test:drain"
        log.ensure_session(session_id, SessionAttributes("internal", "excluded"))
        reader = log.reader(session_id)
        inputs = log.writer(
            session_id, author="user", source="conversation", body_types=(Input,),
            content={"text": check_text, "channel.origin": check_origin},
        )
        inputs.append("drain-input", Input((
            ContentPart("text", "completed before the report"),
            ContentPart("channel.origin", {
                "channel": "test", "chat_id": "drain", "sender": "user",
            }),
        )))
        outputs = log.writer(
            session_id, author="assistant", source="conversation", body_types=(Output,),
            content={"text": check_text},
        )
        prior_output = outputs.append(
            "drain-output", Output((ContentPart("text", "already complete"),), "complete"),
        )

        entered = asyncio.Event()
        release = asyncio.Event()
        child_tasks: list[Task] = []
        update_task = None
        report_task = None
        admission_wait = None
        captured = None
        update_result_consumed = False
        wait_result_consumed = False
        report_result_consumed = False
        try:
            conversation = _provider_context(host, "conversation")
            subagent = _provider_context(host, "subagent")
            async with conversation.runtime_scope():
                complete = conversation.require(CONVERSATION_COMPLETE)
                captured = conversation.capture_runtime_scope()
            admission_wait = asyncio.create_task(captured.wait_admission_closed())

            async def report(task: Task, current):
                child_tasks.append(task)
                entered.set()
                await release.wait()
                return current.snapshot()[-1]

            async def run_report():
                async with subagent.runtime_scope():
                    return await subagent.require(CONVERSATION_COMPLETE)(session_id, report)

            report_task = asyncio.create_task(run_report())
            await asyncio.wait_for(entered.wait(), 10)

            tools_generation = host.generation("tools")
            assert tools_generation is not None and tools_generation.fiber is not None
            tools = _provider_context(host, "tools")
            async with tools.runtime_scope():
                catalog = tools.require(TOOLS)

            conversation_source = tmp_path / "plugins/conversation/plugin.py"
            _write_python_source(
                conversation_source,
                conversation_source.read_text() + "\n# local owner drain probe\n",
            )
            update_task = asyncio.create_task(host.reconcile_changed())

            done, _ = await asyncio.wait({admission_wait}, timeout=10)
            assert admission_wait in done, "Conversation provider 没有关闭新调用接纳"
            wait_result_consumed = True
            _ = admission_wait.result()
            await captured.close()
            assert not update_task.done(), "局部更新必须等待已接纳的报告 Task"

            async def forbidden_report(_task, _reader):
                raise AssertionError("drain 后的新 Conversation 调用不得执行 program")

            with pytest.raises(CompositionError) as caught:
                await complete(session_id, forbidden_report)
            assert caught.value.code == "OWNER_UNAVAILABLE"

            assert host.generation("tools") is tools_generation
            assert tools_generation.state == "active"
            async with tools.runtime_scope():
                assert tools.require(TOOLS) is catalog

            release.set()
            done, _ = await asyncio.wait({report_task}, timeout=10)
            assert report_task in done, "已接纳的 Conversation Task 没有物理结算"
            report_result_consumed = True
            assert report_task.result() == prior_output
            assert child_tasks and child_tasks[0].done

            done, _ = await asyncio.wait({update_task}, timeout=15)
            assert update_task in done, "局部更新未在原报告 Task 物理结算后完成"
            update_result_consumed = True
            changes = update_task.result()
            changed = {item["plugin_id"]: item for item in changes}
            assert changed["conversation"]["publication_state"] == "active"
            assert host.generation("conversation") is not None
            assert host.generation("conversation").fiber is not conversation.fiber
            assert host.generation("tools") is tools_generation
        finally:
            release.set()
            if captured is not None:
                await captured.close()
            if admission_wait is not None and not admission_wait.done():
                admission_wait.cancel()
                try:
                    await admission_wait
                except asyncio.CancelledError:
                    wait_result_consumed = True
            elif admission_wait is not None and not wait_result_consumed:
                _ = admission_wait.result()
            if update_task is not None and not update_task.done():
                done, _ = await asyncio.wait({update_task}, timeout=10)
                if update_task not in done:
                    update_result_consumed = True
                    _ = await update_task
            if update_task is not None and not update_result_consumed:
                update_result_consumed = True
                _ = update_task.result()
            if report_task is not None and not report_task.done():
                done, _ = await asyncio.wait({report_task}, timeout=10)
                if report_task not in done:
                    report_result_consumed = True
                    _ = await report_task
            if report_task is not None and not report_result_consumed:
                report_result_consumed = True
                _ = report_task.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_trace", [False, True])
async def test_sync_spawn_persists_internal_flow_and_replays_original_result(tmp_path, broken_trace, monkeypatch):
    async with application(tmp_path) as (_host, log, execution, reply):
        if broken_trace:
            trace = tmp_path / "workspace/memory/spawn_trace.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("existing diagnostic\n")
            original_open = Path.open
            def open_file(path, mode="r", *args, **kwargs):
                if path == trace and mode == "a":
                    raise PermissionError("diagnostic file is read-only")
                return original_open(path, mode, *args, **kwargs)
            monkeypatch.setattr(Path, "open", open_file)
        result = await asyncio.wait_for(execution.execute_call(reply), 15)
        assert result.outcome == "success" and "child finished" in text_part(result.parts[0])
        sessions = [key for key in log.catalog().snapshot_heads() if key.startswith("subagent:")]
        assert len(sessions) == 1
        reader = log.reader(sessions[0])
        assert reader.attributes.visibility == "internal" and reader.attributes.learning == "excluded"
        rows = reader.snapshot()
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
        request = next(mapping_part(part) for part in rows[0].body.parts if isinstance(part, ContentPart) and part.kind == "subagent.request")
        path = tmp_path / "workspace/subagent-runs" / request["job_id"] / "answer.txt"
        assert path.read_text() == "once"
        stamp = path.stat().st_mtime_ns
        repeated = await execution.execute_call(reply)
        assert (repeated.outcome, repeated.parts) == (result.outcome, result.parts)
        assert reader.snapshot() == rows and path.stat().st_mtime_ns == stamp
        assert len([row for row in log.reader("test:parent").snapshot() if isinstance(row.body, Input)]) == 1
        if broken_trace:
            assert trace.read_text() == "existing diagnostic\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("send_failure", [None, "failed", "raise"])
async def test_background_spawn_returns_receipt_and_returns_result_once(tmp_path, monkeypatch, send_failure):
    closed = asyncio.Event()
    save = OwnerTransaction.save
    def observe(self, key, value, **kwargs):
        result = save(self, key, value, **kwargs)
        if value.get("settled") is True:
            closed.set()
        return result
    monkeypatch.setattr(OwnerTransaction, "save", observe)
    async with application(tmp_path, background=True) as (host, log, execution, reply):
        CONTROLS[str(tmp_path)].send_failure = send_failure
        result = await asyncio.wait_for(execution.execute_call(reply), 15)
        assert result.outcome == "success" and "已创建后台任务" in text_part(result.parts[0])
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("test:parent").snapshot()
                outputs = [row for row in rows if row.source.startswith("subagent:")
                           and isinstance(row.body, Output) and row.body.finish == "complete"]
                if outputs:
                    return outputs[-1]
        message = await asyncio.wait_for(completed(), 15)
        assert "child finished" in text_part(message.body.parts[0])
        assert "main summary" in text_part(message.body.parts[0])
        _, address, sent = await asyncio.wait_for(CONTROLS[str(tmp_path)].sent.get(), 10)
        assert address == "parent" and sent == message
        await asyncio.wait_for(closed.wait(), 10)
        assert all(record.value["settled"] for _, record in log.owner("plugin:subagent").list())
        assert CONTROLS[str(tmp_path)].main_calls == 1
        assert CONTROLS[str(tmp_path)].report_owner_checks == 1
        repeated = await execution.execute_call(reply)
        assert (repeated.outcome, repeated.parts) == (result.outcome, result.parts)
        assert len([row for row in log.reader("test:parent").snapshot() if isinstance(row.body, Input)]) == 1


def additional_call(log, original, number):
    reader = log.reader("test:parent")
    output = log.writer(reader.session_id, author="assistant", source="fixture", body_types=(Output,),
                        content={"text": check_text}, check_call=lambda call: None)
    identity = "parent-call-" + str(number)
    output.append(identity, Output((original.request(),), "continue"))
    ref = CallRef(identity, 0)
    writer = log.writer(reader.session_id, author="tool", source="fixture", body_types=(ToolResult,),
                        content={"text": check_text}, call_ref=ref)
    return MessageReply(identity + ":result", ref, reader, writer, lambda: None)


@pytest.mark.asyncio
async def test_capacity_and_cancel_hold_until_original_child_is_drained(tmp_path):
    async with application(tmp_path, background=True, block=True) as (host, log, execution, reply):
        control = CONTROLS[str(tmp_path)]
        for index in range(3):
            call = reply if index == 0 else additional_call(log, reply, index)
            result = await asyncio.wait_for(execution.execute_call(call), 10)
            assert result.outcome == "success"
            await asyncio.wait_for(control.entered.get(), 10)
        before = log.catalog().snapshot_heads()
        refused = await execution.execute_call(additional_call(log, reply, 3))
        assert refused.outcome == "error" and "capacity reached" in text_part(refused.parts[0])
        assert set(before) == set(log.catalog().snapshot_heads())
        children = [log.reader(key) for key in before if key.startswith("subagent:")]
        request = next(mapping_part(part) for part in children[0].snapshot()[0].body.parts if isinstance(part, ContentPart) and part.kind == "subagent.request")
        manage = await _bind_tool(host, "spawn_manage")
        cancelled = await asyncio.wait_for(
            execution.execute(
                "cancel", manage, {"action": "cancel", "job_id": request["job_id"]}
            ),
            10,
        )
        assert cancelled.outcome == "success" and "cancel_requested" in text_part(
            cancelled.parts[0]
        )
        assert all(not isinstance(row.body, Output) for row in children[0].snapshot())
        control.release.set()
        async def completed():
            async for _ in log.catalog().follow():
                outputs = [row for row in log.reader("test:parent").snapshot()
                           if row.source.startswith("subagent:") and isinstance(row.body, Output)
                           and row.body.finish == "complete"]
                if len(outputs) == 3:
                    return outputs
        inputs = await asyncio.wait_for(completed(), 10)
        assert inputs is not None
        assert sum("cancelled" in text_part(row.body.parts[0]) for row in inputs) == 1
        assert sum("child finished" in text_part(row.body.parts[0]) for row in inputs) == 2
        assert control.calls == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["started", "finished", "announced"])
async def test_background_reopen_keeps_input_and_tool_choice_and_only_returns_once(tmp_path, monkeypatch, stage):
    fault = asyncio.Event()
    original_append = MessageWriter.append
    def append(writer, identity, body, **kwargs):
        if writer.session_id == "test:parent" and writer.source.startswith("subagent:") and isinstance(body, Output) and body.finish == "complete":
            if stage == "announced":
                original_append(writer, identity, body, **kwargs)
            fault.set()
            raise OSError("crash at parent handoff")
        return original_append(writer, identity, body, **kwargs)
    if stage != "started":
        monkeypatch.setattr(MessageWriter, "append", append)
    async with application(
        tmp_path, background=True, block=stage == "started", main_tool=stage == "started",
    ) as (host, log, execution, reply):
        receipt = await execution.execute_call(reply)
        control = CONTROLS[str(tmp_path)]
        if stage == "started":
            # The live watcher entered the child program and is held at the model boundary.
            await asyncio.wait_for(control.entered.get(), 10)
        else:
            await asyncio.wait_for(fault.wait(), 10)
            monkeypatch.setattr(MessageWriter, "append", original_append)
        assert receipt.outcome == "success"
        session_id = next(key for key in log.catalog().snapshot_heads() if key.startswith("subagent:"))
        original = log.reader(session_id).snapshot()
        assert len(original) == (1 if stage == "started" else 4)
        assert control.calls == (1 if stage == "started" else 2)
        model_generation = host.generation("models_fixture")
        assert model_generation is not None
        stable_model = model_generation.archive_ref
        await host.terminate_all()
        log.close()
        control.release.set()
        # 当前插件处理原已接纳事实；已完成结果不因源码变化重算。
        provider = tmp_path / "plugins/models_fixture/plugin.py"
        _write_python_source(
            provider,
            provider.read_text().replace('LLMResponse("child finished")', 'LLMResponse("new provider result")')
            .replace('control.sent.put_nowait((key, address, message))',
                     'raise RuntimeError("current sender must not replace the original")'),
        )
        files = tmp_path / "plugins/standard_tools/files.py"
        _write_python_source(
            files,
            files.read_text().replace(
                "        value = (", '        raw["content"] = "new tool content"\n        value = ('
            ),
        )
        workspace = tmp_path / "workspace"
        reopened = MessageLog(workspace / "sessions.db")
        metadata = ArtifactStore(workspace / "sessions.db")
        artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
        resumed_bus = EventBus()
        resumed_gate = RestartGate(
            boot_id=f"subagent-reopen-{tmp_path.name}", supervised=True,
            commit=lambda _request_id: None,
        )
        resumed_gate.prepare("hold-default-reply")
        resumed = PluginManager([tmp_path / "plugins"], event_bus=resumed_bus, workspace=workspace,
                                installed_cache_root=tmp_path / "cache", message_log=reopened,
                                channel_attachment_store=artifacts, restart_gate=resumed_gate)
        try:
            await resumed.load_all()
            restored_model = resumed.generation("models_fixture")
            restored_tools = resumed.generation("standard_tools")
            assert restored_model is not None and restored_model.archive_ref == stable_model
            assert restored_tools is not None
            stable_tools = restored_tools.archive_ref
            # 重启先恢复原选择；本地 Loader 再显式接受当前源码换代。
            changes = await resumed.reconcile_changed()
            changed = {item["plugin_id"]: item for item in changes}
            for plugin_id in ("models_fixture", "standard_tools"):
                assert changed[plugin_id]["publication_state"] == "active"
            current_model = resumed.generation("models_fixture")
            current_tools = resumed.generation("standard_tools")
            assert current_model is not None and current_model.archive_ref != stable_model
            assert current_tools is not None and current_tools.archive_ref != stable_tools
            await _capture_report_owners(resumed, control)
            async def completed():
                async for _ in reopened.catalog().follow():
                    messages = reopened.reader("test:parent").snapshot()
                    returned = [message for message in messages if isinstance(message.body, Output)
                                and message.source.startswith("subagent:") and message.body.finish == "complete"]
                    if returned:
                        return returned
            returned = await asyncio.wait_for(completed(), 10)
            assert returned is not None
            expected_result = "new provider result" if stage == "started" else "child finished"
            assert len(returned) == 1 and f"main summary: {expected_result}" in text_part(returned[0].body.parts[0])
            # 当前 sender 报错时保留原任务事实，不伪造送达。
            assert CONTROLS[str(tmp_path)].sent.empty()
            expected_main_calls = 2 if stage in {"started", "finished"} else 1
            assert CONTROLS[str(tmp_path)].main_calls == expected_main_calls
            assert CONTROLS[str(tmp_path)].report_owner_checks == expected_main_calls
            assert ("new provider result" in text_part(returned[0].body.parts[0])) == (stage == "started")
            assert reopened.reader(session_id).snapshot()[0] == original[0]
            assert CONTROLS[str(tmp_path)].calls == (3 if stage == "started" else 2)
            if stage == "started":
                assert (workspace / "main-report.txt").read_text() == "new tool content"
            request = next(mapping_part(part) for part in original[0].body.parts if isinstance(part, ContentPart) and part.kind == "subagent.request")
            task_dir = workspace / "subagent-runs" / request["job_id"]
            assert (task_dir / "answer.txt").read_text() == "once"
            subagent_context = _provider_context(resumed, "subagent")
            async with subagent_context.runtime_scope():
                bindings = subagent_context.require(BINDINGS)
                tools_value = request.get("tools")
                assert isinstance(tools_value, Mapping)
                write_binding = tools_value.get("write_file")
                assert isinstance(write_binding, str)
                description = bindings.describe(write_binding, TOOLS)
                assert isinstance(description, Mapping)
                state = description.get("state")
                assert isinstance(state, Mapping)
                assert state.get("allowed_dir") == str(task_dir)
            await resumed.terminate_all()
            await resumed_bus.aclose()
            reopened.close()
            metadata.close()
            reopened = MessageLog(workspace / "sessions.db")
            metadata = ArtifactStore(workspace / "sessions.db")
            artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
            resumed_bus = EventBus()
            resumed_gate = RestartGate(
                boot_id=f"subagent-reopen-check-{tmp_path.name}", supervised=True,
                commit=lambda _request_id: None,
            )
            resumed_gate.prepare("hold-default-reply")
            resumed = PluginManager([tmp_path / "plugins"], event_bus=resumed_bus, workspace=workspace,
                                    installed_cache_root=tmp_path / "cache", message_log=reopened,
                                    channel_attachment_store=artifacts, restart_gate=resumed_gate)
            await resumed.load_all()
            # 同步读取持久日志，不用延迟猜测是否重复；已结算原指针不能再接纳 Task。
            manage = await _bind_tool(resumed, "spawn_manage")
            bindings_context = _provider_context(resumed, "tools")
            async with bindings_context.runtime_scope():
                bindings = bindings_context.require(BINDINGS)
                async with open_tool(bindings, manage) as tool:
                    result = await tool.invoke("list", {"action": "list"})
                    assert '"running_count": 0' in text_part(result.parts[0])
            completed_messages = [
                message for message in reopened.reader("test:parent").snapshot()
                if isinstance(message.body, Output)
                and message.source.startswith("subagent:")
                and message.body.finish == "complete"
            ]
            assert len(completed_messages) == 1
            assert CONTROLS[str(tmp_path)].calls == (3 if stage == "started" else 2)
        finally:
            await resumed.terminate_all()
            await resumed_bus.aclose()
            reopened.close()
            metadata.close()


@pytest.mark.asyncio
async def test_background_main_program_keeps_tools_and_new_input_interrupts_it(tmp_path):
    async with application(tmp_path, background=True, block_main=True, main_tool=True) as (host, log, execution, reply):
        control = CONTROLS[str(tmp_path)]
        await execution.execute_call(reply)
        request = await asyncio.wait_for(control.main_entered.get(), 10)
        assert "## 后台任务结果" in str(request.messages[-1])
        assert request.messages[-1]["role"] == "user"
        assert all("## 后台任务结果" not in str(item) for item in request.messages if item["role"] == "system")
        assert any(tool["function"]["name"] == "write_file" for tool in request.tools)
        assert control.restart_gate is not None
        control.restart_gate.abort("hold-default-reply")
        root = host.live_root
        assert root is not None
        await root.context.require(CHANNEL_INPUT)(
            "test:parent",
            "human-followup",
            ChannelInboundMessage(
                "test", "user", "parent", "[human followup]", datetime.now(UTC), {},
            ),
        )
        control.main_release.set()
        _, address, result = await asyncio.wait_for(control.sent.get(), 10)
        assert address == "parent" and "main summary" in text_part(result.body.parts[0])
        rows = log.reader("test:parent").snapshot()
        assert [item.message_id for item in rows if isinstance(item.body, Input)] == ["parent-input", "human-followup"]
        assert any(item.source == "conversation" and isinstance(item.body, Output) and item.body.finish == "complete" for item in rows)
        report = [item for item in rows if item.source.startswith("subagent:")]
        assert [type(item.body) for item in report] == [Output, ToolResult, Output]
        assert control.main_calls == 3
        assert control.report_owner_checks == control.main_calls
        assert (tmp_path / "workspace/main-report.txt").read_text() == "main result"
