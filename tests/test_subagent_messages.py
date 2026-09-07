import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import Tasks
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.log import MessageLog, MessageWriter
from session.artifact_store import ArtifactStore
from plugins.content.plugin import check_text
from plugins.conversation.plugin import check_origin
from plugins.tools.api import MessageReply
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import TOOLS, open_tool
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from tests.test_standard_tools import environment


@dataclass
class ModelControl:
    entered: asyncio.Queue = field(default_factory=asyncio.Queue)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    main_calls: int = 0
    main_entered: asyncio.Queue = field(default_factory=asyncio.Queue)
    main_release: asyncio.Event = field(default_factory=asyncio.Event)
    main_tool: bool = False
    sent: asyncio.Queue = field(default_factory=asyncio.Queue)


CONTROLS: dict[str, ModelControl] = {}


@asynccontextmanager
async def application(tmp_path, *, background=False, start=True, block=False, block_main=False, main_tool=False):
    host, store, log, artifacts, sources = environment(tmp_path, reply=True)
    for name in ("sources", "conversation", "react", "subagent", "reply", "delivery", "delivery_policy"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    provider = sources / "models_fixture"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from agent.plugin_composition import CHAT_MODELS
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.delivery.api import Receipt
import json
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ModelRole, ToolCall
from plugins.models.projection import MODEL_CALLS
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
api_version = 3
name = "models_fixture"
inject = (DELIVERY_SENDERS,)
version = "1.0.0"
async def apply(ctx, config):
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
        auth_identity="fixture", model="fixture", role=ModelRole.AGENT, reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=10000), capability_sources=CapabilitySources(), capability_digest="fixture")
    model = _BoundChat(descriptor, Driver(), store)
    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            yield SimpleNamespace(chat=lambda role: model)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    class Sender:
        idempotent = True
        async def send(self, key, address, message):
            control.sent.put_nowait((key, address, message))
            return Receipt(status="delivered", provider_ids=(key,))
        async def query(self, key, address):
            return None
    @asynccontextmanager
    async def sender():
        yield Sender()
    control = CONTROLS[CONTROL_PATH]
    await ctx.require(DELIVERY_SENDERS).register(ctx, name="test", idempotent=True, open=sender)
''')
    control = ModelControl(main_tool=main_tool)
    if not block_main:
        control.main_release.set()
    if not block:
        control.release.set()
    CONTROLS[str(tmp_path)] = control
    module = provider / "plugin.py"
    text = module.read_text().replace("async def apply(ctx, config):", "from " + __name__ + " import CONTROLS\nasync def apply(ctx, config):")
    text = text.replace("CONTROL_PATH", repr(str(tmp_path)))
    text = text.replace("        async def complete(self, request):", "        async def complete(self, request):\n            control = CONTROLS[" + repr(str(tmp_path)) + "]\n            if 'background_task_result' in str(request.messages):\n                control.main_calls += 1\n                control.main_entered.put_nowait(request)\n                await control.main_release.wait()\n                if control.main_tool and 'main-report.txt' not in str(request.messages[:-1]):\n                    return LLMResponse(None, [ToolCall('main-write', 'write_file', {'path': CONTROL_REPORT_PATH, 'content': 'main result'})])\n                return LLMResponse('main summary: ' + ('cancelled' if 'cancelled' in str(request.messages[-1]) else 'child finished'))\n            if '[human followup]' in str(request.messages):\n                return LLMResponse('human answer')\n            control.calls += 1\n            control.entered.put_nowait(request)\n            await control.release.wait()")
    text = text.replace("CONTROL_REPORT_PATH", repr(str(tmp_path / "workspace/main-report.txt")))
    module.write_text(text)
    tasks = Tasks()
    try:
        await host.load_all()
        if start:
            await host.start_runtime()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            bindings = snapshot.composition_root.context.require(BINDINGS)
            tools = snapshot.composition_root.context.require(TOOLS)
            binding = tools.bind("spawn", bindings)
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
        execution = ToolExecution(log.owner("plugin:tools"), tasks, partial(open_tool, bindings), authorize, task_key="effects")
        yield host, log, execution, reply
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()


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
        assert result.outcome == "success" and "child finished" in result.parts[0].value
        sessions = [key for key in log.catalog().snapshot_heads() if key.startswith("subagent:")]
        assert len(sessions) == 1
        reader = log.reader(sessions[0])
        assert reader.attributes.visibility == "internal" and reader.attributes.learning == "excluded"
        rows = reader.snapshot()
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
        request = next(part.value for part in rows[0].body.parts if part.kind == "subagent.request")
        path = tmp_path / "workspace/subagent-runs" / request["job_id"] / "answer.txt"
        assert path.read_text() == "once"
        stamp = path.stat().st_mtime_ns
        assert await execution.execute_call(reply) == result
        assert reader.snapshot() == rows and path.stat().st_mtime_ns == stamp
        assert len([row for row in log.reader("test:parent").snapshot() if isinstance(row.body, Input)]) == 1
        if broken_trace:
            assert trace.read_text() == "existing diagnostic\n"


@pytest.mark.asyncio
async def test_background_spawn_returns_receipt_and_returns_result_once(tmp_path):
    async with application(tmp_path, background=True) as (host, log, execution, reply):
        result = await asyncio.wait_for(execution.execute_call(reply), 15)
        assert result.outcome == "success" and "已创建后台任务" in result.parts[0].value
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("test:parent").snapshot()
                outputs = [row for row in rows if row.source.startswith("subagent:")
                           and isinstance(row.body, Output) and row.body.finish == "complete"]
                if outputs:
                    return outputs[-1]
        message = await asyncio.wait_for(completed(), 15)
        assert "child finished" in message.body.parts[0].value
        assert "main summary" in message.body.parts[0].value
        _, address, sent = await asyncio.wait_for(CONTROLS[str(tmp_path)].sent.get(), 10)
        assert address == "parent" and sent == message
        assert CONTROLS[str(tmp_path)].main_calls == 1
        assert await execution.execute_call(reply) == result
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
        assert refused.outcome == "error" and "capacity reached" in refused.parts[0].value
        assert set(before) == set(log.catalog().snapshot_heads())
        children = [log.reader(key) for key in before if key.startswith("subagent:")]
        request = next(part.value for part in children[0].snapshot()[0].body.parts if part.kind == "subagent.request")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            manage = ctx.require(TOOLS).bind("spawn_manage", ctx.require(BINDINGS))
        cancelled = await asyncio.wait_for(execution.execute("cancel", manage, {"action": "cancel", "job_id": request["job_id"]}), 10)
        assert cancelled.outcome == "success" and "cancel_requested" in cancelled.parts[0].value
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
        assert sum("cancelled" in row.body.parts[0].value for row in inputs) == 1
        assert sum("child finished" in row.body.parts[0].value for row in inputs) == 2
        assert control.calls == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["input", "finished", "announced"])
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
    if stage != "input":
        monkeypatch.setattr(MessageWriter, "append", append)
    async with application(tmp_path, background=True, start=stage != "input") as (host, log, execution, reply):
        receipt = await execution.execute_call(reply)
        if stage != "input":
            await asyncio.wait_for(fault.wait(), 10)
            monkeypatch.setattr(MessageWriter, "append", original_append)
        assert receipt.outcome == "success"
        session_id = next(key for key in log.catalog().snapshot_heads() if key.startswith("subagent:"))
        original = log.reader(session_id).snapshot()
        assert len(original) == (1 if stage == "input" else 4)
        assert CONTROLS[str(tmp_path)].calls == (0 if stage == "input" else 2)
        await host.terminate_all()
        log.close()
        # 原已接纳程序和工具来自归档；当前文件的行为变化不应改写原任务。
        provider = tmp_path / "plugins/models_fixture/plugin.py"
        provider.write_text(provider.read_text().replace('LLMResponse("child finished")', 'LLMResponse("new provider result")')
                            .replace('control.sent.put_nowait((key, address, message))',
                                     'raise RuntimeError("current sender must not replace the original")'))
        files = tmp_path / "plugins/standard_tools/files.py"
        files.write_text(files.read_text().replace("        value = (", '        raw["content"] = "new tool content"\n        value = ('))
        workspace = tmp_path / "workspace"
        reopened = MessageLog(workspace / "sessions.db")
        metadata = ArtifactStore(workspace / "sessions.db")
        artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
        resumed = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=workspace,
                                installed_cache_root=tmp_path / "cache", message_log=reopened,
                                channel_attachment_store=artifacts)
        try:
            await resumed.load_all()
            await resumed.start_runtime()
            async def completed():
                async for _ in reopened.catalog().follow():
                    messages = reopened.reader("test:parent").snapshot()
                    returned = [message for message in messages if isinstance(message.body, Output)
                                and message.source.startswith("subagent:") and message.body.finish == "complete"]
                    if returned:
                        return returned
            returned = await asyncio.wait_for(completed(), 10)
            assert len(returned) == 1 and "main summary: child finished" in returned[0].body.parts[0].value
            _, address, sent = await asyncio.wait_for(CONTROLS[str(tmp_path)].sent.get(), 10)
            assert address == "parent" and sent == returned[0]
            assert CONTROLS[str(tmp_path)].main_calls == (2 if stage == "finished" else 1)
            assert "new provider result" not in returned[0].body.parts[0].value
            assert reopened.reader(session_id).snapshot()[0] == original[0]
            assert CONTROLS[str(tmp_path)].calls == 2
            request = next(part.value for part in original[0].body.parts if part.kind == "subagent.request")
            task_dir = workspace / "subagent-runs" / request["job_id"]
            assert (task_dir / "answer.txt").read_text() == "once"
            async with lease_runtime_snapshot(resumed.snapshot_store) as snapshot:
                bindings = snapshot.composition_root.context.require(BINDINGS)
                assert bindings.describe(request["tools"]["write_file"], TOOLS)["state"]["allowed_dir"] == str(task_dir)
            await resumed.terminate_all()
            reopened.close()
            metadata.close()
            reopened = MessageLog(workspace / "sessions.db")
            metadata = ArtifactStore(workspace / "sessions.db")
            artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
            resumed = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=workspace,
                                    installed_cache_root=tmp_path / "cache", message_log=reopened,
                                    channel_attachment_store=artifacts)
            await resumed.load_all()
            await resumed.start_runtime()
            # 同步读取持久日志，不用延迟猜测是否重复；已结算原指针不能再接纳 Task。
            async with lease_runtime_snapshot(resumed.snapshot_store) as snapshot:
                root = snapshot.composition_root
                context = root.context
                # 真实管理工具重读已结算来源，无活动 job。
                bindings = context.require(BINDINGS)
                manage = context.require(TOOLS).bind("spawn_manage", bindings)
                async with open_tool(bindings, manage) as tool:
                    result = await tool.invoke("list", {"action": "list"})
                    assert '"running_count": 0' in result.parts[0].value
            assert len(await completed()) == 1 and CONTROLS[str(tmp_path)].calls == 2
        finally:
            await resumed.terminate_all()
            reopened.close()
            metadata.close()


@pytest.mark.asyncio
async def test_background_main_program_keeps_tools_and_new_input_interrupts_it(tmp_path):
    from plugins.conversation.plugin import CONVERSATION
    async with application(tmp_path, background=True, block_main=True, main_tool=True) as (host, log, execution, reply):
        control = CONTROLS[str(tmp_path)]
        await execution.execute_call(reply)
        request = await asyncio.wait_for(control.main_entered.get(), 10)
        assert "background_task_result" in str(request.messages[-1])
        assert request.messages[-1]["role"] == "user"
        assert all("background_task_result" not in str(item) for item in request.messages if item["role"] == "system")
        assert any(tool["function"]["name"] == "write_file" for tool in request.tools)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            conversation = snapshot.composition_root.context.require(CONVERSATION)("test:parent")
            await conversation.accept("human-followup", Input((ContentPart("text", "[human followup]"),)))
        control.main_release.set()
        _, address, result = await asyncio.wait_for(control.sent.get(), 10)
        assert address == "parent" and "main summary" in result.body.parts[0].value
        rows = log.reader("test:parent").snapshot()
        assert [item.message_id for item in rows if isinstance(item.body, Input)] == ["parent-input", "human-followup"]
        assert any(item.source == "conversation" and isinstance(item.body, Output) and item.body.finish == "complete" for item in rows)
        report = [item for item in rows if item.source.startswith("subagent:")]
        assert [type(item.body) for item in report] == [Output, ToolResult, Output]
        assert control.main_calls == 3
        assert (tmp_path / "workspace/main-report.txt").read_text() == "main result"
