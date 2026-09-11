import asyncio
from collections.abc import AsyncIterator
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities,
    ModelExecution, ModelRole, ToolCall as ModelToolCall,
)
from agent.plugin_composition.tasks import TASKS
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from agent.plugin_contracts.content import CONTENT, check_text
from agent.plugin_contracts.context import MATERIALS
from agent.plugin_contracts.context import CONTEXT
from plugins.conversation.program import run_reply
from plugins.conversation.source import Conversation
from agent.plugin_contracts.model_content import render_content
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.react.plugin import react
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS
from agent.plugin_contracts.turn_projection import TURN_PROJECTION
from session.log import MessageLog
from session.message import ContentPart, Control, Input, Output, ToolResult


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["complete", "interrupt", "input_before_effect", "input_during_reduction", "summarized_input"])
async def test_ordinary_program_keeps_content_live_until_real_tool_settlement(tmp_path, case, monkeypatch):
    sources = tmp_path / "plugins"
    for name in ("content", "context", "tools", "turn_projection"):
        shutil.copytree(Path(__file__).resolve().parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    path = sources / "probe"
    path.mkdir()
    (path / "plugin.py").write_text('''
import asyncio
from contextlib import asynccontextmanager
from agent.plugin_composition import ServiceKey
from agent.plugin_contracts.tool_api import Result
from session.message import ContentPart
api_version = 3
name = "probe"
version = "1.0.0"
inject = (ServiceKey("tools.v1"),)
async def apply(ctx, config):
    class Target:
        idempotent = False
        async def prepare(self, args, source=None):
            return args
        async def invoke(self, key, args):
            reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
            try:
                result = (await reader.readline()).decode().strip()
                return Result("success", (ContentPart("text", result),))
            finally:
                writer.close()
                await writer.wait_closed()
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open(state):
        yield Target()
    await ctx.require(inject[0]).register(ctx, name="example", description="write test file",
        parameters={"type":"object"}, open=open)
    await ctx.provide(ServiceKey("probe"), ctx)
'''.replace("SOCKET_PATH", repr(str(tmp_path / "tool.sock"))))
    log = MessageLog(tmp_path / "sessions.db")
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    requests = []
    entered, release = asyncio.Event(), asyncio.Event()
    authorizing, authorized = asyncio.Event(), asyncio.Event()
    if case == "summarized_input":
        from dataclasses import replace
        from agent.plugin_contracts.context import Summary
        from plugins.context.materials import MaterialView
        original_prepare = MaterialView.prepare
        async def summarized_prepare(self, messages, source, **kwargs):
            prepared = await original_prepare(self, messages, source, **kwargs)
            if any(isinstance(message.body, ToolResult) for message in messages):
                return replace(prepared, summary=Summary(
                    "published", tuple(message.message_id for message in messages[2:]), "tool work summary"))
            return prepared
        monkeypatch.setattr(MaterialView, "prepare", summarized_prepare)
    if case == "input_during_reduction":
        from plugins.context.materials import MaterialView
        original_reduce = MaterialView.reduce
        async def delayed_reduce(self, *args, **kwargs):
            if not authorizing.is_set():
                authorizing.set()
                await authorized.wait()
            return await original_reduce(self, *args, **kwargs)
        monkeypatch.setattr(MaterialView, "reduce", delayed_reduce)
    async def serve(reader, writer):
        entered.set()
        await release.wait()
        (tmp_path / "effect.txt").write_text("written")
        writer.write(b"written\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_unix_server(serve, path=tmp_path / "tool.sock")
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools=()):
            return 50
        def estimate_appended_message_tokens(self, messages):
            return 0
        async def complete(self, request):
            requests.append(request)
            if len(requests) == 1:
                return LLMResponse("working", [ModelToolCall("provider-call", "example", {})])
            if case == "summarized_input":
                log.save_binding("published", {"target": "summary-test"})
                assert [row["role"] for row in request.messages] == ["user", "user", "user"]
                assert request.messages[1]["content"][0]["text"] == "first"
                assert request.messages[2]["content"][0]["text"] == "second"
                assert '"summary":"tool work summary"' in request.messages[0]["content"]
                assert "old input" not in str(request.messages)
                assert "other source" not in str(request.messages)
            return LLMResponse("finished")
    descriptor = BoundModelDescriptor(
        binding_id="model", plugin_snapshot_id="snapshot", model_revision=0,
        model_id="model", connection_id="connection", driver_id="driver",
        driver_contract_version="1", auth_identity="test", model="test", role=ModelRole.AGENT,
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="test",
    )
    model = _BoundChat(descriptor, Driver(), store)
    class Execution:
        def chat(self, role: ModelRole) -> _BoundChat:
            assert role in ModelRole
            return model

    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None) -> AsyncIterator[ModelExecution]:
            del model_id, reasoning_effort
            yield Execution()

        independent_execution = execution
    async def authorize(binding, arguments):
        if case == "input_before_effect":
            authorizing.set()
            await authorized.wait()
        return {"decision": "allowed"}
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            ctx = root.require(ServiceKey("probe"))
            def writer(body):
                return root.require(MESSAGE_WRITERS).bind(
                    ctx, author="user", source="conversation", body_types=(body,),
                    content={"text": check_text},
                )("s")
            async def run(task, reader, source):
                return await run_reply(
                    ctx, task, reader, source, models=Models(), content=root.require(CONTENT),
                    context=root.require(CONTEXT), tools=root.require(TOOLS), react=react,
                    materials=root.require(MATERIALS), render_content=lambda part: render_content(part, artifacts={}),
                    turn_projection=root.require(TURN_PROJECTION),
                    read_call=store.read_call,
                    authorize=authorize,
                    tool_view=root.require(ALL_TOOLS)(),
                    max_output_tokens=100,
                    max_steps=4,
                )
            conversation = Conversation(
                reader=log.reader("s"), inputs=writer(Input), controls=writer(Control),
                tasks=root.require(TASKS).open(ctx),
            )
            if case == "summarized_input":
                writer(Input).append("old-u", Input((ContentPart("text", "old input"),)))
                writer(Output).append("old-a", Output((ContentPart("text", "old answer"),), "complete"))
                log.writer("s", author="test", source="wake", body_types=(Input,), content={"text": check_text}).append(
                    "other", Input((ContentPart("text", "other source"),)))
            await conversation.accept("u1", Input((ContentPart("text", "first"),)))
            if case == "summarized_input":
                await conversation.accept("u2", Input((ContentPart("text", "second"),)))
            task = await conversation.start(run)
            joined = asyncio.create_task(task.join())
            ready = asyncio.create_task((authorizing if case in {"input_before_effect", "input_during_reduction"} else entered).wait())
            try:
                done, _ = await asyncio.wait((joined, ready), timeout=10, return_when=asyncio.FIRST_COMPLETED)
                if joined in done:
                    joined.result()
                assert ready in done, "tool did not enter"
            finally:
                ready.cancel()
                await asyncio.gather(ready, return_exceptions=True)
            if case == "interrupt":
                await conversation.accept("u2", Input((ContentPart("text", "second"),)))
            if case in {"input_before_effect", "input_during_reduction"}:
                # 直接追加模拟日志通知尚未送达，Source Task 仍是 active。
                writer(Input).append("u2", Input((ContentPart("text", "second"),)))
                assert task.active
                authorized.set()
            release.set()
            if case not in {"complete", "summarized_input"}:
                with pytest.raises(asyncio.CancelledError):
                    await joined
                if case in {"input_before_effect", "input_during_reduction"}:
                    assert not entered.is_set()
                    assert not any(isinstance(item.body, ToolResult) for item in log.reader("s").snapshot())
                if case == "input_during_reduction":
                    assert requests == []
                next_task = await conversation.start(run)
                await next_task.join()
            else:
                await joined
            messages = log.reader("s").snapshot()
            assert sum(isinstance(item.body, ToolResult) for item in messages) == 1
            assert next(item.body for item in messages if isinstance(item.body, ToolResult)).outcome == "success"
            assert isinstance(messages[-1].body, Output) and messages[-1].body.finish == "complete"
            assert (tmp_path / "effect.txt").read_text() == "written"
            assert any(row["role"] == "tool" for row in requests[-1].messages) is (case != "summarized_input")
    finally:
        authorized.set()
        release.set()
        await host.terminate_all()
        server.close()
        await server.wait_closed()
        log.close()
