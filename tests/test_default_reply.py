import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog
from session.message import Input, Output, ToolResult


@asynccontextmanager
async def application(tmp_path, *, replying, start=True, missing_tool=False, discovery=False, compaction=False,
                      output_tokens=4096, keep_recent_tokens=128, summary_padding=0):
    sources = tmp_path / "plugins"
    for name in ("content", "context", "tools", "conversation", "react", "turn_projection", *(('reply',) if replying else ())):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    if discovery:
        shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                        ignore=shutil.ignore_patterns("__pycache__"))
    if compaction:
        shutil.copytree(Path(__file__).parents[1] / "plugins/compaction", sources / "compaction",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (sources / "compaction/akashic.plugin.toml").write_text(
            'schema_version = 1\nname = "compaction"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')
        module = sources / "context/plugin.py"
        module.write_text(module.read_text().replace('summary_source: tuple[str, str] | None = None',
            'summary_source: tuple[str, str] | None = ("compaction", "compaction")'))
        module = sources / "compaction/message_plugin.py"
        module.write_text(module.read_text().replace('Field(default=20_000,', f'Field(default={keep_recent_tokens},'))
        reply = sources / 'reply/plugin.py'
        reply.write_text(reply.read_text().replace('Field(default=4096,', f'Field(default={output_tokens},'))
    if missing_tool:
        reply = sources / "reply/plugin.py"
        reply.write_text(reply.read_text().replace(
            "tools: tuple[str, ...] | None = None", 'tools: tuple[str, ...] | None = ("gone",)'))
    provider = sources / "test_provider"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from types import SimpleNamespace
from pathlib import Path
from agent.plugin_composition import CHAT_MODELS, ServiceKey
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ModelRole, ToolCall
from plugins.models.projection import MODEL_CALLS
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.tools.api import Result
from plugins.tools.plugin import TOOLS
from session.message import ContentPart
api_version = 3
name = "test_provider"
version = "1.0.0"
inject = (TOOLS,)
async def apply(ctx, config):
    calls = []
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            calls.append(request)
            if len(calls) == 1:
                return LLMResponse(None, [ToolCall("provider-call", "write_evidence", {})])
            return LLMResponse("finished")
    descriptor = BoundModelDescriptor(
        binding_id="fixture-model", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture",
        driver_contract_version="1", auth_identity="fixture", model="fixture", role=ModelRole.AGENT,
        reasoning_effort=None, capabilities=ModelCapabilities(context_window=10000),
        capability_sources=CapabilitySources(), capability_digest="fixture",
    )
    model = _BoundChat(descriptor, Driver(), store)
    class Models:
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            yield SimpleNamespace(chat=lambda role: model)
    class Target:
        idempotent = False
        async def prepare(self, args, source=None):
            return args
        async def invoke(self, key, args):
            with Path(EFFECT_PATH).open("a") as handle:
                handle.write("once\\n")
            return Result("success", (ContentPart("text", "written"),))
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open(state):
        yield Target()
    await ctx.require(TOOLS).register(ctx, name="write_evidence", description="record local test evidence",
        parameters={"type":"object"}, open=open)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(ServiceKey("fixture.calls"), calls)
'''.replace("EFFECT_PATH", repr(str(tmp_path / "effect.txt"))))
    if discovery:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace(
            'if len(calls) == 1:',
            'if len(calls) == 1:\n                return LLMResponse(None, [ToolCall("search-call", "tool_search", {"query": "select:write_evidence"})])\n            if len(calls) == 2:'))
    if compaction:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace('calls = []', 'calls = []\n    business = []').replace(
            'return 10', 'return len(str(messages)) // 4').replace('if len(calls) == 1:', '''if "[Source messages]" in str(request.messages):
                from plugins.compaction.message_summary import HEADINGS
                return LLMResponse("\\n".join(heading + "\\nPreserved facts." for heading in HEADINGS))
            business.append(request)
            if len(business) == 1:''').replace("Preserved facts.", "Preserved facts." + "z" * summary_padding))
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        if start:
            await host._start_current_runtime_snapshot()
        yield log, host
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replying", [False, True])
async def test_installed_default_reply_is_an_independent_log_consumer(tmp_path, replying):
    from agent.plugin_composition import ServiceKey
    async with application(tmp_path, replying=replying) as (log, host):
        message = ChannelInboundMessage("test", "user", "room", "do the work",
                                        datetime(2026, 9, 5, tzinfo=UTC), {})
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            accept = snapshot.composition_root.context.require(CHANNEL_INPUT)
            accepted = await accept("test:room", "u1", message)
        assert isinstance(accepted.body, Input)
        if replying:
            async def completed():
                async for _ in log.catalog().follow():
                    rows = log.reader("test:room").snapshot()
                    if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                        return rows
            rows = await asyncio.wait_for(completed(), 5)
            assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
            assert (tmp_path / "effect.txt").read_text() == "once\n"
        else:
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
                assert calls == []
                assert all("[Source messages]" in str(call.messages) for call in calls)
            assert log.reader("test:room").snapshot() == (accepted,)
            assert not (tmp_path / "effect.txt").exists()


@pytest.mark.asyncio
async def test_bad_reply_tool_configuration_fails_before_consuming_any_input(tmp_path):
    async with application(tmp_path, replying=True, start=False, missing_tool=True) as (log, host):
        with pytest.raises(ValueError, match="未安装的工具"):
            await host._start_current_runtime_snapshot()
        assert log.catalog().snapshot_heads() == {}


@pytest.mark.asyncio
async def test_default_reply_discovers_then_calls_tool_without_react_search_branch(tmp_path):
    from agent.plugin_composition import ServiceKey
    async with application(tmp_path, replying=True, discovery=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "s", "u", ChannelInboundMessage("test", "user", "s", "record evidence",
                                                 datetime(2026, 9, 5, tzinfo=UTC), {}))
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("s").snapshot()
                if isinstance(rows[-1].body, Output) and rows[-1].body.finish == "complete":
                    return rows
        rows = await asyncio.wait_for(completed(), 5)
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output, ToolResult, Output]
        assert rows[2].body.parts[-1].kind == "tool.selection"
        assert (tmp_path / "effect.txt").read_text() == "once\n"
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
            assert [tool["function"]["name"] for tool in calls[0].tools] == ["tool_search"]
            assert {tool["function"]["name"] for tool in calls[1].tools} == {"tool_search", "write_evidence"}


@pytest.mark.asyncio
@pytest.mark.parametrize("has_cut,soft_only,large_summary", [(True, False, False), (False, False, False), (False, True, False), (False, False, True)])
async def test_actual_reply_compacts_history_before_provider_and_records_each_successful_use(tmp_path, has_cut, soft_only, large_summary):
    from agent.plugin_composition import ServiceKey
    from plugins.compaction.records import SummaryRecords
    from plugins.content.plugin import check_text
    from plugins.models.projection import MODEL_CALLS
    from session.message import ContentPart, Control

    async with application(tmp_path, replying=True, start=False, compaction=True,
                           output_tokens=1000 if soft_only or large_summary else 4096,
                           keep_recent_tokens=6000 if soft_only else 128, summary_padding=2400 if large_summary else 0) as (log, host):
        writer = log.writer("s", author="test", source="conversation", body_types=(Input, Output),
                            content={"text": check_text})
        size = 5000 if has_cut or soft_only or large_summary else 6000
        for index in range(4 if has_cut or large_summary else 3):
            writer.append(f"old-u{index}", Input((ContentPart("text", f"old input {index}: " + "x" * size),)))
            writer.append(f"old-a{index}", Output((ContentPart("text", f"old answer {index}: " + "y" * size),), "complete"))
        writer.append("current", Input((ContentPart("text", "current request"),)))
        original = log.reader("s").snapshot()
        await host._start_current_runtime_snapshot()
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("s").snapshot()
                if any(row.seq > original[-1].seq and (
                    isinstance(row.body, Output) and row.body.finish == "complete"
                    or isinstance(row.body, Control) and row.body.action == "failure") for row in rows):
                    return rows
        rows = await asyncio.wait_for(completed(), 10)
        assert rows[:len(original)] == original
        record = SummaryRecords(log.owner("plugin:compaction")).head("s")
        if not has_cut:
            assert record is None
            assert isinstance(rows[-1].body, Control) and rows[-1].body.action == "failure"
            if large_summary:
                assert "摘要后的完整请求仍超过" in rows[-1].body.reason
            elif soft_only:
                assert "近期原文保留量内没有合法摘要切点" in rows[-1].body.reason
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
                assert len(calls) == (1 if large_summary else 0)
                assert all("[Source messages]" in str(call.messages) for call in calls)
            assert not (tmp_path / "effect.txt").exists()
            return
        assert record is not None and record.tokens_after < record.tokens_before
        assert record.source_message_ids == tuple(row.message_id for row in original[4:6])
        outputs = [row for row in rows[len(original):] if isinstance(row.body, Output)]
        assert [row.body.finish for row in outputs] == ["continue", "complete"]
        refs = [next(part.value["reference"] for part in row.body.parts
                     if isinstance(part, ContentPart) and part.kind == "context.summary") for row in outputs]
        assert refs[0] == refs[1]
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            assert all(ctx.require(MODEL_CALLS)(identity)["state"] == "success" for identity in record.model_call_ids)
            calls = ctx.require(ServiceKey("fixture.calls"))
            assert len(calls) == 3
            assert all(all(f"old {role} {index}:" not in str(request.messages)
                           for role in ("input", "answer") for index in (0, 1)) for request in calls)
            assert all("current request" in str(request.messages) for request in calls[1:])
        assert (tmp_path / "effect.txt").read_text() == "once\n"
