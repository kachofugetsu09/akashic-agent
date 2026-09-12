from plugins.context.api import check_summary as _model_summary_check
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import shutil
from collections.abc import Mapping
from typing import cast

import pytest

from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog
from session.message import Control, Input, Output, ToolResult


@asynccontextmanager
async def application(tmp_path, *, replying, start=True, missing_tool=False, discovery=False, compaction=False,
                      output_tokens=4096, keep_recent_tokens=128, summary_padding=0, provider_effect_data=False,
                      updates=False, validation_passed=True, extra_sources=None):
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    for name in (
        "sources",
        "content",
        "context",
        "tools",
        "conversation",
        "react",
        "turn_projection",
        "reply_program",
        *(("reply", "tool_search") if replying else ()),
    ):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    if updates:
        from tests.test_delivery_bindings import sources as delivery_sources
        delivery_sources(sources)
        shutil.copytree(Path(__file__).parents[1] / "plugins/plugin_update", sources / "plugin_update",
                        ignore=shutil.ignore_patterns("__pycache__"))
    if compaction:
        shutil.copytree(Path(__file__).parents[1] / "plugins/compaction", sources / "compaction",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (sources / "compaction/akashic.plugin.toml").write_text(
            'schema_version = 1\nname = "compaction"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')
        settings = tmp_path / "workspace/plugin-data/context-builtin/config.local.toml"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text('summary_source = ["compaction", "compaction"]\n')
        module = sources / "compaction/message_plugin.py"
        module.write_text(module.read_text().replace('Field(default=20_000,', f'Field(default={keep_recent_tokens},'))
        reply = sources / 'reply/plugin.py'
        reply.write_text(reply.read_text().replace('Field(default=4096,', f'Field(default={output_tokens},'))
    if missing_tool:
        settings = tmp_path / "workspace/plugin-data/reply-builtin/config.local.toml"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text('tools = ["gone"]\n')
    provider = sources / "test_provider"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from types import SimpleNamespace
from pathlib import Path
from agent.plugin_composition import CHAT_MODELS, ServiceKey
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ModelRole, ToolCall
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.tools.api import Result
from plugins.standard_tools.shell import shell_cleanup
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
    await ctx.require(TOOLS).declare_group(ctx, always_on=True)
    await ctx.require(TOOLS).register(ctx, name="write_evidence", description="record local test evidence",
        parameters={"type":"object"}, open=open)
    await ctx.provide(ServiceKey("tools.cleanup.v1"), shell_cleanup)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
    await ctx.provide(ServiceKey("fixture.calls"), calls)
'''.replace("EFFECT_PATH", repr(str(tmp_path / "effect.txt"))))
    if provider_effect_data:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace(
            repr(str(tmp_path / "effect.txt")), 'ctx.runtime.data_dir / "effect.txt"'))
    if updates:
        import json
        module = provider / "plugin.py"
        verdict = json.dumps({"passed": validation_passed, "reason": "tool evidence checked"})
        module.write_text(module.read_text().replace('return LLMResponse("finished")', f'return LLMResponse({verdict!r})'))
    if discovery:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace(
            'if len(calls) == 1:',
            'if len(calls) == 1:\n                return LLMResponse(None, [ToolCall("search-call", "tool_search", {"query": "write_evidence"})])\n            if len(calls) == 2:').replace('declare_group(ctx, always_on=True)', 'declare_group(ctx, description="Write local evidence")').replace(
            'ToolCall("provider-call", "write_evidence", {})',
            'ToolCall("provider-call", "tool_call", {"name": "write_evidence", "arguments": {}})'))
    if compaction:
        module = provider / "plugin.py"
        module.write_text(module.read_text().replace('calls = []', 'calls = []\n    business = []').replace(
            'return 10', 'return len(str(messages)) // 4').replace('if len(calls) == 1:', '''if "[Source messages]" in str(request.messages):
                from plugins.compaction.message_summary import HEADINGS
                return LLMResponse("\\n".join(heading + "\\nPreserved facts." for heading in HEADINGS))
            business.append(request)
            if len(business) == 1:''').replace("Preserved facts.", "Preserved facts." + "z" * summary_padding))
    if extra_sources is not None:
        extra_sources(sources)
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore

    log = MessageLog(tmp_path / "sessions.db")
    artifact_store = ArtifactStore(tmp_path / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(
        workspace=workspace, metadata_store=artifact_store
    )
    host = PluginManager(
        [sources],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home/cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    try:
        await host.load_all()
        if start:
            await host.start_runtime()
        yield log, host
    finally:
        await host.terminate_all()
        log.close()
        artifact_store.close()


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
            assert rows is not None
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
    async with application(tmp_path, replying=True, start=False, missing_tool=True) as (
        log,
        host,
    ):
        assert host.generation("reply") is None
        gate = host.latest_gate("reply")
        assert gate is not None and gate.status == "failed"
        assert gate.failure_reason == "tools: Extra inputs are not permitted"
        assert log.catalog().snapshot_heads() == {}


@pytest.mark.asyncio
async def test_reply_commits_plugin_metadata_and_history_reads_it_without_the_plugin(tmp_path):
    """实际插件注册、模型与工具循环、writer 授权和重启读取共用一份附加信息。"""
    from contextlib import closing
    from infra.channels.message_view import message_rows
    from session.message_codec import json_value

    def extra(sources):
        plugin = sources / "citation"
        plugin.mkdir()
        (plugin / "plugin.py").write_text('''
from plugins.content.plugin import CONTENT
api_version = 3
name = "citation"
version = "1.0.0"
inject = (CONTENT,)
async def apply(ctx, config):
    async def decode(source, references):
        return (), {"version": 1, "references": [{"ref": "remembered", "declared": True}]} if source.text else {}
    await ctx.require(CONTENT).register(ctx, {
        "name": "citation", "prompt": "", "content": {}, "decode": decode})
''')

    async with application(tmp_path, replying=True, extra_sources=extra) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "s", "u", ChannelInboundMessage("test", "user", "s", "record evidence",
                                                 datetime(2026, 9, 5, tzinfo=UTC), {}))
        async def completed():
            async for _ in log.catalog().follow():
                for row in log.reader("s").snapshot():
                    if isinstance(row.body, Output) and row.body.finish == "complete":
                        return row
        output = await asyncio.wait_for(completed(), 5)
        assert json_value(output.metadata) == {"citation": {"version": 1, "references": [{"ref": "remembered", "declared": True}]}}
        assert all(part.kind != "citation" for part in output.body.parts)
    shutil.rmtree(tmp_path / "plugins/citation")
    with closing(MessageLog(tmp_path / "sessions.db")) as restarted:
        assert restarted.reader("s").get(output.message_id) == output
        rows = message_rows(restarted.reader("s").read_page())
        assert rows[-1]["metadata"] == {"citation": {"version": 1, "references": [{"ref": "remembered", "declared": True}]}}


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
        assert rows is not None
        assert [type(row.body) for row in rows] == [
            Input,
            Output,
            ToolResult,
            Output,
            ToolResult,
            Output,
        ]
        assert rows[2].body.parts[-1].kind == "text"
        assert (tmp_path / "effect.txt").read_text() == "once\n"
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(
                ServiceKey("fixture.calls")
            )
            expected = {"tool_search", "tool_call"}
            assert {tool["function"]["name"] for tool in calls[0].tools} == expected
            assert {tool["function"]["name"] for tool in calls[1].tools} == expected
            import json
            from agent.plugin_composition import CHAT_MODELS
            from agent.plugin_composition.bindings import BINDINGS
            from agent.plugin_composition.models import ModelRole
            from plugins.models.content import render_content
            from plugins.models.projection import MODEL_CALLS, MessageProjection
            from plugins.tools.plugin import TOOLS
            payload = json.loads(cast(str, rows[2].body.parts[0].value))
            assert payload["matched_groups"][0]["tools"][0]["function"]["name"] == "write_evidence"
            assert "matched_groups" in str(calls[1].messages)
            ctx = snapshot.composition_root.context
            # 新投影从持久日志重建；摘要覆盖搜索结果时，只有请求视图失去 schema。
            async with ctx.require(CHAT_MODELS).execution() as execution:
                model = execution.chat(ModelRole.AGENT)
                bindings = ctx.require(BINDINGS)
                def tool_name(binding):
                    return cast(str, cast(Mapping[str, object], bindings.describe(binding, TOOLS)["tool"])["name"])
                projection = MessageProjection(model, check_summary=_model_summary_check, source="conversation", render_content=lambda part: render_content(part, artifacts={}),
                                               tool_name=tool_name, read_call=ctx.require(MODEL_CALLS))
                before = log.reader("s").snapshot()
                retained = projection.render(before, after_seq=-1)
                compacted = projection.render(before, after_seq=rows[2].seq)
                assert "matched_groups" in str(retained.messages)
                assert "matched_groups" not in str(compacted.messages)
                assert log.reader("s").snapshot() == before


@pytest.mark.asyncio
async def test_default_reply_applies_provider_tool_capacity_before_first_request(tmp_path):
    def constrain_provider(sources):
        module = sources / "test_provider/plugin.py"
        source = module.read_text()
        assert source.count("max_tool_schemas = None") == 1
        assert source.count('parameters={"type":"object"}, open=open)') == 1
        module.write_text(
            source.replace("max_tool_schemas = None", "max_tool_schemas = 1")
        )

    from agent.plugin_composition import ServiceKey
    async with application(
        tmp_path, replying=True, discovery=True, extra_sources=constrain_provider
    ) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "s", "u", ChannelInboundMessage("test", "user", "s", "record evidence",
                                                 datetime(2026, 9, 5, tzinfo=UTC), {}))

        async def failed():
            async for _ in log.catalog().follow():
                rows = log.reader("s").snapshot()
                if isinstance(rows[-1].body, Control) and rows[-1].body.action == "failure":
                    return rows

        rows = await asyncio.wait_for(failed(), 5)
        assert rows is not None
        assert isinstance(rows[-1].body, Control)
        assert "容量不足" in (rows[-1].body.reason or "")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
            assert calls == []


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
        await host.start_runtime()
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
            reason = rows[-1].body.reason
            assert reason is not None
            if large_summary:
                assert "摘要后的完整请求仍超过" in reason
            elif soft_only:
                assert "近期原文保留量内没有合法摘要切点" in reason
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
        refs = [next(cast(Mapping[str, object], part.value)["reference"] for part in row.body.parts
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


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_call", [
    'ToolCall("bad-call", "tool_call", {"name": "write_evidence", "arguments": "{}"})',
    'ToolCall("bad-call", "tool_call", {"name": "uninstalled_tool", "arguments": {}})',
    'ToolCall("bad-call", "old_direct_tool", {})',
])
async def test_reply_recovers_rejected_protocol_without_creating_tool_effect(tmp_path, bad_call):
    """真实 Reply 反馈格式或过期名称错误，修正后只执行有效调用并可重放。"""
    from agent.plugin_composition import ServiceKey
    from session.message import ContentPart, ToolCall

    def extra(sources):
        provider = sources / "test_provider/plugin.py"
        code = provider.read_text().replace(
            'return LLMResponse(None, [ToolCall("provider-call", "write_evidence", {})])',
            f'return LLMResponse(None, [{bad_call}])\n'
            '            if len(calls) == 2:\n'
            '                return LLMResponse(None, [ToolCall("good-call", "tool_call", {"name": "write_evidence", "arguments": {}})])',
        ).replace('declare_group(ctx, always_on=True)', 'declare_group(ctx, description="Write local evidence")')
        provider.write_text(code)

    async with application(tmp_path, replying=True, extra_sources=extra) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            accept = ctx.require(CHANNEL_INPUT)
            calls = ctx.require(ServiceKey("fixture.calls"))
            await accept("test:room", "bad-input", ChannelInboundMessage(
                "test", "user", "room", "do the work", datetime.now(UTC), {},
            ))
        async def completed(count):
            async for _ in log.catalog().follow():
                rows = log.reader("test:room").snapshot()
                if any(isinstance(row.body, Control) and row.body.action == "failure" for row in rows):
                    pytest.fail("模型协议错误终止了回复")
                if sum(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows) == count:
                    return rows
        rows = await asyncio.wait_for(completed(1), 5)
        assert rows is not None
        assert (tmp_path / "effect.txt").read_text() == "once\n"
        assert len(calls) == 3
        assert [type(row.body) for row in rows] == [Input, Output, Output, ToolResult, Output]
        rejected = rows[1].body
        assert isinstance(rejected, Output) and rejected.finish == "continue"
        assert not any(isinstance(part, ToolCall) for part in rejected.parts)
        assert any(isinstance(part, ContentPart) and part.kind == "model.tool_rejection" for part in rejected.parts)
        for request in calls[1:]:
            rejection = [row for row in request.messages if row.get("tool_call_id") == "bad-call"]
            assert len(rejection) == 1 and "调用未执行" in str(rejection[0]["content"])
            system = [row for row in request.messages if row["role"] == "system"]
            assert "test_provider：Write local evidence" in str(system)
            assert all("可搜索工具目录" not in str(row) for row in request.messages if row["role"] != "system")
        before = rows
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "test:room", "follow-up", ChannelInboundMessage(
                    "test", "user", "room", "continue", datetime.now(UTC), {},
                ),
            )
        await asyncio.wait_for(completed(2), 5)
        assert log.reader("test:room").snapshot()[:len(before)] == before
        assert any(row.get("tool_call_id") == "bad-call" for row in calls[-1].messages)
        assert (tmp_path / "effect.txt").read_text() == "once\n"
