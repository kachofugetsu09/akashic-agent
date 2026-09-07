import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.content.plugin import CONTENT
from plugins.context.materials import MATERIALS
from plugins.tools.api import MessageReply
from plugins.tools.plugin import TOOLS
from agent.plugin_composition.bindings import BINDINGS
from session.log import MessageLog
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult


@asynccontextmanager
async def application(tmp_path):
    root = tmp_path / "plugins"
    for name in ("akasha", "turn_projection", "content", "context", "tools"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, root / name,
                       ignore=shutil.ignore_patterns("__pycache__"))
    # 候选使用真正的新入口；正式 manifest 留到完整插件功能验收后一次切换。
    (root / "akasha/akashic.plugin.toml").write_text('''
schema_version = 1
name = "akasha"
version = "4.0.0"
api_version = 3
entrypoint = "message_plugin.py"
''')
    provider = root / "fixture_embeddings"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from agent.plugin_composition import EMBEDDINGS, ServiceKey
from agent.plugin_composition.models import EmbeddingSpaceDescriptor, EmbeddingResult
api_version = 3
name = "fixture_embeddings"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    embedded = asyncio.Event()
    descriptor = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="fixture", model_revision=0, model_id="fixture", connection_id="fixture",
        driver_id="fixture", driver_contract_version="1", auth_identity="fixture",
        connection_fingerprint="fixture", model="fixture", dimensions=2,
        normalization="unit", capability_digest="fixture")
    class Model:
        async def embed(self, texts):
            with Path(LOG_PATH).open("a") as handle:
                handle.write(repr(list(texts)) + "\\n")
            if "learned answer" in texts:
                embedded.set()
            return EmbeddingResult(tuple((0.6, 0.8) for text in texts))
    model = Model()
    model.descriptor = descriptor
    class Embeddings:
        def describe(self, *, model_id=None):
            return descriptor
        @asynccontextmanager
        async def bind(self, *, model_id=None):
            yield model
    await ctx.provide(EMBEDDINGS, Embeddings())
    await ctx.provide(ServiceKey("fixture.embedded"), embedded)
'''.replace("LOG_PATH", repr(str(tmp_path / "embedding-calls.txt"))))
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([root], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        await host._start_current_runtime_snapshot()
        yield log, host
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_actual_plugin_learns_provides_materials_and_runs_archived_recall_tool(tmp_path):
    async with application(tmp_path) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            bindings = ctx.require(BINDINGS)
            tools = ctx.require(TOOLS)
            identity = tools.bind("recall_memory", bindings)
            async with ctx.require(CONTENT).bind() as content:
                inputs = log.writer("s", author="user", source="conversation", body_types=(Input,), content=content.checks)
                outputs = log.writer("s", author="assistant", source="conversation", body_types=(Output,),
                                     content=content.checks, check_call=lambda call: None)
                inputs.append("u", Input((ContentPart("text", "original memory"),)))
                outputs.append("a", Output((ContentPart("text", "learned answer"),), "complete"))
                await asyncio.wait_for(ctx.require(ServiceKey("fixture.embedded")).wait(), 10)
                inputs.append("q", Input((ContentPart("text", "remember the detail"),)))
                async with ctx.require(MATERIALS).bind() as materials:
                    material = await materials.prepare(log.reader("s").snapshot(), "conversation")
                assert [ref.ref for ref in material.references] == ["u", "a"]
                read_recall = ctx.require(ServiceKey("akasha.recalls.v1"))
                observed = read_recall(material.references[0].retrieval_ref)
                assert observed.graph_version == 1
                outputs.append("request", Output((ToolCall(identity, {"query": "original memory"}),), "continue"))
                ref = CallRef("request", 0)
                reply = MessageReply("result", ref, log.reader("s"), log.writer(
                    "s", author="tool", source="conversation", body_types=(ToolResult,), call_ref=ref,
                    content=content.checks), lambda: None)
                async def authorize(binding, arguments):
                    return {"allowed": True}
                execution = tools.execution(authorize)
                result = await execution.execute_call(reply)
                assert result.outcome == "success"
                marker = result.parts[-1]
                assert marker.kind == "akasha.recall"
                recalled = read_recall(marker.value["retrieval_ref"])
                assert recalled.source.session_id == "s"
                assert recalled.source.call_ref == ref
                assert recalled.graph_version == 1
                before = (tmp_path / "embedding-calls.txt").read_text()
                assert await execution.execute_call(reply) == result
                assert (tmp_path / "embedding-calls.txt").read_text() == before
                assert len([message for message in log.reader("s").snapshot()
                            if isinstance(message.body, ToolResult)]) == 1
                async with ctx.require(MATERIALS).bind() as materials:
                    after_tool = await materials.prepare(log.reader("s").snapshot(), "conversation")
                assert [reference.ref for reference in after_tool.references] == ["u", "a"]
                assert {reference.retrieval_ref for reference in after_tool.references} == {marker.value["retrieval_ref"]}
                # 同 owner 的另一条调用也不能借用先前 CallRef 的查询事实。
                outputs.append("forged-request", Output((ToolCall(identity, {"query": "another query"}),), "continue"))
                forged_ref = CallRef("forged-request", 0)
                log.writer("s", author="tool", source="conversation", body_types=(ToolResult,),
                           call_ref=forged_ref, content=content.checks).append(
                    "forged-result", ToolResult(forged_ref, "success", (marker,)))
                async with ctx.require(MATERIALS).bind() as materials:
                    with pytest.raises(ValueError, match="不属于实际工具调用"):
                        await materials.prepare(log.reader("s").snapshot(), "conversation")


@pytest.mark.asyncio
async def test_prepared_recall_survives_config_change_and_source_removal(tmp_path):
    from agent.plugin_composition.bindings import Bindings
    from plugins.tools.plugin import open_tool

    async with application(tmp_path) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            bindings = ctx.require(BINDINGS)
            identity = ctx.require(TOOLS).bind("recall_memory", bindings)
            config_path = snapshot.generations["akasha"].data_dir / "config.local.toml"
            async with ctx.require(CONTENT).bind() as content:
                inputs = log.writer("s", author="user", source="conversation", body_types=(Input,),
                                    content=content.checks)
                outputs = log.writer("s", author="assistant", source="conversation", body_types=(Output,),
                                     content=content.checks, check_call=lambda call: None)
                inputs.append("u", Input((ContentPart("text", "original memory"),)))
                outputs.append("a", Output((ContentPart("text", "learned answer"),), "complete"))
                await asyncio.wait_for(ctx.require(ServiceKey("fixture.embedded")).wait(), 10)
                inputs.append("q", Input((ContentPart("text", "recall it"),)))
                async with ctx.require(MATERIALS).bind() as materials:
                    result = await materials.prepare(log.reader("s").snapshot(), "conversation")
                assert [reference.ref for reference in result.references] == ["u", "a"]
                async with open_tool(bindings, identity) as tool:
                    prepared = await tool.prepare({"query": "original memory"})

    # 重启前改变可变配置并移除源码；归档闭包仍须使用原配置、原图与原预算。
    config_path.write_text('db_path = "other.db"\ninject_max_chars = 1\n')
    shutil.rmtree(tmp_path / "plugins")
    restored_log = MessageLog(tmp_path / "sessions.db")
    restored_host = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                                  installed_cache_root=tmp_path / "home", message_log=restored_log)
    restored_bindings = Bindings(restored_log, restored_host._archive, restored_host.open_binding)
    try:
        async with open_tool(restored_bindings, identity) as tool:
            result = await tool.invoke("restored", prepared)
            assert result.outcome == "success"
            assert "learned answer" in str(result.parts)
            before = (tmp_path / "embedding-calls.txt").read_text()
            assert await tool.query("restored") == result
            assert (tmp_path / "embedding-calls.txt").read_text() == before
        assert not (tmp_path / "workspace/memory/other.db").exists()
    finally:
        await restored_host.terminate_all()
        restored_log.close()


@pytest.mark.asyncio
async def test_inspector_reads_actual_queries_through_the_mobile_provider(tmp_path):
    from agent.plugins.mobile_ui import PluginMobileUiProvider

    async with application(tmp_path) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            revision = snapshot.generations["akasha"].source_revision
            async with ctx.require(CONTENT).bind() as content:
                log.writer("s", author="user", source="conversation", body_types=(Input,),
                           content=content.checks).append("q", Input((ContentPart("text", "remember it"),)))
                async with ctx.require(MATERIALS).bind() as materials:
                    await materials.prepare(log.reader("s").snapshot(), "conversation")
        provider = PluginMobileUiProvider(host)
        try:
            before = (tmp_path / "embedding-calls.txt").read_text()
            listing = await provider.query("akasha", revision, "inspector.recent", {},
                                           session_id=None, turn_id=None)
            assert listing["total"] == 1
            query = listing["items"][0]
            detail = await provider.query("akasha", revision, "inspector.detail", {"query_id": query["query_id"]},
                                          session_id=None, turn_id=None)
            assert detail["graph_version"] == 0
            assert detail["hits"] == []
            assert detail["presented_count"] == 0
            assert detail["source"] == {"kind": "context", "session_id": "s", "source": "conversation", "through_seq": 0}
            assert (tmp_path / "embedding-calls.txt").read_text() == before
        finally:
            provider._executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_mobile_inspector_bounds_long_messages_without_dropping_hit_members(tmp_path):
    from agent.plugins.mobile_ui import PluginMobileUiProvider

    async with application(tmp_path) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            revision = snapshot.generations["akasha"].source_revision
            async with ctx.require(CONTENT).bind() as content:
                inputs = log.writer("s", author="user", source="conversation", body_types=(Input,), content=content.checks)
                outputs = log.writer("s", author="assistant", source="conversation", body_types=(Output,), content=content.checks)
                inputs.append("long", Input((ContentPart("text", "长" * (193 * 1024)),)))
                inputs.append("correction", Input((ContentPart("text", "keep this correction"),)))
                outputs.append("answer", Output((ContentPart("text", "learned answer"),), "complete"))
                await asyncio.wait_for(ctx.require(ServiceKey("fixture.embedded")).wait(), 10)
                inputs.append("query", Input((ContentPart("text", "recall it"),)))
                async with ctx.require(MATERIALS).bind() as materials:
                    prepared = await materials.prepare(log.reader("s").snapshot(), "conversation")
                identity = prepared.references[0].retrieval_ref
        provider = PluginMobileUiProvider(host)
        try:
            detail = await provider.query("akasha", revision, "inspector.detail", {"query_id": identity},
                                          session_id=None, turn_id=None)
            assert detail["schema"] == "akasha.queries.v1"
            messages = detail["hits"][0]["messages"]
            assert [message["message_id"] for message in messages] == ["long", "correction", "answer"]
            assert messages[0]["preview"] == "长" * 240
            assert [message["truncated"] for message in messages] == [True, False, False]
            assert [message["presented"] for message in messages] == [False, True, True]
            assert len(log.reader("s").get("long").body.parts[0].value) == 193 * 1024
        finally:
            provider._executor.shutdown(wait=True)
