"""一张学习图不可用时，其他图仍独立消费真实 Message 日志。"""
import asyncio
from contextlib import closing
from pathlib import Path
import re
import shutil
import sqlite3

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS, CallSource
from agent.plugin_contracts.context import MATERIALS
from plugins.akasha.infrastructure.persistence import load_consumption
from plugins.akasha.scopes import ScopePolicies, graph_key, graph_path
from plugins.content.plugin import check_text
from session.log import SessionAttributes
from session.message import CallRef, ContentPart, Input, Output, ToolCall
from plugins.tools.plugin import open_tool
from tests.test_default_reply import application


def add_memory(sources: Path) -> None:
    """安装真实 Akasha；确定性 embedding provider 只控制外部模型边界。"""
    shutil.copytree(Path(__file__).parents[1] / "plugins/akasha", sources / "akasha",
                    ignore=shutil.ignore_patterns("__pycache__"))
    provider = sources / "test_provider/plugin.py"
    with provider.open("a") as stream:
        stream.write('''

original_apply = apply
async def apply(ctx):
    import asyncio
    from agent.plugin_composition import EMBEDDINGS
    from agent.plugin_composition.models import EmbeddingSpaceDescriptor, EmbeddingResult
    await original_apply(ctx)
    ready = asyncio.Event()
    class Embedding:
        descriptor = EmbeddingSpaceDescriptor(
            plugin_snapshot_id="fixture", model_revision=0, model="fixture",
            model_id="fixture", driver_id="fixture", driver_contract_version="1",
            connection_id="fixture", auth_identity="fixture", connection_fingerprint="fixture",
            dimensions=3, normalization="l2", capability_digest="fixture",
        )
        async def embed(self, texts):
            ready.set()
            return EmbeddingResult(tuple((1.0, 0.0, 0.0) for _ in texts))
    class Embeddings:
        def describe(self, *, model_id=None):
            return Embedding.descriptor
        @asynccontextmanager
        async def bind(self, *, model_id=None):
            yield Embedding()
    await ctx.provide(EMBEDDINGS, Embeddings())
    await ctx.provide(ServiceKey("fixture.embedding-ready"), ready)
''')


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_scope", [None, "broken"])
async def test_unavailable_graph_does_not_stop_other_graphs(tmp_path: Path, broken_scope: str | None) -> None:
    """O / MEM-013：故障图可见，健康图仍学习且不清除旁图错误。"""
    # 1. 原有图缺少消费出处，需要显式重建；不能自动重放或清空它。
    memory = tmp_path / "workspace/memory/akasha.db"
    broken_key = graph_key(()) if broken_scope is None else graph_key((("project", broken_scope),))
    broken_path = graph_path(memory, broken_key)
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(broken_path)) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.commit()
    original_graph = broken_path.read_bytes()
    async with application(tmp_path, replying=False, start=False, extra_sources=add_memory) as (log, host):
        policy = ScopePolicies(log.owner("plugin:akasha:scope-policy"), log.catalog())
        if broken_scope is not None:
            policy.set("project", broken_scope, "isolated")
            log.ensure_session("broken", SessionAttributes.scoped({"project": broken_scope}))
        policy.set("project", "healthy", "isolated")
        log.ensure_session("healthy", SessionAttributes.scoped({"project": "healthy"}))
        for session in ("broken", "healthy"):
            log.writer(session, author="user", source="conversation", body_types=(Input,),
                       content={"text": check_text}).append(
                session + "-input", Input((ContentPart("text", "keep this fact"),)))
        log.writer("healthy", author="assistant", source="conversation", body_types=(Output,),
                   content={"text": check_text}).append(
            "healthy-output", Output((ContentPart("text", "remembered fact"),), "complete"))
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        # 2. 由真实后台学习触发 embedding；材料读取的同图锁等待发布完成。
        await asyncio.wait_for(root.context.require(ServiceKey("fixture.embedding-ready")).wait(), 5)
        async with root.context.require(MATERIALS).bind() as materials:
            _ = await materials.prepare(log.reader("healthy").snapshot(), "conversation")
            failed = await materials.prepare(log.reader("broken").snapshot(), "conversation")
        state = load_consumption(graph_path(memory, graph_key((("project", "healthy"),))))
        assert state is not None
        assert [entry.ending[1] for entry in state.applied] == ["healthy-output"]
        assert broken_key in str(failed) and "重建" in str(failed)
        # 3. 健康图正常召回自己的已学问答，不能清除故障图的错误。
        log.writer("healthy", author="user", source="conversation", body_types=(Input,),
                   content={"text": check_text}).append(
            "healthy-next", Input((ContentPart("text", "what was that fact?"),)))
        async with root.context.require(MATERIALS).bind() as materials:
            recalled = await materials.prepare(log.reader("healthy").snapshot(), "conversation")
        assert broken_path.read_bytes() == original_graph
        assert "healthy-output" in str(recalled)
        assert any(broken_key in (item.reason or "") and "重建" in (item.reason or "")
                   for item in root.receipt().health)

        # 4. 显式反馈也必须拒绝已知故障图，不能接受之后无法学习的标记。
        bindings = root.context.require(BINDINGS)
        tools = root.context.require(TOOLS)
        view = root.context.require(ALL_TOOLS)()
        tool_ids = [await tools.bind_scoped(view.select(name), bindings)
                    for name in ("remember_memory", "forget_memory")]
        arguments = {"message_ids": ["current_user_message"]}
        log.writer("broken", author="assistant", source="conversation", body_types=(Output,),
                   content={"text": check_text}, check_call=lambda call: None).append(
            "feedback", Output(tuple(ToolCall(identity, arguments) for identity in tool_ids), "continue"))
        for index, identity in enumerate(tool_ids):
            async with open_tool(bindings, identity) as tool:
                with pytest.raises(RuntimeError, match=re.escape(broken_key) + ".*重建"):
                    await tool.prepare(arguments, CallSource(CallRef("feedback", index),
                                                            log.reader("broken").snapshot()))
