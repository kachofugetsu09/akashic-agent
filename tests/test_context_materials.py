from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore, lease_runtime_snapshot
from plugins.content.api import Reference
from plugins.context.api import Materials, Summary
from plugins.context.materials import ContextMaterials
from session.message import ContentPart


@asynccontextmanager
async def catalog(*, prompt_sources=None, summary_source=None):
    root = CompositionRoot("materials")
    service = ContextMaterials(root.context, prompt_sources=prompt_sources or {}, summary_source=summary_source)
    contexts = {}
    async def mounted(ctx):
        contexts[ctx.runtime.plugin_id] = ctx
    for identity in ("trusted", "evil"):
        await root.mount(mounted, name=identity, runtime=PluginRuntime(
            identity, "generation", Path("/tmp"), Path("/tmp"), Path("/tmp"), {},
        ))
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    try:
        async with lease_runtime_snapshot(store):
            yield contexts["trusted"], service, contexts["evil"]
    finally:
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_materials_fix_explicit_order_and_keep_retrieval_evidence_out_of_prompt(tmp_path):
    profile = tmp_path / "profile.md"
    profile.write_text("published profile")
    calls = []
    async def memory(snapshot, source):
        calls.append("memory")
        return Materials("", (ContentPart("memory", profile.read_text()),),
                         references=(Reference("memory:1", retrieval_ref="retrieval:1"),))
    async def persona(snapshot, source):
        calls.append("persona")
        return Materials("fixed persona")
    async with catalog(prompt_sources={"persona": "trusted"}) as (ctx, service, evil):
        for wants_prompt in (False, True):
            with pytest.raises(PermissionError, match="实际插件"):
                await service.register(evil, name="persona", prepare=persona, prompt=wants_prompt)
        await service.register(ctx, name="memory", prepare=memory, after=("persona",))
        await service.register(ctx, name="persona", prepare=persona, prompt=True)
        async with service.bind() as view:
            result = await view.prepare((), "conversation")
            assert result.system_prompt == "fixed persona"
            assert result.context == (ContentPart("memory", "published profile"),)
            assert result.references == (Reference("memory:1", retrieval_ref="retrieval:1"),)
            assert calls == ["persona", "memory"]
        with pytest.raises(RuntimeError, match="关闭"):
            await view.prepare((), "conversation")


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["prompt", "summary", "reference", "dependency"])
async def test_materials_reject_unauthorized_prompt_and_conflicting_owners(conflict):
    async def first(snapshot, source):
        return Materials("", summary=Summary("summary:1", ("u1",), "one"),
                         references=(Reference("ref", resolved_ref="first"),))
    async def second(snapshot, source):
        return Materials(
            "forged" if conflict == "prompt" else "",
            summary=Summary("summary:2", ("u1",), "two") if conflict == "summary" else None,
            references=(Reference("ref", resolved_ref="second"),) if conflict == "reference" else (),
        )
    async with catalog(summary_source=("first", "trusted")) as (ctx, service, evil):
        with pytest.raises(PermissionError, match="配置"):
            await service.register(ctx, name="forged", prepare=first, prompt=True)
        await service.register(ctx, name="first", prepare=first)
        await service.register(ctx, name="second", prepare=second,
                               after=("missing",) if conflict == "dependency" else ("first",))
        with pytest.raises(PermissionError if conflict in {"prompt", "summary"} else ValueError):
            async with service.bind() as view:
                await view.prepare((), "conversation")


@pytest.mark.asyncio
async def test_only_summary_owner_can_reduce_and_closed_view_cannot_publish():
    import asyncio
    from agent.plugin_composition.models import ModelRequest

    entered, release = asyncio.Event(), asyncio.Event()
    async def prepare(snapshot, source):
        return Materials("", summary=Summary("old", ("u1",), "old summary"))
    async def reduce(snapshot, materials, request, model, projection, *, source, force):
        assert source == "conversation" and force
        entered.set()
        await release.wait()
        return Summary("new", ("u1", "a1"), "new summary")
    async with catalog(summary_source=("summary", "trusted")) as (ctx, service, evil):
        with pytest.raises(PermissionError, match="摘要"):
            await service.register(evil, name="retrieval", prepare=prepare, reduce=reduce)
        await service.register(ctx, name="summary", prepare=prepare, reduce=reduce)
        async with service.bind() as view:
            material = await view.prepare((), "conversation")
            operation = asyncio.create_task(view.reduce((), material, ModelRequest(messages=[]), None, None,
                                                       source="conversation", force=True))
            await entered.wait()
        release.set()
        with pytest.raises(RuntimeError, match="关闭"):
            await operation


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["none", "same", "new_ref_only", "changed_same_ref", "lost_source"])
async def test_reduction_preserves_durable_identity_and_recognizes_no_progress(case):
    from agent.plugin_composition.models import ModelRequest

    previous = Summary("published", ("u1", "a1"), "durable text")
    async def prepare(snapshot, source):
        return Materials("", summary=previous)
    async def reduce(snapshot, materials, request, model, projection, *, source, force):
        return {
            "none": None,
            "same": previous,
            "new_ref_only": Summary("new", previous.source_message_ids, previous.content),
            "changed_same_ref": Summary(previous.reference, previous.source_message_ids, "changed text"),
            "lost_source": Summary("new", ("u1",), "changed text"),
        }[case]
    async with catalog(summary_source=("summary", "trusted")) as (ctx, service, evil):
        await service.register(ctx, name="summary", prepare=prepare, reduce=reduce)
        async with service.bind() as view:
            material = await view.prepare((), "conversation")
            if case in {"changed_same_ref", "lost_source"}:
                with pytest.raises(ValueError, match="不能"):
                    await view.reduce((), material, ModelRequest(messages=[]), None, None,
                                      source="conversation", force=True)
            else:
                assert await view.reduce((), material, ModelRequest(messages=[]), None, None,
                                         source="conversation", force=True) is previous
