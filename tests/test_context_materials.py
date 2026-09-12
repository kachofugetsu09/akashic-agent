from contextlib import asynccontextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore, lease_runtime_snapshot
from agent.plugin_composition.models import BoundChatModel, LLMResponse, ModelRequest
from plugins.context.api import ContextModel, MaterialData, Materials
from plugins.context.materials import ContextMaterials
from session.message import Message


class _UnreachedModel:
    @property
    def descriptor(self):
        raise AssertionError("closed MaterialView must not use the model")

    async def complete(self, request: ModelRequest) -> LLMResponse:
        raise AssertionError("closed MaterialView must not complete")

    def estimate_context_tokens(
        self, messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] = (),
    ) -> int:
        raise AssertionError("closed MaterialView must not estimate")

    def estimate_appended_message_tokens(
        self, messages: Sequence[Mapping[str, object]],
    ) -> int:
        raise AssertionError("closed MaterialView must not estimate")

    max_tool_schemas = None


class _UnreachedProjection:
    context_window = None
    max_tool_schemas = None

    def render(
        self, messages: tuple[Message, ...], *, after_seq: int,
        summary_reference: str | None = None, fresh: bool = False,
    ) -> ModelRequest:
        raise AssertionError("closed MaterialView must not render")

    def estimate(self, request: ModelRequest) -> int:
        raise AssertionError("closed MaterialView must not estimate")


_UNREACHED_MODEL: BoundChatModel = _UnreachedModel()
_UNREACHED_PROJECTION: ContextModel = _UnreachedProjection()


def _material(*, system_prompt: str = "", reminders=(), summary=None, references=()):
    return {
        "system_prompt": system_prompt,
        "reminders": tuple(reminders),
        "summary": summary,
        "references": tuple(references),
    }


def _reminder(name: str, text: str, priority: int):
    return {"name": name, "text": text, "priority": priority}


def _summary(reference: str, source_message_ids: tuple[str, ...], content: str):
    return {"reference": reference, "source_message_ids": source_message_ids, "content": content}


def _reference(ref: str, resolved_ref=None, retrieval_ref=None):
    return {"ref": ref, "resolved_ref": resolved_ref, "retrieval_ref": retrieval_ref}


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
        return _material(
            reminders=(_reminder("memory", profile.read_text(), 300),),
            references=(_reference("memory:1", retrieval_ref="retrieval:1"),),
        )
    async def persona(snapshot, source):
        calls.append("persona")
        return _material(system_prompt="fixed persona")
    async with catalog(prompt_sources={"persona": "trusted"}) as (ctx, service, evil):
        for wants_prompt in (False, True):
            with pytest.raises(PermissionError, match="实际插件"):
                await service.register(evil, name="persona", prepare=persona, prompt=wants_prompt)
        await service.register(ctx, name="memory", prepare=memory, priority=200)
        await service.register(ctx, name="persona", prepare=persona, prompt=True, priority=100)
        async with service.bind() as view:
            result = await view.prepare((), "conversation")
            assert result["system_prompt"] == "fixed persona"
            assert result["reminders"] == (_reminder("memory", "published profile", 300),)
            assert result["references"] == (_reference("memory:1", retrieval_ref="retrieval:1"),)
            assert calls == ["memory", "persona"]
        with pytest.raises(RuntimeError, match="关闭"):
            await view.prepare((), "conversation")


@pytest.mark.asyncio
async def test_material_provider_must_return_structural_mapping():
    async def legacy(snapshot, source) -> MaterialData:
        return cast(MaterialData, Materials(""))

    async with catalog() as (ctx, service, _):
        await service.register(ctx, name="legacy", prepare=legacy)
        async with service.bind() as view:
            with pytest.raises(TypeError, match="materials 必须是字符串键对象"):
                await view.prepare((), "conversation")


@pytest.mark.asyncio
async def test_program_excludes_retrieval_without_running_it_or_losing_persona():
    calls = []

    async def memory(snapshot, source):
        calls.append("memory")
        return _material(reminders=(_reminder("text", "retrieved private context", 300),))

    async def persona(snapshot, source):
        calls.append("persona")
        return _material(system_prompt="fixed persona")

    async with catalog(prompt_sources={"persona": "trusted"}) as (ctx, service, _):
        await service.register(ctx, name="persona", prepare=persona, prompt=True, priority=100)
        await service.register(ctx, name="memory", prepare=memory, priority=200)
        async with service.bind(exclude=frozenset({"memory"})) as view:
            result = await view.prepare((), "scheduler:job")
        assert result["system_prompt"] == "fixed persona"
        assert not result["reminders"]
        assert calls == ["persona"]
        # 显式排除不改变全局注册；普通回复仍能取得原有检索。
        async with service.bind() as view:
            result = await view.prepare((), "conversation")
        assert result["reminders"][0]["text"] == "retrieved private context"


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["prompt", "summary", "reference"])
async def test_materials_reject_unauthorized_prompt_and_conflicting_owners(conflict):
    async def first(snapshot, source):
        return _material(
            summary=_summary("summary:1", ("u1",), "one"),
            references=(_reference("ref", resolved_ref="first"),),
        )
    async def second(snapshot, source):
        return _material(
            system_prompt="forged" if conflict == "prompt" else "",
            summary=_summary("summary:2", ("u1",), "two") if conflict == "summary" else None,
            references=(_reference("ref", resolved_ref="second"),) if conflict == "reference" else (),
        )
    async with catalog(summary_source=("first", "trusted")) as (ctx, service, evil):
        with pytest.raises(PermissionError, match="配置"):
            await service.register(ctx, name="forged", prepare=first, prompt=True)
        await service.register(ctx, name="first", prepare=first)
        await service.register(ctx, name="second", prepare=second,
                               priority=200)
        with pytest.raises(PermissionError if conflict in {"prompt", "summary"} else ValueError):
            async with service.bind() as view:
                await view.prepare((), "conversation")


@pytest.mark.asyncio
async def test_only_summary_owner_can_reduce_and_closed_view_cannot_publish():
    import asyncio
    from agent.plugin_composition.models import ModelRequest

    entered, release = asyncio.Event(), asyncio.Event()
    async def prepare(snapshot, source):
        return _material(summary=_summary("old", ("u1",), "old summary"))
    async def reduce(snapshot, materials, request, model, projection, *, source, force):
        assert source == "conversation" and force
        entered.set()
        await release.wait()
        return _summary("new", ("u1", "a1"), "new summary")
    async with catalog(summary_source=("summary", "trusted")) as (ctx, service, evil):
        with pytest.raises(PermissionError, match="摘要"):
            await service.register(evil, name="retrieval", prepare=prepare, reduce=reduce)
        await service.register(ctx, name="summary", prepare=prepare, reduce=reduce)
        async with service.bind() as view:
            material = await view.prepare((), "conversation")
            operation = asyncio.create_task(view.reduce((), material, ModelRequest(messages=[]), _UNREACHED_MODEL, _UNREACHED_PROJECTION,
                                                       source="conversation", force=True))
            await entered.wait()
        release.set()
        with pytest.raises(RuntimeError, match="关闭"):
            await operation


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["none", "same", "new_ref_only", "changed_same_ref", "lost_source"])
async def test_reduction_preserves_durable_identity_and_recognizes_no_progress(case):
    from agent.plugin_composition.models import ModelRequest

    previous = _summary("published", ("u1", "a1"), "durable text")
    async def prepare(snapshot, source):
        return _material(summary=previous)
    async def reduce(snapshot, materials, request, model, projection, *, source, force):
        return {
            "none": None,
            "same": previous,
            "new_ref_only": _summary("new", previous["source_message_ids"], previous["content"]),
            "changed_same_ref": _summary(previous["reference"], previous["source_message_ids"], "changed text"),
            "lost_source": _summary("new", ("u1",), "changed text"),
        }[case]
    async with catalog(summary_source=("summary", "trusted")) as (ctx, service, evil):
        await service.register(ctx, name="summary", prepare=prepare, reduce=reduce)
        async with service.bind() as view:
            material = await view.prepare((), "conversation")
            if case in {"changed_same_ref", "lost_source"}:
                with pytest.raises(ValueError, match="不能"):
                    await view.reduce((), material, ModelRequest(messages=[]), _UNREACHED_MODEL, _UNREACHED_PROJECTION,
                                      source="conversation", force=True)
            else:
                result = await view.reduce((), material, ModelRequest(messages=[]), _UNREACHED_MODEL, _UNREACHED_PROJECTION,
                                            source="conversation", force=True)
                assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_reminder_order_uses_owner_and_name_and_keeps_each_request_snapshot(reverse):
    """安装顺序不影响同优先级块；下一次准备不改写上一份请求。"""
    current = "old"

    async def trusted(snapshot, source):
        return _material(reminders=(_reminder("z", "trusted-z", 200), _reminder("a", current, 200)))

    async def evil(snapshot, source):
        return _material(reminders=(_reminder("a", "evil-a", 200), _reminder("early", "early", 100)))

    async with catalog() as (ctx, service, other):
        registrations = [(ctx, "t", trusted), (other, "e", evil)]
        for owner, name, prepare in reversed(registrations) if reverse else registrations:
            await service.register(owner, name=name, prepare=prepare)
        async with service.bind() as view:
            before = await view.prepare((), "conversation")
            current = "new"
            after = await view.prepare((), "conversation")
        assert [item["text"] for item in before["reminders"]] == ["early", "evil-a", "old", "trusted-z"]
        assert [item["text"] for item in after["reminders"]] == ["early", "evil-a", "new", "trusted-z"]


@pytest.mark.asyncio
async def test_duplicate_reminder_identity_rejects_different_priorities():
    async def first(snapshot, source):
        return _material(reminders=(_reminder("same", "first", 100),))

    async def second(snapshot, source):
        return _material(reminders=(_reminder("same", "second", 200),))

    async with catalog() as (ctx, service, _):
        await service.register(ctx, name="one", prepare=first)
        await service.register(ctx, name="two", prepare=second)
        async with service.bind() as view:
            with pytest.raises(ValueError, match="身份重复"):
                await view.prepare((), "conversation")


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [-100, 300])
async def test_display_priority_cannot_move_a_write_before_a_failed_preparation(tmp_path, priority):
    """优先级变化只改输出排列，不能让原本被前置错误阻止的写入发生。"""
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    artifact = tmp_path / "prepared"

    async def fail(snapshot, source):
        entered.set()
        await release.wait()
        raise OSError("source read failed")

    async def write(snapshot, source):
        artifact.write_text("prepared")
        return _material()

    async with catalog() as (ctx, service, _):
        await service.register(ctx, name="a", prepare=fail, priority=100)
        await service.register(ctx, name="z", prepare=write, priority=priority)
        async with service.bind() as view:
            task = asyncio.create_task(view.prepare((), "conversation"))
            await entered.wait()
            assert not artifact.exists()
            release.set()
            with pytest.raises(OSError, match="source read failed"):
                await task
        assert not artifact.exists()


@pytest.mark.asyncio
async def test_system_priority_sorts_output_without_reordering_preparation():
    calls = []

    async def first(snapshot, source):
        calls.append("a")
        return _material(system_prompt="first")

    async def second(snapshot, source):
        calls.append("z")
        return _material(system_prompt="second")

    async with catalog(prompt_sources={"a": "trusted", "z": "trusted"}) as (ctx, service, _):
        await service.register(ctx, name="z", prepare=second, priority=-100, prompt=True)
        await service.register(ctx, name="a", prepare=first, priority=100, prompt=True)
        async with service.bind() as view:
            result = await view.prepare((), "conversation")
    assert calls == ["a", "z"]
    assert result["system_prompt"] == "second\n\nfirst"
