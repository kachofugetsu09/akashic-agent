import asyncio
from contextlib import asynccontextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.model import FiberState, ServiceKey
from agent.plugin_composition.models import BoundChatModel, LLMResponse, ModelRequest
from plugins.context.api import ContextModel, MaterialData, Materials
from plugins.context.materials import ContextMaterials, MATERIALS
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

    def key_recovery(self, request_key: str) -> str:
        raise AssertionError(f"closed MaterialView must not recover {request_key}")


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
    services = {}
    contexts = {}

    async def provider(ctx):
        service = ContextMaterials(
            ctx, prompt_sources=prompt_sources or {}, summary_source=summary_source,
        )
        services["materials"] = service
        await ctx.provide(MATERIALS, service, binding_contributors=service.binding_contributors)

    async def mounted(ctx):
        _ = ctx.require(MATERIALS)
        contexts[ctx.runtime.plugin_id] = ctx

    await root.mount(provider, name="materials-provider")
    for identity in ("trusted", "evil"):
        await root.mount(
            mounted,
            name=identity,
            inject=(MATERIALS,),
            runtime=PluginRuntime(
                identity, "generation", Path("/tmp"), Path("/tmp"), Path("/tmp"), {},
            ),
        )
    try:
        yield contexts["trusted"], services["materials"], contexts["evil"]
    finally:
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
        reminders = result["reminders"]
        assert isinstance(reminders, tuple)
        assert reminders[0]["text"] == "retrieved private context"


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
        before_reminders, after_reminders = before["reminders"], after["reminders"]
        assert isinstance(before_reminders, tuple) and isinstance(after_reminders, tuple)
        assert [item["text"] for item in before_reminders] == ["early", "evil-a", "old", "trusted-z"]
        assert [item["text"] for item in after_reminders] == ["early", "evil-a", "new", "trusted-z"]


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
    entered, release = asyncio.Event(), asyncio.Event()
    artifact = tmp_path / "prepared"

    async def fail(snapshot, source):
        async with ctx.runtime_scope():
            entered.set()
            await release.wait()
            raise OSError("source read failed")

    async def write(snapshot, source):
        artifact.write_text("prepared")
        return _material()

    async with catalog() as (ctx, service, _):
        await service.register(ctx, name="a", prepare=fail, priority=100)
        await service.register(ctx, name="z", prepare=write, priority=priority)
        async def prepare_in_work_task():
            async with service.bind() as view:
                await view.prepare((), "conversation")

        task = asyncio.create_task(prepare_in_work_task())
        await entered.wait()
        assert not artifact.exists()
        release.set()
        with pytest.raises(OSError, match="source read failed"):
            await task
        assert not artifact.exists()
        assert not ctx.fiber._fiber._in_flight_calls
        assert not service._ctx.fiber._fiber._in_flight_calls


@pytest.mark.asyncio
async def test_material_view_keeps_source_scope_during_unload_and_filters_next_bind():
    prepare_entered = asyncio.Event()
    consumer_released = asyncio.Event()
    cleanup_calls = 0
    consumer_cleanup_calls = 0
    scope_work = []
    unrelated_scope_work = []

    async with catalog(
        prompt_sources={"source": "trusted"},
        summary_source=("source", "trusted"),
    ) as (ctx, service, unrelated):
        root = ctx._root
        probe_key = ServiceKey("materials.drain.probe")
        await ctx.provide(probe_key, object())

        async def hard_consumer(consumer_ctx):
            nonlocal consumer_cleanup_calls
            _ = consumer_ctx.require(probe_key)

            def setup():
                def cleanup():
                    nonlocal consumer_cleanup_calls
                    consumer_cleanup_calls += 1
                    consumer_released.set()

                return cleanup

            await consumer_ctx.effect(setup, label="materials-drain-consumer")

        consumer_fiber = await root.mount(
            hard_consumer,
            name="materials-drain-consumer",
            inject=(probe_key,),
            runtime=PluginRuntime(
                "drain-consumer", "materials-drain", Path("/tmp"),
                Path("/tmp"), Path("/tmp"), {},
            ),
        )
        unrelated_state = unrelated.fiber.state
        unrelated_activation = unrelated.fiber.activation_token
        unrelated_events = []
        await unrelated.on(RUNTIME_STARTED, lambda _event: unrelated_events.append("started"))
        await unrelated.on(RUNTIME_STOPPING, lambda _event: unrelated_events.append("stopping"))
        unrelated_events_before = tuple(unrelated_events)

        def cleanup():
            nonlocal cleanup_calls
            cleanup_calls += 1

        await ctx.effect(lambda: cleanup, label="materials-source-resource")

        async def prepare(_snapshot, _source):
            async with ctx.runtime_scope():
                prepare_entered.set()
                await consumer_released.wait()
                scope_work.append(("prepare", ctx.fiber.state))
                return _material(
                    system_prompt="source",
                    summary=_summary("old", ("u1",), "old summary"),
                )

        async def secondary_prepare(_snapshot, _source):
            async with ctx.runtime_scope():
                scope_work.append(("secondary", ctx.fiber.state))
                return _material()

        async def reduce(_snapshot, _materials, _request, _model, _projection, *, source, force):
            assert source == "conversation" and force
            async with ctx.runtime_scope():
                scope_work.append(("reduce", ctx.fiber.state))
                return _summary("new", ("u1", "a1"), "new summary")

        await service.register(ctx, name="source", prepare=prepare, prompt=True, reduce=reduce)
        await service.register(ctx, name="secondary", prepare=secondary_prepare)

        async def dispose_after_prepare():
            await prepare_entered.wait()
            await ctx.fiber.dispose()

        dispose_task = asyncio.create_task(dispose_after_prepare())
        try:
            async with service.bind() as view:
                prepared = await view.prepare((), "conversation")
                assert ctx.fiber.state is FiberState.UNLOADING
                assert cleanup_calls == 0
                assert consumer_cleanup_calls == 1
                assert scope_work == [
                    ("secondary", FiberState.ACTIVE),
                    ("prepare", FiberState.UNLOADING),
                ]
                assert len(ctx.fiber._fiber._in_flight_calls) == 1

                async with unrelated.runtime_scope():
                    unrelated_scope_work.append(unrelated.fiber.state)
                assert unrelated_scope_work == [FiberState.ACTIVE]

                with pytest.raises(ValueError, match="Prompt"):
                    async with service.bind():
                        pytest.fail("required source must fail while unloading")
                async with service.bind(exclude=frozenset({"source"})) as excluded:
                    assert await excluded.prepare((), "scheduler:job") == _material()
                    assert len(ctx.fiber._fiber._in_flight_calls) == 1

                reduced = await view.reduce(
                    (), prepared, ModelRequest(messages=[]), _UNREACHED_MODEL,
                    _UNREACHED_PROJECTION, source="conversation", force=True,
                )
                assert reduced == _summary("new", ("u1", "a1"), "new summary")
                assert scope_work[-1] == ("reduce", FiberState.UNLOADING)
            await dispose_task
        finally:
            if not dispose_task.done():
                if not prepare_entered.is_set():
                    prepare_entered.set()
                consumer_released.set()
                await dispose_task

        assert ctx.fiber.state is FiberState.DISPOSED
        assert cleanup_calls == 1
        assert consumer_fiber.state is FiberState.PENDING
        assert not ctx.fiber._fiber._in_flight_calls
        assert unrelated_state is FiberState.ACTIVE
        assert unrelated.fiber.state is unrelated_state
        assert unrelated.fiber.activation_token is unrelated_activation
        assert tuple(unrelated_events) == unrelated_events_before


@pytest.mark.asyncio
async def test_material_bind_skips_loading_sources_and_keeps_required_failure_loud(tmp_path):
    root = CompositionRoot("materials-loading")
    service_holder = {}
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def provider(ctx):
        service = ContextMaterials(ctx, prompt_sources={"required": "loading"})
        service_holder["service"] = service
        await ctx.provide(MATERIALS, service, binding_contributors=service.binding_contributors)

    async def loading(ctx):
        service = ctx.require(MATERIALS)

        async def required(_snapshot, _source):
            calls.append("required")
            return _material(system_prompt="required")

        async def ordinary(_snapshot, _source):
            calls.append("ordinary")
            return _material(reminders=(_reminder("ordinary", "ordinary", 1),))

        await service.register(ctx, name="required", prepare=required, prompt=True)
        await service.register(ctx, name="ordinary", prepare=ordinary)

        async def on_started(_event):
            started.set()
            await release.wait()

        await ctx.on(RUNTIME_STARTED, on_started)

    await root.mount(provider, name="materials-provider")
    mount_task = asyncio.create_task(
        root.mount(
            loading,
            name="loading-source",
            inject=(MATERIALS,),
            runtime=PluginRuntime(
                "loading", "materials-loading", tmp_path, tmp_path, tmp_path, {},
            ),
        )
    )
    try:
        await started.wait()
        service = service_holder["service"]
        with pytest.raises(ValueError, match="Prompt"):
            async with service.bind():
                pytest.fail("required LOADING source must fail closed")
        async with service.bind(exclude=frozenset({"required"})) as view:
            assert await view.prepare((), "scheduler:job") == _material()
        assert calls == []

        release.set()
        loading_fiber = await mount_task
        assert loading_fiber.state is FiberState.ACTIVE
        async with service.bind() as view:
            result = await view.prepare((), "conversation")
        assert result["system_prompt"] == "required"
        assert result["reminders"] == (_reminder("ordinary", "ordinary", 1),)
        assert calls == ["ordinary", "required"]
    finally:
        release.set()
        if not mount_task.done():
            await mount_task
        await root.dispose()


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
