"""O: independent material owners prepare together and merge in stable order."""
import asyncio

import pytest

from agent.plugin_composition import CompositionRoot, Context, FiberHandle, FiberState, PluginRuntime
from plugins.context.contract import MATERIALS_V4
from plugins.context.materials import ContextMaterials
from plugins.context.api import decode_material
from plugins.models.settings import UpdateConnection
from agent.plugin_composition import tasks as task_runtime
from agent.plugin_composition.tasks import Tasks
from tests.support.material_models import material_models


@pytest.mark.asyncio
async def test_independent_material_owners_overlap_on_a_real_root(tmp_path):
    root = CompositionRoot("material-execution")
    services: list[ContextMaterials] = []
    entered, release, second = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def setup(ctx):
        services.append(ContextMaterials(ctx, prompt_sources={}))

    await root.mount(setup, name="materials")
    service = services[0]

    async def first(ctx):
        async def prepare(snapshot, source):
            entered.set()
            await release.wait()
            return {"reminders": ({"name": "a", "text": "A", "priority": 20},)}
        await service.register(ctx, name="a", prepare=prepare, kind="context")

    async def other(ctx):
        async def prepare(snapshot, source):
            second.set()
            return {"reminders": ({"name": "b", "text": "B", "priority": 10},)}
        await service.register(ctx, name="b", prepare=prepare, kind="context")

    for name, apply in (("a", first), ("b", other)):
        await root.mount(apply, name=name,
                         runtime=PluginRuntime(name, "materials", tmp_path, tmp_path, tmp_path, {}))
    try:
        async with service.bind() as view:
            job = asyncio.create_task(view.prepare((), "conversation"))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await asyncio.wait_for(second.wait(), 1)
            finally:
                release.set()
                result = await job
            assert [item.text for item in decode_material(result).reminders] == ["B", "A"]
    finally:
        release.set()
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_material_failure_and_cancellation_drain_real_owners(tmp_path, cancel):
    root = CompositionRoot("material-drain")
    services: list[ContextMaterials] = []
    entered, release = asyncio.Event(), asyncio.Event()
    drained = []
    first_error = ValueError("first owner failed")

    async def setup(ctx):
        services.append(ContextMaterials(ctx, prompt_sources={}))

    await root.mount(setup, name="materials")
    service = services[0]

    async def first(ctx):
        async def prepare(snapshot, source):
            try:
                if cancel:
                    await release.wait()
                raise first_error
            finally:
                drained.append("a")
        await service.register(ctx, name="a", prepare=prepare, kind="context")

    async def second(ctx):
        async def prepare(snapshot, source):
            entered.set()
            try:
                await release.wait()
                raise RuntimeError("second owner failed")
            finally:
                drained.append("b")
        await service.register(ctx, name="b", prepare=prepare, kind="context")

    for name, apply in (("a", first), ("b", second)):
        await root.mount(apply, name=name,
                         runtime=PluginRuntime(name, "materials", tmp_path, tmp_path, tmp_path, {}))
    try:
        async with service.bind() as view:
            job = asyncio.create_task(view.prepare((), "conversation"))
            await asyncio.wait_for(entered.wait(), 2)
            assert not job.done()
            if cancel:
                job.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await job
            else:
                release.set()
                with pytest.raises(ValueError) as caught:
                    await job
                assert caught.value is first_error
            assert sorted(drained) == ["a", "b"]
    finally:
        release.set()
        await root.dispose()


@pytest.mark.asyncio
async def test_material_models_keep_parent_revision_and_open_own_bindings(tmp_path):
    """A settings update cannot change models halfway through joined material work."""
    entered, release, sibling = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with material_models(tmp_path) as models:
        services = []
        parent_models = []
        seen = []

        async def setup(ctx):
            services.append(ContextMaterials(ctx, prompt_sources={}))

        await models.root.mount(setup, name="materials")
        service = services[0]

        async def first(ctx):
            async def embed():
                async with models.embeddings.bind() as model:
                    seen.append(model.descriptor.model_revision)
                    assert len((await model.embed(["material"])).vectors[0]) == 3

            async def prepare(snapshot, source):
                entered.set()
                await release.wait()
                await embed()
                # Nested joined children carry data, never the parent's driver.
                await asyncio.create_task(embed(), context=task_runtime.child_task_context())
                async with models.chat_models.execution() as execution:
                    model = execution.chat()
                    seen.append(model.descriptor.model_revision)
                    assert model is not parent_models[0]
                return {"reminders": ({"name": "a", "text": "A", "priority": 20},)}
            await service.register(ctx, name="a", prepare=prepare, kind="recall")

        async def other(ctx):
            async def prepare(snapshot, source):
                sibling.set()
                return {"reminders": ({"name": "b", "text": "B", "priority": 10},)}
            await service.register(ctx, name="b", prepare=prepare, kind="context")

        for name, apply in (("a", first), ("b", other)):
            await models.root.mount(apply, name=name,
                runtime=PluginRuntime(name, "materials", tmp_path, tmp_path, tmp_path, {}))

        async def run():
            async with models.chat_models.execution() as execution, service.bind() as view:
                parent_models.append(execution.chat())
                return await view.prepare((), "conversation")

        job = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.wait_for(sibling.wait(), 2)
            parent = parent_models[0].descriptor
            await models.settings.apply(UpdateConnection(
                expected_revision=parent.model_revision, connection_id=parent.connection_id,
                name="changed during materials", auth_identity=parent.auth_identity,
                endpoint="https://changed.invalid",
            ))
        finally:
            release.set()
        result = await asyncio.wait_for(job, 5)
        assert seen == [parent.model_revision] * 3
        assert [item.text for item in decode_material(result).reminders] == ["B", "A"]
        assert all(item.endpoint == "https://fixture.invalid" for item in models.driver.opened[-3:])
        assert models.driver.closed == len(models.driver.opened)


@pytest.mark.asyncio
async def test_raw_model_children_reject_borrowed_execution_and_independent_tasks_use_new_settings(tmp_path):
    """Joined material work does not weaken raw-child or independent-Task contracts."""
    async with material_models(tmp_path) as models:
        async def embed():
            async with models.embeddings.bind() as model:
                await model.embed(["independent"])
                return model.descriptor.model_revision

        async with models.chat_models.execution() as execution:
            parent = execution.chat().descriptor
            with pytest.raises(RuntimeError, match="不能由子 task 继承"):
                await asyncio.create_task(embed())
            changed = await models.settings.apply(UpdateConnection(
                expected_revision=parent.model_revision, connection_id=parent.connection_id,
                name="new independent work", auth_identity=parent.auth_identity,
                endpoint="https://changed.invalid",
            ))

            async def joined():
                tasks = Tasks()
                try:
                    task = await tasks.admit("embedding", lambda slot: slot.start(lambda _: embed()))
                    assert await task.join() == changed.revision
                    async with models.chat_models.independent_execution() as independent:
                        assert independent.chat().descriptor.model_revision == changed.revision
                finally:
                    await tasks.close()

            await asyncio.create_task(joined(), context=task_runtime.child_task_context())
            assert execution.chat().descriptor.model_revision == parent.model_revision
        assert models.driver.closed == len(models.driver.opened)


@pytest.mark.asyncio
async def test_cancelled_material_drains_real_model_binding(tmp_path):
    """Cancellation closes both the child's embedding and parent's chat connection."""
    release = asyncio.Event()
    async with material_models(tmp_path, release=release) as models:
        services = []

        async def setup(ctx):
            service = ContextMaterials(ctx, prompt_sources={})
            services.append(service)
            async def prepare(snapshot, source):
                async with models.embeddings.bind() as model:
                    await model.embed(["cancelled"])
                return {}
            await service.register(ctx, name="embedding", prepare=prepare, kind="recall")

        await models.root.mount(setup, name="materials",
            runtime=PluginRuntime("materials", "materials", tmp_path, tmp_path, tmp_path, {}))

        async def run():
            async with models.chat_models.execution(), services[0].bind() as view:
                return await view.prepare((), "conversation")

        job = asyncio.create_task(run())
        try:
            await asyncio.wait_for(models.driver.entered.wait(), 2)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job
            assert models.driver.closed == len(models.driver.opened) == 2
        finally:
            release.set()
            if not job.done():
                job.cancel()
            await asyncio.gather(job, return_exceptions=True)


@pytest.mark.asyncio
async def test_material_prepare_reuses_bound_scope_while_owner_is_unloading(tmp_path):
    """Bound material callers can keep preparing across steps while an owner drains."""
    root = CompositionRoot("material-unloading")
    services: list[ContextMaterials] = []
    pointer_contexts: list[Context] = []
    pointer_fibers: list[FiberHandle] = []

    async def setup(ctx: Context):
        service = ContextMaterials(ctx, prompt_sources={})
        services.append(service)
        await ctx.provide(MATERIALS_V4, service)

    async def watch_apply(ctx: Context):
        async def register_pointer(child: Context):
            pointer_contexts.append(child)

            async def prepare(snapshot, source):
                async with child.runtime_scope():
                    return {"reminders": ({"name": "pointer", "text": "P", "priority": 20},)}

            return await child.require(MATERIALS_V4).register(
                child, name="pointer", kind="context", prepare=prepare,
            )

        pointer_fibers.append(await ctx.inject((MATERIALS_V4,), register_pointer, name="pointer"))

    await root.mount(
        setup,
        name="context",
        runtime=PluginRuntime("context", "materials", tmp_path, tmp_path, tmp_path, {}),
    )
    watch_fiber = await root.mount(
        watch_apply,
        name="pointer",
        runtime=PluginRuntime("pointer", "materials", tmp_path, tmp_path, tmp_path, {}),
    )
    service = services[0]
    pointer_fiber = pointer_fibers[0]
    pointer_ctx = pointer_contexts[0]
    unload: asyncio.Task[None] | None = None
    try:
        async with service.bind() as view:
            first = await view.prepare((), "conversation")
            assert [item.text for item in decode_material(first).reminders] == ["P"]

            probe = pointer_ctx.capture_runtime_scope()
            try:
                unload = asyncio.create_task(watch_fiber.dispose())
                await asyncio.wait_for(probe.wait_admission_closed(), 2)
            finally:
                await probe.close()

            assert pointer_fiber.state is FiberState.UNLOADING
            assert not unload.done()

            second = await view.prepare((), "conversation")
            assert [item.text for item in decode_material(second).reminders] == ["P"]
            assert not unload.done()

        assert unload is not None
        await asyncio.wait_for(unload, 2)
        assert pointer_fiber.state is FiberState.DISPOSED
        assert watch_fiber.state is FiberState.DISPOSED
    finally:
        if unload is not None and not unload.done():
            await asyncio.gather(unload, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_cancelled_material_prepare_during_owner_unload_releases_captured_scopes(tmp_path):
    """Cancelling prepare while an owner is unloading still lets unload finish after bind exits."""
    root = CompositionRoot("material-unloading-cancel")
    services: list[ContextMaterials] = []
    pointer_contexts: list[Context] = []
    pointer_fibers: list[FiberHandle] = []
    slow_entered, slow_release = asyncio.Event(), asyncio.Event()

    async def setup(ctx: Context):
        service = ContextMaterials(ctx, prompt_sources={})
        services.append(service)
        await ctx.provide(MATERIALS_V4, service)

    async def watch_apply(ctx: Context):
        async def register_pointer(child: Context):
            pointer_contexts.append(child)

            async def prepare(snapshot, source):
                return {"reminders": ({"name": "pointer", "text": "P", "priority": 20},)}

            return await child.require(MATERIALS_V4).register(
                child, name="pointer", kind="context", prepare=prepare,
            )

        pointer_fibers.append(await ctx.inject((MATERIALS_V4,), register_pointer, name="pointer"))

    async def slow_apply(ctx: Context):
        async def prepare(snapshot, source):
            slow_entered.set()
            await slow_release.wait()
            return {}

        await ctx.require(MATERIALS_V4).register(ctx, name="slow", kind="recall", prepare=prepare)

    await root.mount(
        setup,
        name="context",
        runtime=PluginRuntime("context", "materials", tmp_path, tmp_path, tmp_path, {}),
    )
    watch_fiber = await root.mount(
        watch_apply,
        name="pointer",
        runtime=PluginRuntime("pointer", "materials", tmp_path, tmp_path, tmp_path, {}),
    )
    await root.mount(
        slow_apply,
        name="slow",
        inject=(MATERIALS_V4,),
        runtime=PluginRuntime("slow", "materials", tmp_path, tmp_path, tmp_path, {}),
    )

    service = services[0]
    pointer_fiber = pointer_fibers[0]
    pointer_ctx = pointer_contexts[0]
    unload: asyncio.Task[None] | None = None

    async def run_bound():
        nonlocal unload
        async with service.bind() as view:
            probe = pointer_ctx.capture_runtime_scope()
            try:
                unload = asyncio.create_task(watch_fiber.dispose())
                await asyncio.wait_for(probe.wait_admission_closed(), 2)
            finally:
                await probe.close()
            return await view.prepare((), "conversation")

    job = asyncio.create_task(run_bound())
    try:
        await asyncio.wait_for(slow_entered.wait(), 2)
        assert pointer_fiber.state is FiberState.UNLOADING
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert unload is not None
        await asyncio.wait_for(unload, 2)
        assert pointer_fiber.state is FiberState.DISPOSED
        assert watch_fiber.state is FiberState.DISPOSED
    finally:
        slow_release.set()
        if not job.done():
            job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        if unload is not None and not unload.done():
            await asyncio.gather(unload, return_exceptions=True)
        await root.dispose()
