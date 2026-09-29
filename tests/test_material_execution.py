"""O: independent material owners prepare together and merge in stable order."""
import asyncio

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from plugins.context.materials import ContextMaterials
from plugins.context.api import decode_material


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
