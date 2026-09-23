"""The live Root grants service access through exact owner calls."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition import CompositionError, CompositionRoot, PluginRuntime, ServiceKey
from agent.plugin_composition.context import Context, RuntimeScope


PROBE = ServiceKey[object]("test.channel_scope_probe")


@asynccontextmanager
async def live_root(tmp_path):
    root = CompositionRoot("channel-scope-test")
    service = object()

    async def provide(ctx: Context) -> None:
        await ctx.provide(PROBE, service)

    fiber = await root.mount(
        provide, name="probe",
        runtime=PluginRuntime("probe", "probe-generation", tmp_path, tmp_path, tmp_path, {}),
    )
    try:
        yield root, fiber.context, service
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_public_lease_exposes_no_snapshot_root_or_service_surface(tmp_path):
    """A captured call carries authority without exposing the old snapshot graph."""
    async with live_root(tmp_path) as (_root, ctx, service):
        async with ctx.runtime_scope():
            scope = ctx.capture_runtime_scope()
            assert type(scope) is RuntimeScope
            for attr in ("snapshot", "composition_root", "context", "root", "require", "get"):
                assert not hasattr(scope, attr), attr
            assert not hasattr(scope, "__dict__")
            assert ctx.require_runtime_owner(PROBE, service) == "probe"
            await scope.close()


@pytest.mark.asyncio
async def test_channel_input_requires_current_task_exact_root_and_active(tmp_path):
    """A service call rejects ambient, child Task, foreign Root, and retired authority."""
    async with live_root(tmp_path / "a") as (_first, ctx, service):
        async with live_root(tmp_path / "b") as (_second, other, foreign):
            with pytest.raises(CompositionError, match="OwnerCall"):
                ctx.require_runtime_owner(PROBE, service)

            async with ctx.runtime_scope():
                assert ctx.require_runtime_owner(PROBE, service) == "probe"
                with pytest.raises(CompositionError) as caught:
                    ctx.require_runtime_owner(PROBE, foreign)
                assert caught.value.code == "SERVICE_SCOPE_MISMATCH"

                async def inherited_context() -> None:
                    ctx.require_runtime_owner(PROBE, service)

                with pytest.raises(CompositionError, match="OwnerCall"):
                    await asyncio.create_task(inherited_context())

                transferred = ctx.capture_runtime_scope()

            async def explicit_transfer() -> str:
                async with transferred:
                    return ctx.require_runtime_owner(PROBE, service)

            assert await asyncio.create_task(explicit_transfer()) == "probe"
            with pytest.raises(CompositionError, match="OwnerCall"):
                other.require_runtime_owner(PROBE, foreign)

        with pytest.raises(CompositionError, match="OwnerCall"):
            ctx.require_runtime_owner(PROBE, service)


@pytest.mark.asyncio
async def test_scope_close_and_release_reject_further_binding(tmp_path):
    """An exact captured call is single entry and releases on close."""
    async with live_root(tmp_path) as (_root, ctx, service):
        async with ctx.runtime_scope():
            scope = ctx.capture_runtime_scope()
        async with scope:
            assert ctx.require_runtime_owner(PROBE, service) == "probe"
        with pytest.raises(RuntimeError, match="只能进入一次"):
            async with scope:
                pass
        with pytest.raises(CompositionError, match="OwnerCall"):
            ctx.require_runtime_owner(PROBE, service)
