from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.background_jobs import (
    BACKGROUND_JOBS,
    BackgroundJobDefinition,
    IntervalTrigger,
    PluginBackgroundJobs,
    RetryPolicy,
    _freeze_plugin_background_jobs,
)
from agent.plugin_composition.model import CompositionError


def _runtime(tmp_path: Path, plugin_id: str = "drift") -> PluginRuntime:
    plugin_dir = tmp_path / plugin_id
    plugin_dir.mkdir(parents=True)
    return PluginRuntime(
        plugin_id=plugin_id,
        generation_id="test-generation",
        plugin_dir=plugin_dir,
        data_dir=tmp_path / "data" / plugin_id,
        workspace=tmp_path / "workspace",
        config=None,
    )


def _definition(
    name: str = "merge_proactive_pending",
    *,
    programmatic_turns: bool = False,
) -> BackgroundJobDefinition:
    return BackgroundJobDefinition(
        name=name,
        triggers=(IntervalTrigger(60),),
        handler_export="merge_pending",
        debounce_seconds=5,
        coalesce=True,
        retry_policy=RetryPolicy(
            max_attempts=2,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
        ),
        model_role="proactive.merge",
        programmatic_turns=programmatic_turns,
    )


@pytest.mark.asyncio
async def test_background_job_registry_freezes_binding_and_live_fence(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("jobs-generation-1")
    service = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, service)

    async def apply(ctx) -> None:
        await ctx.require(BACKGROUND_JOBS).register(ctx, _definition())

    fiber = await root.mount(
        apply,
        name="drift",
        inject=(BACKGROUND_JOBS,),
        runtime=_runtime(tmp_path),
    )
    catalog = _freeze_plugin_background_jobs(service, root.instance_token)
    binding = catalog.job("drift:merge_proactive_pending")
    assert binding is not None
    assert binding.generation_id == "jobs-generation-1"
    assert binding.plugin_id == "drift"
    assert binding.handler_export == "merge_pending"
    assert binding.is_live()
    assert catalog.identity == catalog.catalog_digest
    assert catalog["drift:merge_proactive_pending"] is binding
    assert _freeze_plugin_background_jobs(service, root.instance_token) is catalog

    await fiber.dispose()
    assert not binding.is_live()
    assert _freeze_plugin_background_jobs(service, root.instance_token) is catalog
    await root.dispose()


@pytest.mark.asyncio
async def test_background_job_identity_ignores_generation_root_and_runtime_paths(
    tmp_path: Path,
) -> None:
    identities: list[str] = []
    for suffix in ("candidate", "formal"):
        root = CompositionRoot(f"jobs-{suffix}")
        service = PluginBackgroundJobs(root.instance_token)
        _ = await root.context.provide(BACKGROUND_JOBS, service)

        async def apply(ctx) -> None:
            await ctx.require(BACKGROUND_JOBS).register(ctx, _definition())

        _ = await root.mount(
            apply,
            name="drift",
            inject=(BACKGROUND_JOBS,),
            runtime=_runtime(tmp_path / suffix),
        )
        identities.append(
            _freeze_plugin_background_jobs(service, root.instance_token).identity
        )
        await root.dispose()
    assert identities[0] == identities[1]


@pytest.mark.asyncio
async def test_background_job_candidate_freeze_has_no_execution_surface(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("jobs-candidate")
    service = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, service)
    invocation_count = 0

    async def apply(ctx) -> None:
        await ctx.require(BACKGROUND_JOBS).register(ctx, _definition())

    _ = await root.mount(
        apply,
        name="drift",
        inject=(BACKGROUND_JOBS,),
        runtime=_runtime(tmp_path),
    )
    catalog = _freeze_plugin_background_jobs(service, root.instance_token)
    assert invocation_count == 0
    assert len(catalog.descriptors) == 1
    assert catalog.descriptors[0].triggers == (IntervalTrigger(60),)
    await root.dispose()


@pytest.mark.asyncio
async def test_background_job_preserves_explicit_programmatic_turn_declaration(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("jobs-programmatic")
    service = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, service)

    async def apply(ctx) -> None:
        await ctx.require(BACKGROUND_JOBS).register(
            ctx,
            _definition(programmatic_turns=True),
        )

    _ = await root.mount(
        apply,
        name="drift",
        inject=(BACKGROUND_JOBS,),
        runtime=_runtime(tmp_path),
    )
    catalog = _freeze_plugin_background_jobs(service, root.instance_token)
    binding = catalog.job("drift:merge_proactive_pending")
    assert binding is not None
    assert binding.definition.programmatic_turns is True
    assert catalog.descriptors[0].programmatic_turns is True
    await root.dispose()


@pytest.mark.asyncio
async def test_background_job_rejects_duplicate_after_freeze_and_cleans_on_failure(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("jobs-freeze")
    service = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, service)
    captured = None

    async def apply(ctx) -> None:
        nonlocal captured
        captured = ctx
        facade = ctx.require(BACKGROUND_JOBS)
        await facade.register(ctx, _definition())
        await facade.register(ctx, _definition())

    fiber = await root.mount(
        apply,
        name="drift",
        inject=(BACKGROUND_JOBS,),
        runtime=_runtime(tmp_path),
    )
    assert fiber.state.value == "failed"
    assert len(_freeze_plugin_background_jobs(service, root.instance_token)) == 0
    assert captured is not None
    await root.dispose()

    root = CompositionRoot("jobs-freeze-after")
    service = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, service)
    captured_after = None

    async def apply_once(ctx) -> None:
        nonlocal captured_after
        captured_after = ctx
        await ctx.require(BACKGROUND_JOBS).register(ctx, _definition())

    _ = await root.mount(
        apply_once,
        name="drift",
        inject=(BACKGROUND_JOBS,),
        runtime=_runtime(tmp_path / "after"),
    )
    _ = _freeze_plugin_background_jobs(service, root.instance_token)
    assert captured_after is not None
    with pytest.raises(CompositionError, match="已冻结"):
        await service.register(captured_after, _definition())
    await root.dispose()


@pytest.mark.asyncio
async def test_background_job_rejects_cross_root_registration(
    tmp_path: Path,
) -> None:
    root_a = CompositionRoot("jobs-a")
    root_b = CompositionRoot("jobs-b")
    service_a = PluginBackgroundJobs(root_a.instance_token)
    service_b = PluginBackgroundJobs(root_b.instance_token)
    _ = await root_a.context.provide(BACKGROUND_JOBS, service_a)
    _ = await root_b.context.provide(BACKGROUND_JOBS, service_b)

    async def apply_wrong_root(ctx) -> None:
        await service_a.register(ctx, _definition())

    _ = await root_b.mount(
        apply_wrong_root,
        name="drift",
        inject=(BACKGROUND_JOBS,),
        runtime=_runtime(tmp_path),
    )
    assert any(
        "Service 不属于当前 Root" in (fiber.error or "")
        for fiber in root_b.receipt().fibers
    )
    assert len(_freeze_plugin_background_jobs(service_a, root_a.instance_token)) == 0
    assert len(_freeze_plugin_background_jobs(service_b, root_b.instance_token)) == 0
    await root_a.dispose()
    await root_b.dispose()


@pytest.mark.asyncio
async def test_background_job_name_is_unique_per_owner(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("jobs-multi-owner")
    service = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, service)

    async def apply(ctx) -> None:
        await ctx.require(BACKGROUND_JOBS).register(ctx, _definition("refresh"))

    for plugin_id in ("emotion", "fitbit"):
        _ = await root.mount(
            apply,
            name=plugin_id,
            inject=(BACKGROUND_JOBS,),
            runtime=_runtime(tmp_path, plugin_id),
        )

    catalog = _freeze_plugin_background_jobs(service, root.instance_token)
    assert tuple(catalog) == ("emotion:refresh", "fitbit:refresh")
    assert catalog.job("refresh") is None
    await root.dispose()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: IntervalTrigger(0),
        lambda: IntervalTrigger(True),
        lambda: BackgroundJobDefinition("bad", (), "run"),
        lambda: BackgroundJobDefinition(
            "bad",
            (IntervalTrigger(1),) * 2,
            "run",
        ),
        lambda: BackgroundJobDefinition("bad", (IntervalTrigger(1),), "bad export"),
        lambda: BackgroundJobDefinition(
            "bad",
            (IntervalTrigger(1),),
            "run",
            programmatic_turns=cast(bool, 1),
        ),
        lambda: RetryPolicy(max_attempts=0),
        lambda: RetryPolicy(base_delay_seconds=float("nan")),
        lambda: RetryPolicy(max_delay_seconds=float("inf")),
    ),
)
def test_background_job_models_reject_invalid_contract(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
