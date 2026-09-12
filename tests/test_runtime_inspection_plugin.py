"""验证 runtime_inspection 是可选的普通 provider，而非 Core 业务实现。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime, ServiceKey
from plugins.runtime_inspection import plugin
from plugins.runtime_inspection.inspection import (
    RUNTIME_INSPECTION,
    SCHEDULER_INSPECTION,
    SKILL_INSPECTION,
)


class _Scheduler:
    def list_jobs(self) -> tuple[Mapping[str, object], ...]:
        return ({"id": "external-job", "name": "外部任务"},)

    def get_job(self, job_id: str) -> Mapping[str, object] | None:
        return {"id": job_id, "name": "外部任务"}


class _Skills:
    def list_skills(self) -> tuple[Mapping[str, object], ...]:
        return ({"name": "external-skill", "available": True},)


@pytest.mark.asyncio
async def test_runtime_inspection_plugin_owns_documents_and_optional_providers(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (memory / "SELF.md").write_text("# Self\n", encoding="utf-8")
    (memory / "VEDA.md").write_text("# VEDA\n", encoding="utf-8")
    runtime = PluginRuntime(
        plugin_id="runtime_inspection",
        generation_id="runtime-inspection:g1",
        plugin_dir=tmp_path,
        data_dir=tmp_path / "plugin-data",
        workspace=tmp_path,
        config={},
        workspace_files=plugin.workspace_files,
    )
    root = CompositionRoot("runtime-inspection")

    async def mount_runtime(ctx):
        await plugin.apply(ctx, {})

    await root.mount(mount_runtime, name=plugin.name, runtime=runtime)
    provider = root.context.get(RUNTIME_INSPECTION)
    assert provider is not None
    assert [item["id"] for item in provider.list_documents()] == [
        "memory",
        "self",
        "veda",
    ]
    assert provider.get_document("memory")["markdown"] == "# Memory\n"
    assert provider.list_skills() is None
    assert provider.list_jobs() is None

    scheduler = _Scheduler()
    skills = _Skills()

    async def mount_business(ctx):
        await ctx.provide(SCHEDULER_INSPECTION, scheduler)
        await ctx.provide(SKILL_INSPECTION, skills)

    business = await root.mount(mount_business, name="business-providers")
    assert provider.list_jobs() == ({"id": "external-job", "name": "外部任务"},)
    assert provider.list_skills() == ({"name": "external-skill", "available": True},)
    await business.dispose()
    assert provider.list_jobs() is None
    assert provider.list_skills() is None
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_inspection_provider_propagates_business_failure(
    tmp_path: Path,
) -> None:
    runtime = PluginRuntime(
        plugin_id="runtime_inspection",
        generation_id="runtime-inspection:g1",
        plugin_dir=tmp_path,
        data_dir=tmp_path / "plugin-data",
        workspace=tmp_path,
        config={},
        workspace_files=plugin.workspace_files,
    )
    root = CompositionRoot("runtime-inspection-failure")

    async def mount_runtime(ctx):
        await plugin.apply(ctx, {})

    await root.mount(mount_runtime, name=plugin.name, runtime=runtime)

    class BrokenScheduler(_Scheduler):
        def list_jobs(self) -> tuple[Mapping[str, object], ...]:
            raise RuntimeError("scheduler read failed")

    async def mount_business(ctx):
        await ctx.provide(SCHEDULER_INSPECTION, BrokenScheduler())

    await root.mount(mount_business, name="business-providers")
    provider = root.context.require(RUNTIME_INSPECTION)
    with pytest.raises(RuntimeError, match="scheduler read failed"):
        provider.list_jobs()
    await root.dispose()
