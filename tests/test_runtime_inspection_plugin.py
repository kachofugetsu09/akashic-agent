"""验证 runtime_inspection 是可选的普通 provider，而非 Core 业务实现。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.rpc import rpc_method_key
from plugins.runtime_inspection import plugin
from plugins.runtime_inspection.inspection import (
    SCHEDULER_INSPECTION,
    SKILL_INSPECTION,
)


def _payload(value: object) -> Mapping[str, object]:
    """窄化 inspection RPC 的结构化返回值。"""
    if not isinstance(value, Mapping):
        raise AssertionError("inspection RPC 必须返回对象")
    return cast(Mapping[str, object], value)


def _rows(value: object) -> tuple[Mapping[str, object], ...]:
    """窄化 inspection RPC 中的列表行。"""
    if not isinstance(value, (list, tuple)):
        raise AssertionError("inspection RPC rows 必须是列表")
    rows: list[Mapping[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise AssertionError("inspection RPC row 必须是对象")
        rows.append(cast(Mapping[str, object], row))
    return tuple(rows)


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
    async def call(method, params=None):
        operation = root.context.require(rpc_method_key("inspection/" + method))
        return await operation.invoke(operation.params.model_validate(params or {}), None)
    documents = _payload(await call("documents.list"))
    assert [item["id"] for item in _rows(documents["items"])] == [
        "memory",
        "self",
        "veda",
    ]
    memory = _payload(await call("documents.get", {"document_id": "memory"}))
    assert memory["markdown"] == "# Memory\n"
    skills = _payload(await call("skills.list"))
    assert _payload(skills["unavailable"])["code"] == "skills_unavailable"
    jobs = _payload(await call("jobs.list"))
    assert _payload(jobs["unavailable"])["code"] == "scheduler_unavailable"

    scheduler = _Scheduler()
    skills = _Skills()

    async def mount_business(ctx):
        await ctx.provide(SCHEDULER_INSPECTION, scheduler)
        await ctx.provide(SKILL_INSPECTION, skills)

    business = await root.mount(mount_business, name="business-providers")
    jobs = _payload(await call("jobs.list"))
    assert tuple(_rows(jobs["items"])) == ({"id": "external-job", "name": "外部任务"},)
    skills = _payload(await call("skills.list"))
    assert tuple(_rows(skills["items"])) == ({"name": "external-skill", "available": True},)
    await business.dispose()
    jobs = _payload(await call("jobs.list"))
    assert _payload(jobs["unavailable"])["code"] == "scheduler_unavailable"
    skills = _payload(await call("skills.list"))
    assert _payload(skills["unavailable"])["code"] == "skills_unavailable"
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
    operation = root.context.require(rpc_method_key("inspection/jobs.list"))
    with pytest.raises(RuntimeError, match="scheduler read failed"):
        await operation.invoke(operation.params.model_validate({}), None)
    await root.dispose()


@pytest.mark.asyncio
async def test_core_inspection_binds_lease_for_real_skill_projection(tmp_path: Path) -> None:
    """真实技能 owner 的租约读取经 Web/Mobile 共用的检查入口完成。"""
    import shutil
    from bus.event_bus import EventBus
    from agent.plugins.manager import PluginManager
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore
    from agent.plugins.snapshot import get_current_runtime_snapshot
    from infra.mobile_realtime.runtime_inspection import RuntimeInspectionService

    source = tmp_path / "plugins"
    for name in ("content", "context", "tools", "standard_tools", "runtime_inspection"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    asset = source / "external_assets"
    skill = asset / "skills" / "probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: probe\ndescription: lease proof\n---\nRead safely.\n")
    (asset / "plugin.py").write_text(
        "api_version = 3\nname = 'external_assets'\nversion = '1'\n"
        "asset_roots = {'skills': ('skills',)}\ndef apply(ctx, config): pass\n")
    (tmp_path / "workspace").mkdir()
    configuration = tmp_path / "workspace/plugin-data/context-builtin"
    configuration.mkdir(parents=True)
    (configuration / "config.local.toml").write_text('summary_source=[]\nprompt_sources={skills="standard_tools"}\n')
    metadata = ArtifactStore(tmp_path / "workspace" / "sessions.db")
    attachments = ChannelAttachmentArtifactStore(workspace=tmp_path / "workspace", metadata_store=metadata)
    manager = PluginManager([source], event_bus=EventBus(), workspace=tmp_path / "workspace",
                            installed_cache_root=tmp_path / "empty-cache", channel_attachment_store=attachments)
    try:
        await manager.load_all()
        snapshot = manager.snapshot_store.current
        assert snapshot is not None
        service = RuntimeInspectionService(workspace=tmp_path / "workspace", snapshot_store=manager.snapshot_store)
        for _ in range(2):
            result = await service.list_capabilities()
            assert [item["name"] for item in _rows(result["skills"])] == ["probe"]
            assert snapshot.lease_count == 0
            assert get_current_runtime_snapshot() is None
    finally:
        await manager.terminate_all()
        metadata.close()
