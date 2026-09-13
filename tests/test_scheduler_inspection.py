"""验证 scheduler 检查保持在插件边界内。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.akashic_clients.runtime_inspection import ScopedRpcRuntimeInspection
from plugins.scheduler.inspection import SchedulerInspectionProvider
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore


def _job(
    job_id: str,
    fire_at: datetime,
    *,
    enabled: bool = True,
    message: str = "检查状态",
) -> ScheduledJob:
    return ScheduledJob(
        trigger="every",
        tier="instant",
        fire_at=fire_at,
        channel="web",
        chat_id="chat-1",
        interval_seconds=60,
        message=message,
        enabled=enabled,
        name=job_id,
        id=job_id,
    )


def _inspection_service(store):
    """Resolve inspection RPC methods through one current snapshot per call."""

    @asynccontextmanager
    async def open_scope():
        async with lease_runtime_snapshot(store) as snapshot:
            root = snapshot.composition_root
            if root is None:
                raise RuntimeError("inspection fixture 缺少 composition root")
            yield root.context

    return ScopedRpcRuntimeInspection(open_scope)


def test_scheduler_provider_projects_only_enabled_jobs(tmp_path: Path) -> None:
    first = _job("first", datetime(2026, 9, 12, 1, 0, tzinfo=UTC))
    second = _job("second", first.fire_at + timedelta(minutes=1))
    disabled = _job("disabled", first.fire_at - timedelta(minutes=1), enabled=False)
    store = JobStore(tmp_path / "schedules.json")
    store.save({job.id: job for job in (second, disabled, first)})

    provider = SchedulerInspectionProvider(store)

    assert [item["id"] for item in provider.list_jobs()] == ["first", "second"]
    assert provider.get_job("disabled") is None
    detail = provider.get_job("first")
    assert detail is not None
    assert {key: value for key, value in detail.items() if key != "markdown"} == {
        "id": "first",
        "name": "first",
        "trigger": "every",
        "tier": "instant",
        "fire_at": "2026-09-12T01:00:00+00:00",
        "timezone": "UTC",
        "enabled": True,
        "run_count": 0,
    }
    assert detail["markdown"] == (
        "# first\n\n"
        "- **状态：** 启用\n"
        "- **触发：** `every` / `instant`\n"
        "- **计划：** 每 60 秒\n"
        "- **时区：** `UTC`\n"
        "- **运行次数：** 0\n\n"
        "## 内容\n\n"
        "检查状态"
    )


class _Provider:
    def list_jobs(self) -> tuple[Mapping[str, object], ...]:
        return (
            {"id": "external", "display": "来自 scheduler"},
        )

    def get_job(self, job_id: str) -> Mapping[str, object] | None:
        return {"id": job_id, "display": "来自 scheduler"}


@pytest.mark.asyncio
async def test_client_passes_through_scheduler_projection_without_reading_workspace(tmp_path: Path) -> None:
    from agent.plugin_composition import CompositionRoot
    from agent.plugin_composition.rpc import rpc_method_key
    from plugins.runtime_inspection.rpc import rpc_methods
    from plugins.runtime_inspection.inspection import RuntimeInspectionProvider
    from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore, get_current_runtime_snapshot

    root = CompositionRoot("scheduler-inspection")
    provider = _Provider()
    async def apply(ctx):
        for method, operation in rpc_methods(cast(RuntimeInspectionProvider, provider)).items():
            await ctx.provide(rpc_method_key(method), operation)
    await root.mount(apply, name="external-scheduler")
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    service = _inspection_service(store)
    original = provider.list_jobs
    def list_jobs():
        assert get_current_runtime_snapshot() is snapshot
        assert snapshot.lease_count == 1
        return original()
    provider.list_jobs = list_jobs
    try:
        assert await service.list_jobs() == {"items": [{"id": "external", "display": "来自 scheduler"}]}
        assert await service.get_job("external") == {"id": "external", "display": "来自 scheduler"}
        assert snapshot.lease_count == 0
        assert not (tmp_path / "schedules.json").exists()
    finally:
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_client_does_not_swallow_scheduler_provider_failure(tmp_path: Path) -> None:
    from agent.plugin_composition import CompositionRoot
    from agent.plugin_composition.rpc import rpc_method_key
    from plugins.runtime_inspection.rpc import rpc_methods
    from plugins.runtime_inspection.inspection import RuntimeInspectionProvider
    from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore

    class BrokenProvider:
        def list_jobs(self) -> tuple[Mapping[str, object], ...]:
            raise RuntimeError("scheduler read failed")

        def get_job(self, job_id: str) -> Mapping[str, object] | None:
            raise RuntimeError("scheduler read failed")

    root = CompositionRoot("scheduler-inspection-failure")
    async def apply(ctx):
        for method, operation in rpc_methods(cast(RuntimeInspectionProvider, BrokenProvider())).items():
            await ctx.provide(rpc_method_key(method), operation)
    await root.mount(apply, name="external-scheduler")
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    service = _inspection_service(store)
    try:
        with pytest.raises(RuntimeError, match="scheduler read failed"):
            await service.list_jobs()
    finally:
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_client_reports_scheduler_unavailable_without_provider(tmp_path: Path) -> None:
    from agent.plugin_composition import CompositionRoot
    from agent.plugin_composition.rpc import rpc_method_key
    from plugins.runtime_inspection.inspection import RuntimeInspectionProvider
    from plugins.runtime_inspection.rpc import rpc_methods
    from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore

    root = CompositionRoot("scheduler-inspection-unavailable")
    provider = RuntimeInspectionProvider(
        {name: tmp_path / f"{name}.md" for name in ("memory", "self", "veda")}
    )

    async def apply(ctx):
        for method, operation in rpc_methods(provider).items():
            await ctx.provide(rpc_method_key(method), operation)

    await root.mount(apply, name="runtime-inspection")
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    service = _inspection_service(store)
    try:
        assert await service.list_jobs() == {
            "unavailable": {
                "code": "scheduler_unavailable",
                "message": "调度检查服务尚未绑定",
            }
        }
    finally:
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_client_reports_skills_unavailable_without_provider(tmp_path: Path) -> None:
    from agent.plugin_composition import CompositionRoot
    from agent.plugin_composition.rpc import rpc_method_key
    from plugins.runtime_inspection.inspection import RuntimeInspectionProvider
    from plugins.runtime_inspection.rpc import rpc_methods
    from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore

    root = CompositionRoot("skill-inspection")
    provider = RuntimeInspectionProvider(
        {name: tmp_path / f"{name}.md" for name in ("memory", "self", "veda")}
    )

    async def apply(ctx):
        for method, operation in rpc_methods(provider).items():
            await ctx.provide(rpc_method_key(method), operation)

    await root.mount(apply, name="runtime-inspection")
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    service = _inspection_service(store)
    try:
        payload = await service.list_capabilities()
        assert payload == {
            "unavailable": {
                "code": "skills_unavailable",
                "message": "技能检查服务尚未绑定",
            }
        }
    finally:
        await store.close()
        await root.dispose()


def test_client_runtime_inspection_has_no_scheduler_implementation_import() -> None:
    source = (Path(__file__).parents[1] / "plugins/akashic_clients/runtime_inspection.py").read_text(
        encoding="utf-8"
    )

    assert "plugins.scheduler" not in source
    assert "JobStore" not in source
    assert "ScheduledJob" not in source
    assert "scheduler.inspection" not in source
    assert "standard_tools" not in source
    assert "memory/MEMORY.md" not in source
    assert "memory/VEDA.md" not in source
