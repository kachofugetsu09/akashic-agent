"""验证 scheduler 检查保持在插件边界内。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from infra.mobile_realtime.runtime_inspection import (
    RuntimeInspectionError,
    RuntimeInspectionService,
)
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


def test_core_passes_through_scheduler_projection_without_reading_workspace(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    context = SimpleNamespace(
        get=lambda key: provider
        if key.name == "scheduler.inspection.v1"
        else None
    )
    snapshot_store = SimpleNamespace(
        current=SimpleNamespace(composition_root=SimpleNamespace(context=context))
    )
    service = RuntimeInspectionService(
        workspace=tmp_path,
        snapshot_store=snapshot_store,
    )

    assert service.list_jobs() == {
        "items": [{"id": "external", "display": "来自 scheduler"}]
    }
    assert service.get_job("external") == {
        "id": "external",
        "display": "来自 scheduler",
    }
    assert not (tmp_path / "schedules.json").exists()


def test_core_reports_scheduler_unavailable_without_provider(tmp_path: Path) -> None:
    service = RuntimeInspectionService(workspace=tmp_path, snapshot_store=None)

    with pytest.raises(RuntimeInspectionError, match="调度检查服务尚未绑定") as error:
        service.list_jobs()

    assert error.value.code == "scheduler_unavailable"


def test_core_runtime_inspection_has_no_scheduler_implementation_import() -> None:
    source = (Path(__file__).parents[1] / "infra/mobile_realtime/runtime_inspection.py").read_text(
        encoding="utf-8"
    )

    assert "plugins.scheduler" not in source
    assert "JobStore" not in source
    assert "ScheduledJob" not in source
