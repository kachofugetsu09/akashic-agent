from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from infra.mobile_realtime.runtime_inspection import (
    RuntimeInspectionError,
    RuntimeInspectionService,
)
from agent.scheduler import LatencyTracker, ScheduledJob, SchedulerService


def _service(tmp_path: Path) -> tuple[RuntimeInspectionService, SchedulerService]:
    scheduler = SchedulerService(
        store_path=tmp_path / "schedules.json",
        push_tool=object(),
        tracker=LatencyTracker(),
    )
    service = RuntimeInspectionService(
        workspace=tmp_path,
        scheduler=scheduler,
        snapshot_store=None,
    )
    return service, scheduler


def test_documents_use_fixed_allowlist_and_return_markdown(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    path = tmp_path / "memory/MEMORY.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Memory\n\n真实内容", encoding="utf-8")

    listed = service.list_documents()
    document = service.get_document("memory")

    assert len(cast(list[object], listed["items"])) == 6
    assert document["relative_path"] == "memory/MEMORY.md"
    assert document["markdown"] == "# Memory\n\n真实内容"
    with pytest.raises(RuntimeInspectionError, match="未知运行时文档"):
        service.get_document("../../config.toml")


def test_scheduler_projection_reads_live_service_state(tmp_path: Path) -> None:
    service, scheduler = _service(tmp_path)
    job = ScheduledJob(
        id="morning",
        name="晨间提醒",
        trigger="every",
        tier="instant",
        fire_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        channel="mobile",
        chat_id="mobile:test",
        interval_seconds=3600,
        message="起来走一走",
    )
    scheduler.add_job(job)

    listed = service.list_jobs()
    detail = service.get_job("morning")

    assert cast(list[dict[str, object]], listed["items"])[0]["id"] == "morning"
    assert "起来走一走" in cast(str, detail["markdown"])
    with pytest.raises(RuntimeInspectionError, match="定时任务不存在"):
        service.get_job("missing")


@pytest.mark.asyncio
async def test_capabilities_fail_loud_without_runtime_snapshot(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    with pytest.raises(RuntimeInspectionError, match="快照尚未就绪"):
        await service.list_capabilities()
