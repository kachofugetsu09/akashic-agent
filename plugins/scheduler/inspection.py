"""scheduler 自己拥有的只读检查能力。"""

from __future__ import annotations

from collections.abc import Mapping

from agent.plugin_composition import ServiceKey

from .dashboard import job_detail, job_summary
from .store import JobStore


class SchedulerInspectionProvider:
    """只读并投影 scheduler 状态，不修改权威持久化。"""

    def __init__(self, store: JobStore) -> None:
        self._store = store

    def list_jobs(self) -> tuple[Mapping[str, object], ...]:
        """从 owner store 读取启用任务并确定性排序。"""
        return tuple(
            job_summary(job)
            for job in sorted(self._store.load(), key=lambda item: (item.fire_at, item.id))
            if job.enabled
        )

    def get_job(self, job_id: str) -> Mapping[str, object] | None:
        """读取一个启用任务，不改变 scheduler 状态。"""
        for job in self._store.load():
            if job.id == job_id and job.enabled:
                return job_detail(job)
        return None


SCHEDULER_INSPECTION = ServiceKey[SchedulerInspectionProvider]("scheduler.inspection.v1")
