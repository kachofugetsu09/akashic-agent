"""Scheduler 发布的只读任务快照合同。"""

from collections.abc import Mapping
from typing import Protocol

from agent.plugin_composition import ServiceKey


class SchedulerReader(Protocol):
    """异步读取调度快照；取消退出前完成实际读取。"""

    async def list_jobs(self) -> tuple[Mapping[str, object], ...]: ...

    async def get_job(self, job_id: str) -> Mapping[str, object] | None: ...


SCHEDULER_INSPECTION = ServiceKey[SchedulerReader]("scheduler.inspection.v3")
