"""任务与技能目录的只读合同。"""

from collections.abc import Mapping
from typing import Protocol

from agent.plugin_composition import ServiceKey


class SchedulerReader(Protocol):
    """scheduler 只读投影的窄输入。"""

    def list_jobs(self) -> tuple[Mapping[str, object], ...]: ...

    def get_job(self, job_id: str) -> Mapping[str, object] | None: ...


class SkillReader(Protocol):
    """技能目录只读投影的窄输入。"""

    async def list_skills(self) -> tuple[Mapping[str, object], ...]: ...


SCHEDULER_INSPECTION = ServiceKey[SchedulerReader]("scheduler.inspection.v1")
SKILL_INSPECTION = ServiceKey[SkillReader]("standard_tools.skill_inspection.v1")
