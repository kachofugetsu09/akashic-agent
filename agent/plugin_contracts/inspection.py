"""任务与技能目录的只读合同。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey


class SchedulerReader(Protocol):
    """scheduler 只读投影的窄输入。"""

    def list_jobs(self) -> tuple[Mapping[str, object], ...]: ...

    def get_job(self, job_id: str) -> Mapping[str, object] | None: ...


class SkillReader(Protocol):
    """技能目录只读投影的窄输入。"""

    async def list_skills(self) -> tuple[Mapping[str, object], ...]: ...


SCHEDULER_INSPECTION = ServiceKey[SchedulerReader]("scheduler.inspection.v1")
SKILL_INSPECTION = ServiceKey[SkillReader]("standard_tools.skill_inspection.v1")


@dataclass(frozen=True, slots=True)
class Document:
    """owner 发布的只读展示元数据与有界读取口；不授予任意文件访问。"""

    id: str
    title: str
    relative_path: str
    group: str
    description: str
    read: Callable[[int], bytes]
    order: int = 0


class Documents(Protocol):
    async def register(self, ctx: Context, document: Document) -> Effect: ...


DOCUMENTS = ServiceKey[Documents]("inspection.documents.v1")
