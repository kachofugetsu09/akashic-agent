"""任务与技能目录的只读合同。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey


class SchedulerReader(Protocol):
    """异步读取调度快照；取消退出前完成实际读取。"""

    async def list_jobs(self) -> tuple[Mapping[str, object], ...]: ...

    async def get_job(self, job_id: str) -> Mapping[str, object] | None: ...


SCHEDULER_INSPECTION_V3 = ServiceKey[SchedulerReader]("scheduler.inspection.v3")


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
