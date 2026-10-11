"""后台任务完成通知的公共合同；生产方只登记等待入口，投递归 jobs 插件。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import ServiceKey

JobStatus = Literal["completed", "failed", "timeout", "killed"]


@dataclass(frozen=True, slots=True)
class JobOutcome:
    status: JobStatus
    detail: str


# settled 在任务结算时返回结果；返回 None 表示模型已收走结果或任务被显式停止，不再通知。
@dataclass(frozen=True, slots=True)
class JobWatch:
    session_id: str
    source: str
    job_id: str
    kind: str
    label: str
    collect_hint: str
    settled: Callable[[], Awaitable[JobOutcome | None]]


class JobNotices(Protocol):
    def watch(self, job: JobWatch) -> None: ...


JOB_NOTICES = ServiceKey[JobNotices]("jobs.notices.v1")
