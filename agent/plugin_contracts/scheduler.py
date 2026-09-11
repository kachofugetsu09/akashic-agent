"""调度能力的公开结构合同。

`infra/mobile_realtime` 的运行时检查需要列出定时任务。此前它直接 import
`plugins.scheduler.store.JobStore` 并按插件的私有 JSON 结构解析
`workspace/schedules.json` —— 那是 Core 在解析插件的数据格式：插件一旦移出
仓库，Core 就带着一份偷偷依赖的 schema。

本模块把这件事变成一条显式的只读 seam：`JobView` 是插件交给 Core 的纯值，
`SchedulerJobsPort` 是插件实现、Core 消费的只读接口。调度插件缺席时
`get` 返回 None，Core 的检查面因此退化为「没有定时任务」，而不是崩溃。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class JobView:
    """一个定时任务对 Core 只读检查面暴露的纯值。

    `schedule_text` 与 `content` 由调度插件负责渲染/取值，Core 只做展示排版，
    因此 Core 不需要知道 `schedules.json` 的字段语义或 cron 表示。
    """

    id: str
    name: str | None
    trigger: str
    tier: str
    fire_at: str
    timezone: str
    enabled: bool
    run_count: int
    schedule_text: str
    content: str
    state: str = "启用"


@runtime_checkable
class SchedulerJobsPort(Protocol):
    """调度插件对 Core 只读检查面暴露的定时任务视图。"""

    def list_jobs(self) -> tuple[JobView, ...]:
        """按触发时间返回全部任务；不修改调度状态。"""
        ...

    def get_job(self, job_id: str) -> JobView | None:
        """按 id 取一个任务；不存在时返回 None。"""
        ...


SCHEDULER_JOBS = ServiceKey[SchedulerJobsPort]("scheduler.jobs.read.v1")
