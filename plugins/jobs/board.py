"""后台任务通知的进程内状态：一次结算只被一个回合领取。"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace

from .contract import JobOutcome, JobSnapshot, JobWatch

_WAKE = "wake:"
_SETTLED_KEPT = 20


@dataclass(slots=True, eq=False)
class Notice:
    job: JobWatch
    outcome: JobOutcome
    claimed_by: str | None = None

    # 通知只说明结算事实与读取方式；结果正文仍由生产方的读取工具交付。
    def text(self) -> str:
        job, outcome = self.job, self.outcome
        return (
            f"后台任务 {job.job_id}（{job.kind}：{job.label}）已结束 "
            f"[status: {outcome.status}，{outcome.detail}]。{job.collect_hint}\n"
            "这是后台执行结果通知，不是用户的新指令。"
        )


class JobBoard:
    """忙时由当前输入领取并在该输入的每一步重放；闲时由唤醒领取。"""

    def __init__(self, max_consecutive_wakes: int | None = None) -> None:
        self._max_wakes = max_consecutive_wakes
        self._notices: list[Notice] = []
        self._wakes: dict[str, tuple[str | None, int]] = {}

    def settle(self, job: JobWatch, outcome: JobOutcome) -> Notice:
        notice = Notice(job, outcome)
        self._notices.append(notice)
        return notice

    def pending(self, session_id: str) -> tuple[Notice, ...]:
        return tuple(notice for notice in self._notices if notice.job.session_id == session_id)

    def origin_source(self, session_id: str, source: str) -> str:
        """仅从本会话正在唤醒的登记记录取得原来源，不解析或猜测来源名。"""
        for notice in self._notices:
            if (notice.job.session_id == session_id
                    and source == "job:" + notice.job.job_id
                    and notice.claimed_by == _WAKE + notice.job.job_id):
                return notice.job.source
        return source

    # 同一输入的后续步骤看到同一组通知；新输入出现后，旧输入领取过的通知退出。
    def claim(self, session_id: str, source: str, input_id: str) -> tuple[Notice, ...]:
        visible: list[Notice] = []
        for notice in tuple(self._notices):
            if notice.job.session_id != session_id or notice.job.source != source:
                continue
            if notice.claimed_by is None:
                notice.claimed_by = input_id
            if notice.claimed_by == input_id:
                visible.append(notice)
            elif not notice.claimed_by.startswith(_WAKE):
                self._notices.remove(notice)
        return tuple(visible)

    # 在会话槽内决定是否唤醒：已被回合领取或连续唤醒预算耗尽时，留给下一次用户输入注入。
    def take_wake(self, notice: Notice, input_id: str | None) -> bool:
        if notice not in self._notices or notice.claimed_by is not None:
            return False
        session_id = notice.job.session_id
        last_input, count = self._wakes.get(session_id, (input_id, 0))
        if last_input != input_id:
            count = 0
        if self._max_wakes is not None and count >= self._max_wakes:
            return False
        self._wakes[session_id] = (input_id, count + 1)
        notice.claimed_by = _WAKE + notice.job.job_id
        return True

    def finish_wake(self, notice: Notice) -> None:
        if notice in self._notices:
            self._notices.remove(notice)

    # 唤醒失败时撤回领取，下一次用户输入仍能看到这条结算。
    def release_wake(self, notice: Notice) -> None:
        if notice in self._notices and notice.claimed_by == _WAKE + notice.job.job_id:
            notice.claimed_by = None


# 前端只读镜像：运行中的任务按登记顺序保留，已结束的每会话只留最近若干条；关注者按会话被 Event 唤醒。
# 镜像随插件 activation 存亡：close 后所有 follow 结束，由消费方失效旧快照并向新一代重新订阅。
class JobMirror:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, JobSnapshot]] = {}
        self._watchers: dict[str, set[asyncio.Event]] = {}
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self._rows.clear()
        for watchers in self._watchers.values():
            for changed in watchers:
                changed.set()

    def start(self, job: JobWatch) -> None:
        rows = self._rows.setdefault(job.session_id, {})
        rows[job.job_id] = JobSnapshot(
            job.job_id, job.kind, job.label, "running", "", job.started_at_ms or _now_ms(),
            stoppable=job.stop is not None,
        )
        self._signal(job.session_id)

    # outcome 为 None 表示结果已被模型收走或任务被停止：没有可信终态，直接移出镜像。
    def finish(self, job: JobWatch, outcome: JobOutcome | None) -> None:
        rows = self._rows.get(job.session_id, {})
        current = rows.pop(job.job_id, None)
        if current is None:
            return
        if outcome is not None:
            rows[job.job_id] = replace(
                current, status=outcome.status, detail=outcome.detail, finished_at=_now_ms(), stoppable=False,
            )
            settled = [job_id for job_id, row in rows.items() if row.status not in ("running", "stopping")]
            for job_id in settled[:-_SETTLED_KEPT]:
                del rows[job_id]
        if not rows:
            _ = self._rows.pop(job.session_id, None)
        self._signal(job.session_id)

    # 运行中 ↔ 停止中的转换；停止入口失败时退回运行中。返回是否发生了转换。
    def set_stopping(self, session_id: str, job_id: str, stopping: bool) -> bool:
        rows = self._rows.get(session_id, {})
        current = rows.get(job_id)
        if current is None or current.status != ("running" if stopping else "stopping"):
            return False
        rows[job_id] = replace(current, status="stopping" if stopping else "running", stoppable=not stopping)
        self._signal(session_id)
        return True

    def snapshot(self, session_id: str) -> tuple[JobSnapshot, ...]:
        return tuple(self._rows.get(session_id, {}).values())

    async def follow(self, session_id: str) -> AsyncGenerator[tuple[JobSnapshot, ...], None]:
        changed = asyncio.Event()
        self._watchers.setdefault(session_id, set()).add(changed)
        try:
            while not self._closed:
                changed.clear()
                yield self.snapshot(session_id)
                _ = await changed.wait()
        finally:
            watchers = self._watchers.get(session_id)
            if watchers is not None:
                watchers.discard(changed)
                if not watchers:
                    del self._watchers[session_id]

    def _signal(self, session_id: str) -> None:
        for changed in self._watchers.get(session_id, ()):
            changed.set()


def _now_ms() -> int:
    return int(time.time() * 1000)
