"""后台任务通知的进程内状态：一次结算只被一个回合领取。"""
from __future__ import annotations

from dataclasses import dataclass

from .contract import JobOutcome, JobWatch

_WAKE = "wake:"


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
