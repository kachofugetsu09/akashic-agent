"""提供 scheduler 自己拥有的只读展示值。"""

from __future__ import annotations

from .schedule import ScheduledJob


def job_summary(job: ScheduledJob) -> dict[str, object]:
    """投影运行时检查使用的稳定摘要。"""
    return {
        "id": job.id,
        "name": job.name,
        "trigger": job.trigger,
        "action": job.action,
        "context": job.context,
        "session_id": job.session_id,
        "fire_at": job.fire_at.isoformat(),
        "timezone": job.timezone,
        "enabled": job.enabled,
        "run_count": job.run_count,
    }


def job_markdown(job: ScheduledJob) -> str:
    """渲染 scheduler 字段，Core 无需理解任务 schema。"""
    schedule = job.cron_expr or (
        f"每 {job.interval_seconds} 秒"
        if job.interval_seconds is not None
        else job.fire_at.isoformat()
    )
    return "\n".join(
        (
            f"# {job.name or '未命名定时任务'}",
            "",
            f"- **状态：** {'启用' if job.enabled else '停用'}",
            f"- **触发：** `{job.trigger}` / `{job.action}`",
            f"- **会话：** `{job.session_id}`（历史：`{job.context}`）",
            f"- **计划：** {schedule}",
            f"- **时区：** `{job.timezone}`",
            f"- **运行次数：** {job.run_count}",
            "",
            "## 内容",
            "",
            job.text,
        )
    )


def job_detail(job: ScheduledJob) -> dict[str, object]:
    """在任务摘要中加入 scheduler 自己拥有的详情文档。"""
    return {**job_summary(job), "markdown": job_markdown(job)}
