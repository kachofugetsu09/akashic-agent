"""
定时任务工具：ScheduleTool / ListSchedulesTool / CancelScheduleTool

AI 通过这三个工具注册、查询、取消定时任务。
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.scheduler import (
    ScheduledJob,
    SchedulerService,
    compute_fire_at,
    is_cron_expr,
    parse_duration,
)
from agent.tools.base import Tool


class ScheduleTool(Tool):
    name = "schedule"
    description = (
        "注册定时任务。支持三种触发模式：\n"
        "  at    — 指定绝对时间，如 '14:30' 或 '2025-06-01T09:00'\n"
        "  after — 相对延迟，如 '30s' '5m' '2h'（需传 request_time 补偿延迟）\n"
        "  every — 循环，如 '1h' '30m' '0 9 * * *'（每天9点）\n\n"
        "两种执行模式：\n"
        "  instant — 到时直接推送固定消息，适合喝水提醒等固定文本\n"
        "  soft    — 到时调用 AI 生成实时内容，适合天气/新闻等"
    )
    parameters = {
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["instant", "soft"],
                "description": "instant=直接推消息；soft=触发时调用AI生成内容",
            },
            "trigger": {
                "type": "string",
                "enum": ["at", "after", "every"],
                "description": "触发模式",
            },
            "when": {
                "type": "string",
                "description": (
                    "触发时间描述，与 trigger 对应：\n"
                    "  at    → '14:30' 或 '2025-06-01T09:00'\n"
                    "  after → '30s' '5m' '2h'\n"
                    "  every → '1h' '30m' '0 9 * * *'"
                ),
            },
            "message": {
                "type": "string",
                "description": "tier=instant 时的消息内容（必填）",
            },
            "prompt": {
                "type": "string",
                "description": "tier=soft 时触发 AI 的提示词（必填）",
            },
            "channel": {
                "type": "string",
                "description": "目标渠道，如 telegram、qq",
            },
            "chat_id": {
                "type": "string",
                "description": "目标会话 ID",
            },
            "timezone": {
                "type": "string",
                "description": "时区，如 Asia/Shanghai，默认使用系统配置",
            },
            "name": {
                "type": "string",
                "description": "任务名，方便后续用 cancel_schedule 取消",
            },
            "request_time": {
                "type": "string",
                "description": (
                    "trigger=after 时必填：来自 system prompt 的消息接收时间（ISO 格式）。"
                    "用于从用户发消息时刻计算延迟，而非从 tool 调用时刻计算。"
                ),
            },
        },
        "required": ["tier", "trigger", "when", "channel", "chat_id"],
    }

    def __init__(self, service: SchedulerService, default_tz: str = "UTC") -> None:
        self._service = service
        self._default_tz = default_tz

    async def execute(self, **kwargs: Any) -> str:
        tier = kwargs.get("tier", "")
        trigger = kwargs.get("trigger", "")
        when = kwargs.get("when", "")
        message = kwargs.get("message")
        prompt = kwargs.get("prompt")
        channel = kwargs.get("channel", "")
        chat_id = str(kwargs.get("chat_id", ""))
        tz = kwargs.get("timezone") or self._default_tz
        name = kwargs.get("name")
        request_time = kwargs.get("request_time")

        # ── validation ──
        if tier not in ("instant", "soft"):
            return f"错误：tier 须为 instant 或 soft，收到 {tier!r}"
        if trigger not in ("at", "after", "every"):
            return f"错误：trigger 须为 at/after/every，收到 {trigger!r}"
        if tier == "instant" and not message:
            return "错误：tier=instant 时 message 为必填项"
        if tier == "soft" and not prompt:
            return "错误：tier=soft 时 prompt 为必填项"
        if not channel or not chat_id:
            return "错误：channel 和 chat_id 为必填项"

        try:
            ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            return f"错误：无效的时区 {tz!r}"

        # ── compute fire_at ──
        try:
            fire_at = compute_fire_at(trigger, when, tz, request_time)
        except ValueError as e:
            return f"错误：{e}"

        # ── parse every spec ──
        interval_seconds = None
        cron_expr = None
        if trigger == "every":
            try:
                if is_cron_expr(when):
                    cron_expr = when.strip()
                else:
                    iv = parse_duration(when)
                    interval_seconds = int(iv.total_seconds())
            except ValueError as e:
                return f"错误：{e}"

        # ── build & register ──
        job = ScheduledJob(
            trigger=trigger,
            tier=tier,
            fire_at=fire_at,
            channel=channel,
            chat_id=chat_id,
            interval_seconds=interval_seconds,
            cron_expr=cron_expr,
            message=message,
            prompt=prompt,
            name=name,
            timezone=tz,
        )
        self._service.add_job(job)

        # 优先用 fire_at 自带的时区（来自 request_time 的 offset），
        # 让用户看到本地时间而不是 UTC
        try:
            if fire_at.tzinfo is not None and str(fire_at.tzinfo) not in ("UTC", "utc"):
                display_dt = fire_at
            elif request_time:
                parsed_rt = datetime.fromisoformat(request_time)
                display_dt = (
                    fire_at.astimezone(parsed_rt.tzinfo)
                    if parsed_rt.tzinfo
                    else fire_at.astimezone()
                )
            else:
                display_dt = fire_at.astimezone()
            time_str = display_dt.strftime("%Y-%m-%d %H:%M:%S %z")
        except Exception:
            time_str = fire_at.isoformat()

        label = f"「{name}」" if name else job.id[:8]
        return f"已注册定时任务 {label}，首次触发时间：{time_str}"


class ListSchedulesTool(Tool):
    name = "list_schedules"
    description = "列出所有待执行的定时任务"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, service: SchedulerService) -> None:
        self._service = service

    async def execute(self, **kwargs: Any) -> str:
        jobs = self._service.list_jobs()
        if not jobs:
            return "当前没有待执行的定时任务"

        lines = [f"定时任务列表（共 {len(jobs)} 个）："]
        for job in jobs:
            try:
                fire_at_local = job.fire_at.astimezone(ZoneInfo(job.timezone))
                time_str = fire_at_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                time_str = job.fire_at.isoformat()

            label = f"「{job.name}」" if job.name else job.id[:8]
            if job.tier == "instant":
                action = (job.message or "")[:40]
            else:
                action = f"[AI] {(job.prompt or '')[:40]}"

            lines.append(
                f"• {label}  [{job.tier}/{job.trigger}]  "
                f"下次: {time_str}  "
                f"内容: {action}  "
                f"已运行: {job.run_count}次"
            )
        return "\n".join(lines)


class CancelScheduleTool(Tool):
    name = "cancel_schedule"
    description = "取消定时任务。可按任务 ID 或名称取消"
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "任务 ID 或其前缀（至少8位）",
            },
            "name": {
                "type": "string",
                "description": "任务名称",
            },
        },
    }

    def __init__(self, service: SchedulerService) -> None:
        self._service = service

    async def execute(self, **kwargs: Any) -> str:
        job_id = kwargs.get("id", "")
        name = kwargs.get("name", "")

        if not job_id and not name:
            return "错误：id 或 name 至少提供一个"

        if job_id:
            all_ids = list(self._service._jobs.keys())
            matches = [
                jid for jid in all_ids if jid == job_id or jid.startswith(job_id)
            ]
            if not matches:
                return f"未找到 ID 为 {job_id!r} 的任务"
            for jid in matches:
                self._service.cancel_job(jid)
            return f"已取消 {len(matches)} 个任务"

        if name:
            cancelled = self._service.cancel_job_by_name(name)
            if not cancelled:
                return f"未找到名称为 {name!r} 的任务"
            return f"已取消 {len(cancelled)} 个名为 {name!r} 的任务"

        return "未指定有效的取消条件"
