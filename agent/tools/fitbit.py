"""
Fitbit 健康数据工具

依赖本地运行的 fitbit-monitor 服务（默认 http://127.0.0.1:18765）。
服务不可达时返回友好错误，不影响 agent 其他功能。
"""

from __future__ import annotations

import json
from typing import Any

from agent.tools.base import Tool
from core.net.http import (
    HttpRequester,
    RequestBudget,
    get_default_http_requester,
)


def _fmt_duration(minutes: int | None) -> str:
    if minutes is None:
        return "—"
    h, m = divmod(int(minutes), 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


class FitbitHealthSnapshotTool(Tool):
    """获取当前健康快照（心率 / 血氧 / 步数 / 睡眠状态）"""

    name = "fitbit_health_snapshot"
    description = (
        "获取用户当前实时健康状态快照，包括：当前心率（bpm）、血氧（SpO₂）、"
        "今日步数、睡眠状态（sleeping/awake/uncertain）及睡眠概率。"
        "数据来自本地 Fitbit monitor 缓存，含 Fitbit 设备同步延迟约 15-30 分钟。"
        "适用于：用户询问当前状态、agent 判断是否适合打扰、了解用户能量水平。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(
        self,
        monitor_url: str = "http://127.0.0.1:18765",
        requester: HttpRequester | None = None,
    ) -> None:
        self._url = monitor_url.rstrip("/")
        self._requester = requester or get_default_http_requester("local_service")

    def with_requester(self, requester: HttpRequester) -> "FitbitHealthSnapshotTool":
        self._requester = requester
        return self

    async def execute(self, **kwargs: Any) -> str:
        try:
            r = await self._requester.get(
                f"{self._url}/api/data",
                budget=RequestBudget(total_timeout_s=5.0),
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return f"[fitbit_health_snapshot] 无法连接 Fitbit monitor：{e}"

        summary = data.get("summary", {})
        sleep = data.get("sleep", {})
        signals = data.get("signals", {}) or {}
        meta = data.get("data_meta", {}) or {}

        hr = summary.get("heart_rate")
        spo2 = summary.get("spo2")
        steps = summary.get("steps")
        state = sleep.get("state", "unknown")
        reason = sleep.get("reason", "")
        since = sleep.get("since")
        prob = signals.get("sleep_prob")
        source = signals.get("prob_source", "")
        lag = meta.get("data_lag_min")
        hr_time = meta.get("latest_hr_time")
        updated = data.get("last_updated", "")

        lines = [f"【Fitbit 健康快照】{updated}"]
        lines.append(
            f"心率：{'%d bpm' % hr if hr else '无数据'}"
            + (
                f"（数据时间 {hr_time}，延迟约 {lag} 分钟）"
                if hr_time and lag is not None
                else ""
            )
        )
        lines.append(f"血氧：{'%.1f%%' % spo2 if spo2 else '无数据'}")
        lines.append(
            f"今日步数：{int(steps):,} 步" if steps is not None else "今日步数：无数据"
        )

        prob_str = f"{prob:.0%}" if prob is not None else "—"
        source_str = f"，{source}模型" if source and source != "unavailable" else ""
        lines.append(f"睡眠状态：{state}（概率 {prob_str}{source_str}）")
        if reason:
            lines.append(f"  原因：{reason}")
        if since:
            lines.append(f"  持续自：{since}")

        lines.append("注：Fitbit 数据含设备→手机→云端同步延迟，约 15-30 分钟。")
        result = {
            "available": True,
            "data_lag_min": lag,
            "last_updated": updated,
            "heart_rate": hr,
            "spo2": spo2,
            "steps": steps,
            "sleep_state": state,
            "sleep_prob": prob,
        }
        summary_text = "\n".join(lines)
        return json.dumps(result, ensure_ascii=False) + "\n\n" + summary_text


class FitbitSleepReportTool(Tool):
    """获取最近 N 天的睡眠质量报告（时长、效率、深睡、REM、HRV）"""

    name = "fitbit_sleep_report"
    description = (
        "获取用户最近 N 天的睡眠质量报告，包含每晚：入睡/起床时间、"
        "总时长、效率、深睡/REM/浅睡分钟数、HRV（心率变异性，反映恢复质量）。"
        "适用于：用户询问睡眠质量、分析疲劳/压力状态、了解作息规律。"
        "数据直接来自 Fitbit 云端，有 1 天延迟（今天的数据明天才完整）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "查询最近几天，默认 7，最大 30",
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        monitor_url: str = "http://127.0.0.1:18765",
        requester: HttpRequester | None = None,
    ) -> None:
        self._url = monitor_url.rstrip("/")
        self._requester = requester or get_default_http_requester("local_service")

    def with_requester(self, requester: HttpRequester) -> "FitbitSleepReportTool":
        self._requester = requester
        return self

    async def execute(self, **kwargs: Any) -> str:
        days = int(kwargs.get("days", 7))

        try:
            r = await self._requester.get(
                f"{self._url}/api/sleep_report",
                params={"days": days},
                budget=RequestBudget(total_timeout_s=20.0),
                timeout_s=20.0,
            )
            if r.status_code == 401:
                return "[fitbit_sleep_report] Fitbit 未授权，请先完成 OAuth 授权。"
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return f"[fitbit_sleep_report] 无法连接 Fitbit monitor：{e}"

        sm = data.get("summary", {})
        entries = data.get("days", [])

        lines = [f"【Fitbit 睡眠报告 · 最近 {days} 天】"]

        # 每日明细
        for d in reversed(entries):  # 最新在前
            date_str = d.get("date", "")
            if d.get("no_data"):
                hrv_str = f"  HRV {d['hrv_ms']} ms" if d.get("hrv_ms") else ""
                lines.append(f"{date_str}  无睡眠记录{hrv_str}")
                continue

            start = d.get("start_time") or "—"
            end = d.get("end_time") or "—"
            dur = _fmt_duration(d.get("duration_min"))
            eff = d.get("efficiency")
            deep = _fmt_duration(d.get("deep_min"))
            rem = _fmt_duration(d.get("rem_min"))
            light = _fmt_duration(d.get("light_min"))
            wake = _fmt_duration(d.get("wake_min"))
            hrv = d.get("hrv_ms")

            eff_str = f"  效率 {eff}%" if eff is not None else ""
            hrv_str = f"  HRV {hrv} ms" if hrv is not None else ""
            lines.append(
                f"{date_str}  {start}→{end}  {dur}{eff_str}\n"
                f"  深睡 {deep}  REM {rem}  浅睡 {light}  清醒 {wake}{hrv_str}"
            )

        # 均值摘要
        lines.append("─" * 36)
        avg_dur = _fmt_duration(sm.get("avg_duration_min"))
        avg_eff = sm.get("avg_efficiency")
        avg_deep = _fmt_duration(sm.get("avg_deep_min"))
        avg_rem = _fmt_duration(sm.get("avg_rem_min"))
        avg_hrv = sm.get("avg_hrv_ms")
        valid_n = sm.get("days_with_data", 0)

        eff_str = f"  效率 {avg_eff}%" if avg_eff is not None else ""
        hrv_str = f"  HRV {avg_hrv} ms" if avg_hrv is not None else ""
        lines.append(
            f"{valid_n}/{days} 天有数据  均值：时长 {avg_dur}{eff_str}\n"
            f"  深睡 {avg_deep}  REM {avg_rem}{hrv_str}"
        )

        return "\n".join(lines)
