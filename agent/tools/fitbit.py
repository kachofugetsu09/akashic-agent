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
    """获取当前健康快照（心率 / 最近一次睡眠血氧 / 步数 / 睡眠状态）"""

    name = "fitbit_health_snapshot"
    description = (
        "获取用户当前健康状态快照，包括：当前心率（bpm）、最近一次睡眠 SpO₂、"
        "今日步数、睡眠状态（sleeping/awake）及睡眠概率。"
        "其中 SpO₂ 按 Fitbit 官方语义视为最近一次睡眠估算值，不是白天实时血氧。"
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
                f"{self._url}/api/tool/fitbit_health_snapshot",
                budget=RequestBudget(total_timeout_s=5.0),
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return f"[fitbit_health_snapshot] 无法连接 Fitbit monitor：{e}"

        sleep_24h: dict[str, Any] = {}
        reason = ""
        since = None
        try:
            r_detail = await self._requester.get(
                f"{self._url}/api/data",
                budget=RequestBudget(total_timeout_s=5.0),
            )
            r_detail.raise_for_status()
            detail_data = r_detail.json()
            if isinstance(detail_data, dict):
                sleep_obj = detail_data.get("sleep")
                if isinstance(sleep_obj, dict):
                    reason = str(sleep_obj.get("reason") or "").strip()
                    since = sleep_obj.get("since")
        except Exception:
            reason = ""
            since = None

        try:
            if isinstance(data, dict):
                sleep_24h_obj = data.get("sleep_24h")
                if isinstance(sleep_24h_obj, dict):
                    sleep_24h = sleep_24h_obj
        except Exception:
            sleep_24h = {}

        hr = data.get("heart_rate")
        spo2 = data.get("spo2")
        steps = data.get("steps")
        state = data.get("sleep_state", "unknown")
        prob = data.get("sleep_prob")
        lag = data.get("data_lag_min")
        hr_time = None
        source = ""
        spo2_time = data.get("latest_sleep_spo2_time")
        spo2_lag = data.get("spo2_lag_min")
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
        if spo2 is not None:
            spo2_line = f"最近一次睡眠血氧：{spo2:.1f}%"
            if spo2_time:
                spo2_line += f"（数据时间 {spo2_time}"
                if spo2_lag is not None:
                    spo2_line += f"，距今约 {spo2_lag} 分钟"
                spo2_line += "）"
            lines.append(spo2_line)
        else:
            lines.append("最近一次睡眠血氧：无数据")
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

        lines.append("注：心率/步数是当前缓存；SpO₂ 表示最近一次睡眠估算值，不代表白天实时血氧。")
        result = {
            "available": True,
            "data_lag_min": lag,
            "spo2_lag_min": spo2_lag,
            "last_updated": updated,
            "heart_rate": hr,
            "spo2": spo2,
            "latest_sleep_spo2": spo2,
            "latest_sleep_spo2_time": spo2_time,
            "steps": steps,
            "sleep_state": state,
            "sleep_prob": prob,
            "sleep_24h": sleep_24h,
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
