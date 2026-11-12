"""值班面板的数据：全部从已有的日志、事件文件和账本现读，不另存一份。"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

from plugins.ledger.contract import ContentPart, MessageReader, Output, ToolCall, ToolResult
from plugins.ui.contract import PluginUiRpcInvalidRequest

from . import recent
from .events import Event, clip, read_day

_VERDICT = re.compile(r"^- (\d\d:\d\d) #(\S+) \[(.+?)\] (.*?) → (wake|inbox|log)(?:｜(.*?))?（[^（）]*）$")
_ROUND = re.compile(r"^- (\d\d:\d\d) (.+?)：(.+?)（会话 (\S+)）$")
_FAILURE = re.compile(r"^- (\d\d:\d\d) 模型失败：(.*)$")
_MEMORY = re.compile(r"^- (\d\d:\d\d) 记忆改动：([+-]) (.*)$")


# 一天的值班记录按轮次归拢：同一分钟写下的判断、模型失败和主 Agent 的回合是同一轮。
def read_ticks(root: Path, day: date) -> list[dict[str, object]]:
    path = root / "log" / f"{day:%Y-%m-%d}.md"
    ticks: dict[str, dict[str, list[dict[str, str]]]] = {}
    for text in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        if match := _VERDICT.match(text):
            at, event_id, source, title, to, why = match.groups()
            row, item = "verdicts", {"id": event_id, "source": source, "title": title, "to": to, "why": why or ""}
        elif match := _ROUND.match(text):
            at, label, status, session = match.groups()
            row, item = "rounds", {"label": label, "status": status, "session": session}
        elif match := _MEMORY.match(text):
            at, row, item = match.group(1), "memory", {"mark": match.group(2), "text": match.group(3)}
        elif match := _FAILURE.match(text):
            at, row, item = match.group(1), "failures", {"text": match.group(2)}
        else:
            continue
        ticks.setdefault(at, {"verdicts": [], "rounds": [], "failures": [], "memory": []})[row].append(item)
    return [{"at": at, **rows} for at, rows in sorted(ticks.items())]


# 每个报过状态的来源一行：一段是相邻两条状态事件之间的时间，文字是事件原文。前一天最后一条带到当天零点。
def read_lanes(root: Path, day: date) -> list[dict[str, object]]:
    lanes: dict[str, list[dict[str, str]]] = {}
    carried: dict[str, Event] = {}
    for event in read_day(root, day - timedelta(days=1)):
        if event.state:
            carried[event.source] = event
    for source, event in carried.items():
        lanes[source] = [{"at": "00:00", "text": _state_text(event), "since": event.at}]
    for event in read_day(root, day):
        if event.state:
            lanes.setdefault(event.source, []).append({"at": event.at[11:16], "text": _state_text(event), "since": event.at})
    return [{"source": source, "segments": segments} for source, segments in sorted(lanes.items())]


def _state_text(event: Event) -> str:
    return clip(" — ".join(part for part in (event.title, event.text) if part), 200)


# 主对话当天的往来：用户说话和主动消息各在什么时刻。
def read_talk(reader: MessageReader, day: date, now: datetime) -> list[dict[str, str]]:
    hours = int((now - datetime.combine(day, datetime.min.time(), now.tzinfo)).total_seconds() // 3600) + 1
    rows: list[dict[str, str]] = []
    for turn in recent.read_turns(reader, now, hours=hours, scan=3000):
        local = turn.at.astimezone(now.tzinfo)
        if local.date() == day and turn.kind != "reply":
            rows.append({"at": f"{local:%H:%M}", "kind": turn.kind, "text": clip(turn.text, 400)})
    return rows


# 主 Agent 一个后台回合里的动线：按顺序列出工具调用、每次的结果摘要，以及它最后写下的话。
def read_round(reader: MessageReader, name: Callable[[str], str], pages: int = 5) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    try:
        page = reader.read_tail(limit=200)
    except KeyError:
        return steps
    messages = list(page.messages)
    while page.has_more and pages > 1 and page.messages:
        page, pages = reader.read_tail(before_seq=page.messages[0].seq, through_seq=page.through_seq, limit=200), pages - 1
        messages[:0] = page.messages
    for message in messages:
        body = message.body
        if isinstance(body, Output):
            for part in body.parts:
                if isinstance(part, ToolCall):
                    steps.append({"kind": "call", "tool": name(part.binding_id), "text": clip(_plain(part.arguments), 300)})
                elif isinstance(part, ContentPart) and part.kind == "text" and str(part.value).strip():
                    steps.append({"kind": "text", "tool": "", "text": clip(str(part.value).strip(), 600)})
        elif isinstance(body, ToolResult) and steps and steps[-1]["kind"] == "call":
            text = _output(" ".join(str(part.value) for part in body.parts if part.kind == "text"))
            steps[-1]["result"] = ("" if body.outcome == "success" else f"[{body.outcome}] ") + clip(" ".join(text.split()), 200)
    return steps


# 工具结果常是一个 JSON 对象：有 output 字段就只取它，读起来是命令的输出而不是一层包装。
def _output(text: str) -> str:
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return str(value["output"]) if isinstance(value, dict) and isinstance(value.get("output"), str) else text


def _plain(arguments: Mapping[str, object]) -> str:
    return "  ".join(f"{key}={' '.join(str(value).split())}" for key, value in arguments.items())


def parse_day(payload: Mapping[str, object], today: date) -> date:
    raw = payload.get("date")
    if raw is None:
        return today
    try:
        return date.fromisoformat(str(raw))
    except ValueError as error:
        raise PluginUiRpcInvalidRequest("date 需要是 YYYY-MM-DD") from error


def parse_session(payload: Mapping[str, object]) -> str:
    session = str(payload.get("session") or "")
    if not session.startswith("programmatic:watch-"):
        raise PluginUiRpcInvalidRequest("只能读取值班自己的后台会话")
    return session
