"""值班面板的数据：全部从已有的日志、事件文件和账本现读，不另存一份。"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

from plugins.ledger.contract import ContentPart, Message, MessageReader, Output, ToolCall, ToolResult
from plugins.models.contract import ModelCallStats
from plugins.ui.contract import PluginUiRpcInvalidRequest

from . import recent
from .events import Event, clip, read_day
from .watcher import ModelStats

_VERDICT = re.compile(r"^- (\d\d:\d\d) #(\S+) \[(.+?)\] (.*?) → (wake|inbox|log)(?:｜(.*?))?（[^（）]*）$")
_ROUND = re.compile(r"^- (\d\d:\d\d) (.+?)：(.+?)（会话 (\S+)）$")
_FAILURE = re.compile(r"^- (\d\d:\d\d) 模型失败：(.*)$")
_MEMORY = re.compile(r"^- (\d\d:\d\d) 记忆改动：([+-]) (.*)$")
_WATCHER_USAGE = re.compile(
    r"^- (\d\d:\d\d) 值班模型：(.+?)｜(\d+) 次｜(?:(用量未知)|输入 (\d+)(?:（命中 (\d+)(?:/(\d+))?）)?｜输出 (\d+))$"
)


# 一天的值班记录按轮次归拢：同一分钟写下的判断、值班用量、模型失败和主 Agent 的回合是同一轮。
def read_ticks(root: Path, day: date) -> list[dict[str, object]]:
    path = root / "log" / f"{day:%Y-%m-%d}.md"
    ticks: dict[str, dict[str, list[dict[str, object]]]] = {}
    watcher: dict[str, dict[str, ModelStats]] = {}
    for text in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        if match := _VERDICT.match(text):
            at, event_id, source, title, to, why = match.groups()
            row, item = "verdicts", dict[str, object](id=event_id, source=source, title=title, to=to, why=why or "")
        elif match := _ROUND.match(text):
            at, label, status, session = match.groups()
            row, item = "rounds", dict[str, object](label=label, status=status, session=session)
        elif match := _MEMORY.match(text):
            at, row, item = match.group(1), "memory", dict[str, object](mark=match.group(2), text=match.group(3))
        elif match := _FAILURE.match(text):
            at, row, item = match.group(1), "failures", dict[str, object](text=match.group(2))
        elif match := _WATCHER_USAGE.match(text):
            at = match.group(1)
            ticks.setdefault(at, {"verdicts": [], "rounds": [], "failures": [], "memory": []})
            stat = _parse_watcher_usage(match)
            bucket = watcher.setdefault(at, {})
            if stat.model in bucket:
                bucket[stat.model].merge(stat)
            else:
                bucket[stat.model] = stat
            continue
        else:
            continue
        ticks.setdefault(at, {"verdicts": [], "rounds": [], "failures": [], "memory": []})[row].append(item)
    return [
        {"at": at, **rows, "watcher": [item.to_dict() for item in watcher.get(at, {}).values()]}
        for at, rows in sorted(ticks.items())
    ]


def _parse_watcher_usage(match: re.Match[str]) -> ModelStats:
    _, model, raw_calls, unknown, raw_in, raw_hit, raw_hit_in, raw_out = match.groups()
    calls = int(raw_calls)
    if unknown:
        return ModelStats(model=model, calls=calls)
    input_tokens = int(raw_in)
    output_tokens = int(raw_out)
    if raw_hit is None:
        return ModelStats(model=model, calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, reported_calls=calls)
    hit_tokens = int(raw_hit)
    hit_input_tokens = int(raw_hit_in) if raw_hit_in is not None else input_tokens
    return ModelStats(
        model=model, calls=calls, input_tokens=input_tokens, hit_tokens=hit_tokens,
        hit_input_tokens=hit_input_tokens, output_tokens=output_tokens,
        reported_calls=calls, cache_reported_calls=calls,
    )


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def summarize_stats(items: Iterable[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, ModelStats] = {}
    total = ModelStats("total")
    for raw in items:
        model = str(raw["model"])
        stat = ModelStats(
            model=model,
            calls=_as_int(raw.get("calls")),
            input_tokens=_as_int(raw.get("input_tokens")),
            hit_tokens=_as_int(raw.get("hit_tokens")),
            hit_input_tokens=_as_int(raw.get("hit_input_tokens")),
            output_tokens=_as_int(raw.get("output_tokens")),
            reported_calls=_as_int(raw.get("reported_calls")),
            cache_reported_calls=_as_int(raw.get("cache_reported_calls")),
        )
        if model in grouped:
            grouped[model].merge(stat)
        else:
            grouped[model] = stat
        total.merge(stat)
    return {"models": [item.to_dict() for item in grouped.values()], "total": total.to_dict()}


def read_day_usage(
    root: Path, day: date, reader_for: Callable[[str], MessageReader], call_stats: Callable[[str], ModelCallStats],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ticks = read_ticks(root, day)
    watcher_items: list[Mapping[str, object]] = []
    agent_items: list[Mapping[str, object]] = []
    for tick in ticks:
        raw_watcher = tick["watcher"]
        if isinstance(raw_watcher, list):
            for entry in raw_watcher:
                if isinstance(entry, Mapping):
                    watcher_items.append(entry)
        raw_rounds = tick["rounds"]
        if isinstance(raw_rounds, list):
            for round_item in raw_rounds:
                if not isinstance(round_item, dict):
                    continue
                messages = _read_tail_messages(reader_for(str(round_item["session"])), 5)
                usage = _collect_usage(messages, call_stats)
                steps = _read_steps(messages, lambda _binding: "")
                # 回合收尾时它自己写的那段话就是这一轮的交代；列表里直接给出来，不用点进去才知道做了什么。
                round_item["said"] = next((step["text"] for step in reversed(steps) if step["kind"] == "text"), "")
                round_item["steps"] = sum(step["kind"] == "call" for step in steps)
                round_item["usage"] = usage
                models = usage.get("models")
                if isinstance(models, list):
                    for entry in models:
                        if isinstance(entry, Mapping):
                            agent_items.append(entry)
    return ticks, {"watcher": summarize_stats(watcher_items), "agent": summarize_stats(agent_items)}


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


def _read_tail_messages(reader: MessageReader, pages: int = 5) -> list[Message]:
    try:
        page = reader.read_tail(limit=200)
    except KeyError:
        return []
    messages = list(page.messages)
    while page.has_more and pages > 1 and page.messages:
        page, pages = reader.read_tail(before_seq=page.messages[0].seq, through_seq=page.through_seq, limit=200), pages - 1
        messages[:0] = page.messages
    return messages


def _collect_usage(messages: Iterable[Message], call_stats: Callable[[str], ModelCallStats]) -> dict[str, object]:
    grouped: dict[str, ModelStats] = {}
    total = ModelStats("total")
    for message in messages:
        body = message.body
        if not isinstance(body, Output):
            continue
        for part in body.parts:
            if not (isinstance(part, ContentPart) and part.kind == "model.facts" and isinstance(part.value, Mapping)):
                continue
            call_id = part.value.get("call_record_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            try:
                stats = call_stats(call_id)
            except (KeyError, ValueError):
                continue
            bucket = grouped.get(stats.model)
            if bucket is None:
                bucket = grouped[stats.model] = ModelStats(stats.model)
            bucket.add(stats.usage)
            total.add(stats.usage)
    return {"models": [item.to_dict() for item in grouped.values()], "total": total.to_dict()}


# 主 Agent 一个后台回合里的动线：按顺序列出工具调用、每次的结果摘要，以及它最后写下的话。
def read_round_detail(
    reader: MessageReader, name: Callable[[str], str], call_stats: Callable[[str], ModelCallStats], pages: int = 5,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    messages = _read_tail_messages(reader, pages)
    return _read_steps(messages, name), _collect_usage(messages, call_stats)


def _read_steps(messages: Iterable[Message], name: Callable[[str], str]) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for message in messages:
        body = message.body
        if isinstance(body, Output):
            for part in body.parts:
                if isinstance(part, ToolCall):
                    steps.append({"kind": "call", "tool": name(part.binding_id), "text": clip(_plain(part.arguments), 300)})
                elif isinstance(part, ContentPart) and part.kind == "text" and str(part.value).strip():
                    steps.append({"kind": "text", "tool": "", "text": str(part.value).strip()[:2400]})
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
