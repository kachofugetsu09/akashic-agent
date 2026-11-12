"""事件只有一种形状；任何来源的事件都变成一行，按到达日存进工作区 watch/events/。"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

_TEXT_KEYS = ("text", "content", "body", "message", "summary", "description")
_TIME_KEYS = ("at", "time", "published_at", "observed_at", "mail_date")
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T")


@dataclass(frozen=True, slots=True)
class Event:
    id: str
    source: str
    at: str
    title: str
    text: str
    url: str
    # 这条是该来源的当前状态：照常是一条事件，同时顶替这个来源的上一条状态。
    state: bool = False
    # 来源自己的稳定标识；同一来源同一个 key 只记一次。
    key: str = ""

    @property
    def number(self) -> int:
        return int(self.id[1:])


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# 一行给模型看的事件摘要；全文用 read_event 取。
def line(event: Event, limit: int = 220) -> str:
    body = " — ".join(part for part in (clip(event.title, 120), clip(event.text, limit)) if part)
    return f"#{event.id} [{event.source}] {event.at[5:16].replace('T', ' ')}  {body}"


# 把来源给的松散数据收成事件字段：认得常见别名，认不出的字段整体当作正文。
def normalize(number: int, source: str, raw: Mapping[str, object] | str, now: datetime, state: bool = False, key: str = "") -> Event:
    if isinstance(raw, str):
        raw = {"text": raw}
    given = raw.get("key")
    title = raw.get("title")
    text = next((raw[key] for key in _TEXT_KEYS if isinstance(raw.get(key), str) and raw[key]), None)
    url = raw.get("url") or raw.get("link")
    at = next((raw[key] for key in _TIME_KEYS if isinstance(raw.get(key), str) and raw[key]), None)
    if not isinstance(title, str) and text is None:
        text = _flatten(raw)
    title = title if isinstance(title, str) else ""
    text = text if isinstance(text, str) else ""
    if title and text.startswith(title.rstrip("…")):
        title = ""
    return Event(f"e{number}", source[:80], _instant(at, now), title[:500], text[:20000], url[:2000] if isinstance(url, str) else "",
                 state or raw.get("state") is True, (key or (given if isinstance(given, str) else ""))[:300])


# 没有标题和正文的结构化数据：简单的字段先写成 "名: 值"，时间戳和嵌套的部分排在后面，一行摘要里先看到要紧的。
def _flatten(raw: Mapping[str, object]) -> str:
    plain = {key: value for key, value in raw.items() if isinstance(value, (str, int, float)) and value != "" and key not in _TIME_KEYS and key != "key" and (key, value) != ("state", True)}
    stamps = {key for key, value in plain.items() if isinstance(value, str) and _STAMP.match(value)}
    nested = {key: value for key, value in raw.items() if isinstance(value, (dict, list)) and value}
    parts = [f"{key}: {value}" for key, value in plain.items() if key not in stamps] + [f"{key}: {plain[key]}" for key in plain if key in stamps]
    if nested:
        parts.append(json.dumps(nested, ensure_ascii=False, default=str))
    return " · ".join(parts) or json.dumps(raw, ensure_ascii=False, default=str)


def _instant(value: object, now: datetime) -> str:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return now.isoformat(timespec="seconds")
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=now.tzinfo)).astimezone(now.tzinfo).isoformat(timespec="seconds")


class EventLog:
    """按天追加的事件文件；编号全局递增，值班与主 Agent 都用编号指认事件。"""

    def __init__(self, root: Path) -> None:
        self._dir = root / "events"

    def append(self, events: Sequence[Event], now: datetime) -> None:
        if not events:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        with (self._dir / f"{now:%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    # 已落盘的最大编号。状态文件可能比事件文件旧（保存失败、崩溃），启动时以这里为准。
    def last_number(self) -> int:
        if not self._dir.exists():
            return 0
        return max((event.number for path in self._dir.glob("*.jsonl") for event in _read(path)), default=0)

    # 保留期内每个 (来源, key) 对应的事件编号；启动时读一次，用来认出重复交来的事件。
    def keys(self) -> dict[tuple[str, str], str]:
        if not self._dir.exists():
            return {}
        return {(event.source, event.key): event.id for path in sorted(self._dir.glob("*.jsonl")) for event in _read(path) if event.key}

    # 读出编号大于 after 的事件；范围和事件的保留期一致，积压几天的未判事件也不会被跳过。
    def since(self, after: int, now: datetime, days: int = 14) -> list[Event]:
        found: list[Event] = []
        for offset in range(days, -1, -1):
            path = self._dir / f"{now - timedelta(days=offset):%Y-%m-%d}.jsonl"
            if path.exists():
                found.extend(event for event in _read(path) if event.number > after)
        return found

    def get(self, event_id: str, now: datetime, days: int = 14) -> Event | None:
        for offset in range(days + 1):
            path = self._dir / f"{now - timedelta(days=offset):%Y-%m-%d}.jsonl"
            if path.exists():
                for event in _read(path):
                    if event.id == event_id:
                        return event
        return None

    def prune(self, now: datetime, keep_days: int) -> None:
        if not self._dir.exists():
            return
        floor = f"{now - timedelta(days=keep_days):%Y-%m-%d}"
        for path in self._dir.glob("*.jsonl"):
            if path.stem < floor:
                path.unlink()


def _read(path: Path) -> list[Event]:
    events: list[Event] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(Event(**json.loads(raw)))
        except (ValueError, TypeError):
            continue
    return events


# 某一天的全部事件，按写入顺序。
def read_day(root: Path, day: date) -> list[Event]:
    path = root / "events" / f"{day:%Y-%m-%d}.jsonl"
    return _read(path) if path.exists() else []


_PLACE = re.compile(r"^- (.+?)[:：]\s*(-?\d{1,3}\.\d+)\s*[,，]\s*(-?\d{1,3}\.\d+)")
_COORD = re.compile(r"\(?(-?\d{1,3}\.\d{3,})\s*[,，]\s*(-?\d{1,3}\.\d{3,})\)?")


# watch/places.md 里主 Agent 认出来的地点，一行一个："- 名字: 纬度,经度"。
def load_places(root: Path) -> list[tuple[str, float, float]]:
    path = root / "places.md"
    if not path.exists():
        return []
    found = (_PLACE.match(line) for line in path.read_text(encoding="utf-8").splitlines())
    return [(match.group(1).strip(), float(match.group(2)), float(match.group(3))) for match in found if match is not None]


# 事件正文里的坐标落在认识的地点三百米内，就在后面标上名字；值班员和主 Agent 读到的是"公司""家"，不是一对数字。
def name_places(event: Event, places: Sequence[tuple[str, float, float]]) -> Event:
    def label(match: re.Match[str]) -> str:
        lat, lon = float(match.group(1)), float(match.group(2))
        near = min(places, key=lambda place: math.hypot(place[1] - lat, (place[2] - lon) * math.cos(math.radians(lat))))
        metres = math.hypot(near[1] - lat, (near[2] - lon) * math.cos(math.radians(lat))) * 111_000
        return match.group(0) + (f"【{near[0]}】" if metres <= 300 else "")
    return replace(event, text=_COORD.sub(label, event.text), title=_COORD.sub(label, event.title)) if places else event


# 来源交来的一行：是 JSON 对象就按字段取，否则整行当正文。
def parse_line(text: str) -> Mapping[str, object] | str:
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value if isinstance(value, dict) else str(value)


# 读取 watch/sources/*.jsonl 里新追加的整行；文件名就是来源名，删掉文件就是移除来源。
def read_sources(root: Path, offsets: dict[str, int], limit: int = 500) -> list[tuple[str, Mapping[str, object] | str]]:
    rows: list[tuple[str, Mapping[str, object] | str]] = []
    directory = root / "sources"
    if not directory.exists():
        offsets.clear()
        return rows
    present = {path.stem: path for path in sorted(directory.glob("*.jsonl"))}
    # 改名暂停的来源（名字.jsonl.后缀）保留游标，恢复后接着读；文件彻底没了才清掉
    paused = {path.name.split(".jsonl.")[0] for path in directory.glob("*.jsonl.*")}
    for gone in set(offsets) - set(present) - paused:
        del offsets[gone]
    for name, path in present.items():
        start = offsets.get(name, 0)
        with path.open("rb") as file:
            # 文件被截短或重写时从头读起
            if path.stat().st_size < start:
                start = 0
            file.seek(start)
            for raw in file:
                if not raw.endswith(b"\n") or len(rows) >= limit:
                    break
                start += len(raw)
                text = raw.decode("utf-8", "replace").strip()
                if not text:
                    continue
                rows.append((name, parse_line(text)))
        offsets[name] = start
    return rows
