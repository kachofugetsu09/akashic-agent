"""时间线：他对它说的每一句话和每一次状态变化，按时间排在一起，按月一个文件，只追加、不删。
后台回合眼前只有最近一天的对话；更早的事、同一句话他说过几回、每回之前刚发生了什么，都到这里查。"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, tzinfo
from pathlib import Path

from .events import Event, clip
from .recent import Turn

LEAD = timedelta(minutes=30)


# 新的状态变化和他新说的话合成几行，按时间排好。他说的每一句后面带上之前半小时内最近的一次状态变化：
# 单独搜到这一行，也看得出他是在什么情形下说的。known 是用来对情形的状态事件，比 states 往回多带一段。
def rows(states: Sequence[Event], said: Sequence[Turn], known: Sequence[Event], zone: tzinfo) -> list[tuple[datetime, str]]:
    def change(event: Event) -> tuple[datetime, str]:
        return datetime.fromisoformat(event.at).astimezone(zone), f"[{event.source}] {clip(' — '.join(part for part in (event.title, event.text) if part), 80)}"
    changes = sorted(change(event) for event in known)
    found = [(at, f"- [{_stamp(at)}] 状态：{what}") for at, what in map(change, states)]
    for turn in said:
        at = turn.at.astimezone(zone)
        near = [(when, what) for when, what in changes if timedelta(0) <= at - when <= LEAD]
        found.append((at, f"- [{_stamp(at)}] 他说：{clip(turn.text, 300)}" + (f" 〔此前 {near[-1][0]:%H:%M} {near[-1][1]}〕" if near else "")))
    return sorted(found)


def append(root: Path, lines: Sequence[tuple[datetime, str]]) -> None:
    (root / "timeline").mkdir(parents=True, exist_ok=True)
    for at, text in lines:
        with (root / "timeline" / f"{at:%Y-%m}.md").open("a", encoding="utf-8") as file:
            _ = file.write(text + "\n")


def _stamp(at: datetime) -> str:
    return f"{at:%Y-%m-%d} 周{'一二三四五六日'[at.weekday()]} {at:%H:%M}"
