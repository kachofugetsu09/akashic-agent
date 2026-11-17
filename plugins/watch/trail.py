"""后台回合的足迹：每轮结束按实际发生的调用记一行，下一轮醒来时带上。每轮都是新会话，这是它记得自己做过什么的地方。"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from plugins.ledger.contract import Message, Output, ToolCall

from .events import clip

_HOST = re.compile(r"https?://(?:www\.)?([\w-]+(?:\.[\w-]+)*\.[a-z]{2,})", re.I)
# 反向域名写法的应用标识，整个参数值就是它才算
_APP = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,}")
_LINE = re.compile(r"^- \[\d{4}-(\d\d-\d\d) \d\d:\d\d\] .*?｜去过：([^｜]*)")
# 命令行里每一段开头的"程序 子命令"，只看第一行；shell 自己的关键字和翻文件用的基础命令不算去处
_COMMAND = re.compile(r"(?:^|&&|\|\||[;|])\s*([a-z][\w-]*) ([a-z][\w-]*)")
_PLAIN = frozenset("for do if then while function cat ls echo grep find mv cp rm head tail sed awk cd mkdir touch wc sort test ip ps which sleep date".split())
SHOWN, KEPT_DAYS = 12, 30


# 一个回合去过哪些地方：调用参数里出现的站点、应用和命令行里用到的程序，按第一次出现的顺序。不认具体是哪个工具。
def places(messages: Iterable[Message]) -> list[str]:
    seen: dict[str, None] = {}
    for message in messages:
        for part in message.body.parts if isinstance(message.body, Output) else ():
            if isinstance(part, ToolCall):
                for host in _HOST.findall(json.dumps(part.arguments, ensure_ascii=False)):
                    seen.setdefault(host.lower())
                for value in _strings(part.arguments):
                    if _APP.fullmatch(value):
                        seen.setdefault(value)
                command = part.arguments.get("command")
                for match in _COMMAND.finditer(command.split("\n", 1)[0] if isinstance(command, str) else ""):
                    if match.group(1) not in _PLAIN:
                        seen.setdefault(" ".join(match.groups()))
    return list(seen)


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


# 回合结束时追加一行。left 是面板同一份"留下了什么"：对用户说的话和改过的文件。
def record(root: Path, now: datetime, label: str, status: str, went: list[str], left: list[dict[str, str]]) -> None:
    said = [clip(" ".join(item["text"].split()), 80) for item in left if item["kind"] == "said"]
    files = list(dict.fromkeys(item["path"] for item in left if item["kind"] != "said"))
    parts = [f"- [{now:%Y-%m-%d %H:%M}] {label}" + ("" if status == "complete" else f"（{status}，没做完）"),
             "去过：" + ("、".join(went[:12]) or "没有出门")]
    parts += ["对他说：" + text for text in said]
    if files:
        parts.append("改了：" + "、".join(files[:8]))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "trail.md").open("a", encoding="utf-8") as file:
        file.write("｜".join(parts) + "\n")


# 给下一轮看的那一段：最近几轮各做了什么，加上这段时间各处去过几轮、最近是哪天。
def render(root: Path) -> str:
    path = root / "trail.md"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- [")] if path.exists() else []
    if not lines:
        return "你前几轮在后台做过的事：还没有记录。"
    seen: dict[str, tuple[int, str]] = {}
    for line in lines:
        if match := _LINE.match(line):
            for place in match.group(2).split("、"):
                if place and place != "没有出门":
                    seen[place] = (seen.get(place, (0, ""))[0] + 1, match.group(1))
    ranked = sorted(seen.items(), key=lambda item: -item[1][0])
    been = "、".join(f"{place}（{count} 轮，最近 {day}）" for place, (count, day) in ranked[:40]) or "哪里都没去过"
    return ("你前几轮在后台做过的事（系统按实际发生的调用记的，从旧到新；每一轮都是新的会话，这是你唯一记得的部分）：\n"
            + "\n".join(lines[-SHOWN:])
            + f"\n这 {KEPT_DAYS} 天里你去过的地方：{been}。没列在这里的，这段时间你一次都没去过。")
