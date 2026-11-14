"""WATCH.md：主 Agent 写给值班员的嘱托。一行一条，默认会过期，触发过的一次性嘱托自动归档。"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
# 已经报过的坏标记行：值班每 10 分钟读一次嘱托，同一行只报一次
_reported: set[str] = set()
TEMPLATE = """# 值班嘱托

写给值班员的话，一行一条，以 "- " 开头。行首可以带方括号标记：

- `[到 2026-12-01]` 到期日。不写会自动补上；到期没续的会移进 archive.md。
- `[长期]` 触发后保留。不写就是一次性的，值班员按它叫醒过一次就归档。
- `[每天 22:00]` `[工作日 22:00]` `[周末 10:00]` `[周三 09:00]` `[2026-11-12 10:30]` 到这个时间让值班员看一眼，再决定叫不叫醒。

## 叫醒我

## 留意
"""

ORDERS, NOTES = "叫醒我", "留意"
DEFAULT_DAYS = {ORDERS: 14, NOTES: 60}
_ITEM = re.compile(r"^- ((?:\[[^\]]*\]\s*)*)(.*)$")
_TAG = re.compile(r"\[([^\]]*)\]")
_UNTIL = re.compile(r"^到 (\d{4}-\d{2}-\d{2})$")
_CLOCK = re.compile(r"^(每天|工作日|周末|周[一二三四五六日]+) (\d{1,2}):(\d{2})$")
_ONCE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{1,2}):(\d{2})$")
_WEEK = "一二三四五六日"


@dataclass(frozen=True, slots=True)
class Order:
    raw: str
    section: str
    text: str
    until: date | None
    standing: bool
    when: str | None

    @property
    def recurring(self) -> bool:
        return self.when is not None and _ONCE.fullmatch(self.when) is None


# 逐行解析；认不出的方括号标记留在正文里原样给值班员看。
def parse(content: str) -> list[Order]:
    orders: list[Order] = []
    section = ""
    for raw in content.splitlines():
        if raw.startswith("## "):
            section = raw[3:].strip()
            continue
        match = _ITEM.fullmatch(raw.rstrip())
        if match is None or section not in DEFAULT_DAYS or not match.group(2).strip():
            continue
        until: date | None = None
        standing, when, unknown = False, None, ""
        for tag in _TAG.findall(match.group(1)):
            tag = tag.strip()
            expiry = _UNTIL.fullmatch(tag)
            if expiry is not None:
                try:
                    until = date.fromisoformat(expiry.group(1))
                except ValueError:
                    _report_bad_tag(raw, tag)
                    unknown += f"[{tag}] "
            elif tag == "长期":
                standing = True
            elif _is_time(tag):
                when = tag
            else:
                if _ONCE.fullmatch(tag) or _CLOCK.fullmatch(tag) or tag.startswith("到 "):
                    _report_bad_tag(raw, tag)
                unknown += f"[{tag}] "
        orders.append(Order(raw, section, unknown + match.group(2).strip(), until, standing, when))
    return orders


# 时间或到期标记写坏了：这条嘱托不会按时间触发；同一行内容只报一次。
def _report_bad_tag(raw: str, tag: str) -> None:
    if raw in _reported:
        return
    _reported.add(raw)
    logger.warning("[paused] 值班嘱托时间标记无法解析，这条嘱托不会按时间触发：标记=%s 原文=%s", tag[:40], raw[:80])


# 时间写错的标记（25:99、不存在的日期）不算查看时间，原样留在正文里给人看。
def _is_time(tag: str) -> bool:
    once, clock = _ONCE.fullmatch(tag), _CLOCK.fullmatch(tag)
    try:
        if once is not None:
            _ = datetime.fromisoformat(f"{once.group(1)}T{int(once.group(2)):02d}:{once.group(3)}")
            return True
        return clock is not None and int(clock.group(2)) < 24 and int(clock.group(3)) < 60
    except ValueError:
        return False


# 给值班员看的编号清单；编号只在这一次轻唤醒里有效。
def render(orders: list[Order]) -> tuple[str, list[Order]]:
    wake = [order for order in orders if order.section == ORDERS]
    notes = [order for order in orders if order.section == NOTES]
    lines = [f"{index}. {order.text}" + (f"（{order.when} 查看）" if order.when else "") for index, order in enumerate(wake, 1)]
    text = "什么情况叫醒主 Agent：\n" + ("\n".join(lines) or "（目前没有）")
    if notes:
        text += "\n另外请留意：\n" + "\n".join(f"- {order.text}" for order in notes)
    return text, wake


# 这条嘱托的查看时间是否落在 (after, now] 之内。
def due(order: Order, after: datetime, now: datetime) -> bool:
    if order.when is None:
        return False
    once = _ONCE.fullmatch(order.when)
    if once is not None:
        try:
            moment = datetime.fromisoformat(f"{once.group(1)}T{int(once.group(2)):02d}:{once.group(3)}").replace(tzinfo=now.tzinfo)
        except ValueError:
            return False
        return after < moment <= now
    clock = _CLOCK.fullmatch(order.when)
    assert clock is not None
    days = clock.group(1)
    day = after.date()
    while day <= now.date():
        allowed = days == "每天" or (days == "工作日" and day.weekday() < 5) or (days == "周末" and day.weekday() >= 5) \
            or (days.startswith("周") and _WEEK[day.weekday()] in days[1:])
        try:
            moment = datetime(day.year, day.month, day.day, int(clock.group(2)), int(clock.group(3)), tzinfo=now.tzinfo)
        except ValueError:
            return False
        if allowed and after < moment <= now:
            return True
        day += timedelta(days=1)
    return False


# 每次轻唤醒前整理一遍文件：没有就创建，缺到期日的补上，过期的和已触发的一次性嘱托移进归档。
def maintain(root: Path, today: date, fired: frozenset[str] = frozenset()) -> list[Order]:
    path = root / "WATCH.md"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE, encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    by_raw = {order.raw: order for order in parse(content)}
    kept: list[str] = []
    archived: list[str] = []
    for raw in content.splitlines():
        order = by_raw.get(raw)
        if order is None:
            kept.append(raw)
        elif order.until is not None and order.until < today:
            archived.append(f"- {today} 到期：{raw[2:]}")
        elif raw in fired and not (order.standing or order.recurring):
            archived.append(f"- {today} 已触发：{raw[2:]}")
        elif order.until is None:
            kept.append(f"- [到 {today + timedelta(days=DEFAULT_DAYS[order.section])}] {raw[2:]}")
        else:
            kept.append(raw)
    updated = "\n".join(kept) + "\n"
    if updated != content and path.read_text(encoding="utf-8") == content:
        path.write_text(updated, encoding="utf-8")
        if archived:
            with (root / "archive.md").open("a", encoding="utf-8") as file:
                file.write("\n".join(archived) + "\n")
    return parse(path.read_text(encoding="utf-8"))


# 三天内到期的行，复盘时请主 Agent 决定续不续。
def expiring(orders: list[Order], today: date, days: int = 3) -> list[Order]:
    return [order for order in orders if order.until is not None and order.until <= today + timedelta(days=days)]
