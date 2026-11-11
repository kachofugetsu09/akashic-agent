"""最近对话视图：每次唤醒时从消息账本现查，不落任何副本。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from plugins.ledger.contract import DIALOGUE_SOURCES, ContentPart, Input, MessageReader, Output, is_user_input

from .events import clip

Kind = Literal["user", "reply", "push"]


@dataclass(frozen=True, slots=True)
class Turn:
    at: datetime
    kind: Kind
    text: str
    source: str


# 从会话尾部往回读到时间窗边界；只留用户原话、对话回复和不经用户发起的主动消息。
def read_turns(reader: MessageReader, now: datetime, hours: int = 72, scan: int = 800) -> list[Turn]:
    turns: list[Turn] = []
    floor = now - timedelta(hours=hours)
    try:
        page = reader.read_tail(limit=100)
    except KeyError:
        return turns
    seen = 0
    while True:
        for message in reversed(page.messages):
            seen += 1
            if message.recorded_at < floor:
                return list(reversed(turns))
            body = message.body
            if message.recorded_at > now or not isinstance(body, (Input, Output)):
                continue
            text = "\n".join(str(part.value) for part in body.parts if isinstance(part, ContentPart) and part.kind == "text").strip()
            if not text:
                continue
            if is_user_input(message):
                turns.append(Turn(message.recorded_at, "user", text, message.source))
            elif isinstance(body, Output) and body.finish != "continue":
                kind: Kind = "reply" if message.source in DIALOGUE_SOURCES else "push"
                turns.append(Turn(message.recorded_at, kind, text, message.source))
        if not page.has_more or seen >= scan:
            return list(reversed(turns))
        page = reader.read_tail(before_seq=page.messages[0].seq, through_seq=page.through_seq, limit=100)


# 渲染成一条时间线：用户原话给得最长，回复只留开头，主动消息带上用户有没有接话。
def render(turns: list[Turn], now: datetime, budget: int = 2600) -> str:
    lines: list[str] = []
    for index, turn in enumerate(turns):
        stamp = f"{turn.at.astimezone(now.tzinfo):%m-%d %H:%M}"
        if turn.kind == "user":
            lines.append(f"{stamp} 用户：{clip(turn.text, 220)}")
        elif turn.kind == "reply":
            if index and turns[index - 1].kind == "reply":
                continue
            lines.append(f"{stamp} （回复）{clip(turn.text, 70)}")
        else:
            lines.append(f"{stamp} （主动消息）{clip(turn.text, 90)} 〔{_reaction(turns, index)}〕")
    kept: list[str] = []
    used = 0
    for text in reversed(lines):
        if used + len(text) > budget:
            break
        kept.append(text)
        used += len(text)
    return "\n".join(reversed(kept)) or "（这段时间没有对话）"


def _reaction(turns: list[Turn], index: int) -> str:
    later = turns[index + 1:]
    answer = next((turn for turn in later if turn.kind == "user"), None)
    if answer is None:
        return "之后用户还没说过话"
    between = sum(1 for turn in later if turn.kind == "push" and turn.at < answer.at)
    minutes = int((answer.at - turns[index].at).total_seconds() // 60)
    return f"{minutes} 分钟后用户说话" if between == 0 else f"用户 {minutes} 分钟后才说话，中间又有 {between} 条主动消息"


# 主动消息被接话的比例，给值班员和主 Agent 掂量现在开口合不合适。
def stats(turns: list[Turn], now: datetime) -> str:
    def count(hours: int) -> tuple[int, int]:
        total = answered = 0
        for index, turn in enumerate(turns):
            if turn.kind != "push" or turn.at < now - timedelta(hours=hours):
                continue
            total += 1
            following = turns[index + 1] if index + 1 < len(turns) else None
            if following is not None and following.kind == "user" and following.at - turn.at < timedelta(hours=2):
                answered += 1
        return total, answered
    day, day_answered = count(24)
    window, window_answered = count(72)
    last = next((turn for turn in reversed(turns) if turn.kind == "user"), None)
    quiet = "三天以上" if last is None else f"{int((now - last.at).total_seconds() // 60)} 分钟前"
    return (f"过去 24 小时主动消息 {day} 条，{day_answered} 条在两小时内被接话；过去 3 天 {window} 条，{window_answered} 条被接话。"
            f"用户上一次说话：{quiet}。")
