"""Message 出处的唯一解释：作者身份由宿主在写入器绑定时固定，读取方只调用这里的判定。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from .contract import ContentPart, Control, Input, Message, Output, ToolResult


class Actor(StrEnum):
    """写出一条消息的主体种类；插件不再各自比较作者字符串。"""

    HUMAN = "human"
    PROGRAM = "program"
    MODEL = "model"
    TOOL = "tool"
    HOST = "host"
    AUTOMATION = "automation"


# 保留作者只表达主体种类；其余作者是写入插件自选的稳定名字，表示该插件自动产生的消息。
RESERVED_AUTHORS: Mapping[str, Actor] = {
    "user": Actor.HUMAN,
    "program": Actor.PROGRAM,
    "assistant": Actor.MODEL,
    "tool": Actor.TOOL,
    "app": Actor.HOST,
}
# 公开来源接纳的主体：人或外部程序代表自己发言，其 Input 开启一个对话 Turn。
PRINCIPALS = frozenset({Actor.HUMAN, Actor.PROGRAM})
LEGACY_SOURCE = "legacy-unattributed"
LEGACY_AUTHOR = "legacy-attribution-unknown"
# 对话 lane：经公开来源进入的人或程序对话，以及无法再细分的迁移历史；学习与对话上下文默认只读它们。
DIALOGUE_SOURCES: tuple[str, ...] = ("conversation", "programmatic", LEGACY_SOURCE)
_LEGACY_ROLES: Mapping[str, Actor] = {"user": Actor.HUMAN, "assistant": Actor.MODEL}


@dataclass(frozen=True, slots=True)
class Origin:
    """一条消息的出处：主体种类与实际写入它的插件或主体名。"""

    actor: Actor
    producer: str


# 解释任一已接纳消息的出处；迁移历史按保全的原始角色派生，不改写存储的作者。
def origin(message: Message) -> Origin:
    author = message.author
    actor = RESERVED_AUTHORS.get(author)
    if actor is not None:
        return Origin(actor, author)
    if author == LEGACY_AUTHOR and message.source == LEGACY_SOURCE:
        return Origin(_legacy_actor(message), author)
    return Origin(Actor.AUTOMATION, author)


def is_human_input(message: Message) -> bool:
    return isinstance(message.body, Input) and origin(message).actor is Actor.HUMAN


# 人或外部程序的 Input：召回、对话学习与回合起点只认它，自动输入不冒充主体发言。
def is_principal_input(message: Message) -> bool:
    return isinstance(message.body, Input) and origin(message).actor in PRINCIPALS


# 迁移消息只认一份已知格式的出处；格式不符说明迁移数据损坏，不能猜成某种主体。
def _legacy_actor(message: Message) -> Actor:
    body = message.body
    if isinstance(body, Control):
        return Actor.HOST
    parts = [part for part in body.parts
             if isinstance(part, ContentPart) and part.kind == "history.provenance"]
    if not parts:
        # 1. 没有原始角色时只按消息用途派生；无出处的 Input 不能被当作人的发言。
        return Actor.MODEL if isinstance(body, Output) else Actor.TOOL if isinstance(body, ToolResult) else Actor.AUTOMATION
    if len(parts) != 1 or not isinstance(parts[0].value, Mapping):
        raise ValueError("历史消息必须只有一个 provenance 对象")
    value = cast(Mapping[str, object], parts[0].value)
    role = value.get("role")
    if value.get("schema") != "sessions.messages.v0" or not isinstance(role, str) or role not in _LEGACY_ROLES:
        raise ValueError("历史 provenance 格式未知")
    return _LEGACY_ROLES[role]
