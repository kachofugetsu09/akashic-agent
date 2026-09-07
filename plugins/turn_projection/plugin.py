from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agent.plugin_composition import Context, ServiceKey
from session.message import (
    CallRef,
    Input,
    Message,
    Output,
    ToolCall,
    ToolResult,
)

api_version = 3
name = "turn_projection"
version = "1.0.0"
desc = "从消息读取逻辑 Turn，不保存内容或消费进度"
inject = ()


@dataclass(frozen=True, slots=True)
class Turn:
    """一个日志区间的消息引用；不代表运行任务或持久化行。"""

    source: str
    after_seq: int
    through_seq: int
    ending_message_id: str | None
    status: Literal["open", "complete", "quiet", "abandoned"]
    message_ids: tuple[str, ...]
    observations: tuple[tuple[CallRef, str], ...]


def _build_turn(
    source: str,
    after_seq: int,
    through_seq: int,
    ending_message_id: str | None,
    status: Literal["open", "complete", "quiet", "abandoned"],
    messages: Sequence[Message],
) -> Turn:
    """分别返回对话主体与工具观察的引用，不复制消息正文。"""
    return Turn(
        source,
        after_seq,
        through_seq,
        ending_message_id,
        status,
        tuple(
            item.message_id
            for item in messages
            if isinstance(item.body, (Input, Output))
        ),
        tuple(
            (item.body.call_ref, item.message_id)
            for item in messages
            if isinstance(item.body, ToolResult)
        ),
    )


class TurnProjection:
    """对完整 Session 前缀分段；调用之间不保留任何状态。"""

    def project(self, messages: Sequence[Message], source: str) -> tuple[Turn, ...]:
        """按来源的最终回答或放弃边界分组，排除跨段晚到的工具结果。"""
        # 1. 调用者必须提供同一 Session 的有序前缀，不能把任意分页当新起点。
        if not messages:
            return ()
        session_id = messages[0].session_id
        previous_seq = -1
        seen: set[str] = set()
        for message in messages:
            if message.session_id != session_id or message.seq <= previous_seq:
                raise ValueError("Turn 投影要求同一 Session 按 seq 严格递增")
            if message.message_id in seen:
                raise ValueError("Turn 投影不能包含重复 message_id")
            previous_seq = message.seq
            seen.add(message.message_id)

        # 2. 正文与已归属的工具观察暂存于本次调用，闭段后只返回引用。
        turns: list[Turn] = []
        pending: list[Message] = []
        calls: set[CallRef] = set()
        after_seq = -1
        source_head = -1
        for message in messages:
            if message.source != source:
                continue
            source_head = message.seq
            body = message.body
            if isinstance(body, (Input, Output)):
                pending.append(message)
                if isinstance(body, Output):
                    calls.update(
                        CallRef(message.message_id, index)
                        for index, part in enumerate(body.parts)
                        if isinstance(part, ToolCall)
                    )
                    if body.finish != "continue":
                        turns.append(
                            _build_turn(
                                source,
                                after_seq,
                                message.seq,
                                message.message_id,
                                body.finish,
                                pending,
                            )
                        )
                        pending = []
                        calls = set()
                        after_seq = message.seq
            elif isinstance(body, ToolResult):
                if body.call_ref in calls:
                    pending.append(message)
            elif body.action == "abandon":
                if body.through_seq <= after_seq:
                    raise ValueError("abandon 不能重新关闭已经结束的前缀")
                closed = [item for item in pending if item.seq <= body.through_seq]
                pending = [item for item in pending if item.seq > body.through_seq]
                calls = {
                    CallRef(item.message_id, index)
                    for item in pending
                    if isinstance(item.body, Output)
                    for index, part in enumerate(item.body.parts)
                    if isinstance(part, ToolCall)
                }
                pending = [
                    item
                    for item in pending
                    if not isinstance(item.body, ToolResult)
                    or item.body.call_ref in calls
                ]
                if closed:
                    turns.append(
                        _build_turn(
                            source,
                            after_seq,
                            body.through_seq,
                            message.message_id,
                            "abandoned",
                            closed,
                        )
                    )
                after_seq = body.through_seq

        # 3. 暂停、失败和未回答输入只形成 open 尾段，不伪造成功结束点。
        if pending:
            turns.append(
                _build_turn(
                    source,
                    after_seq,
                    source_head,
                    None,
                    "open",
                    pending,
                )
            )
        return tuple(turns)


TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")


async def apply(ctx: Context, config: object) -> None:
    """仅提供普通消费能力；不打开数据库或启动后台任务。"""
    _ = await ctx.provide(TURN_PROJECTION, TurnProjection())
