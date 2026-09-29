from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from agent.plugin_composition import Context
from agent.plugin_contracts import CallRef, Input, Message, Output, ToolCall, ToolResult
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
)

api_version = 3
name = "turn_projection"
version = "1.0.0"
desc = "从消息读取逻辑 Turn，不保存内容或消费进度"
inject = ()


@dataclass(frozen=True, slots=True)
class _Member:
    """投影只保留顺序和引用，正文随读页释放。"""

    seq: int
    message_id: str
    call_ref: CallRef | None = None
    calls: tuple[CallRef, ...] = ()


def _build_turn(
    source: str,
    after_seq: int,
    through_seq: int,
    ending_message_id: str | None,
    status: Literal["open", "complete", "quiet", "abandoned"],
    messages: Sequence[_Member],
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
            if item.call_ref is None
        ),
        tuple(
            (item.call_ref, item.message_id)
            for item in messages
            if item.call_ref is not None
        ),
    )


class TurnProjection:
    """对有序消息流分段；消费进度属于调用方，正文不留在投影中。"""

    def project(
        self, messages: Iterable[Message], source: str, *,
        after_seq: int = -1,
    ) -> tuple[Turn, ...]:
        """从起点或已提交的闭合 Turn 边界投影，排除跨段晚到结果。"""
        # 1. 增量起点必须是已闭合边界；它的 abandon 控制可位于边界之后。
        consumed_through = after_seq
        session_id: str | None = None
        previous_seq = after_seq
        seen: set[str] = set()
        turns: list[Turn] = []
        pending: list[_Member] = []
        calls: set[CallRef] = set()
        source_head = after_seq
        for message in messages:
            if session_id is None:
                session_id = message.session_id
            if message.session_id != session_id or message.seq <= previous_seq:
                raise ValueError("Turn 投影要求同一 Session 按 seq 严格递增")
            if message.message_id in seen:
                raise ValueError("Turn 投影不能包含重复 message_id")
            previous_seq = message.seq
            seen.add(message.message_id)
            if message.source != source:
                continue
            source_head = message.seq
            body = message.body
            if isinstance(body, (Input, Output)):
                refs = tuple(
                    CallRef(message.message_id, index)
                    for index, part in enumerate(body.parts)
                    if isinstance(part, ToolCall)
                ) if isinstance(body, Output) else ()
                pending.append(_Member(message.seq, message.message_id, calls=refs))
                if isinstance(body, Output):
                    calls.update(refs)
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
                    pending.append(_Member(message.seq, message.message_id, body.call_ref))
            elif body.action == "abandon":
                if body.through_seq <= consumed_through:
                    continue
                if body.through_seq <= after_seq:
                    raise ValueError("abandon 不能重新关闭已经结束的前缀")
                closed = [item for item in pending if item.seq <= body.through_seq]
                pending = [item for item in pending if item.seq > body.through_seq]
                calls = {
                    ref for item in pending for ref in item.calls
                }
                pending = [
                    item
                    for item in pending
                    if item.call_ref is None or item.call_ref in calls
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


async def apply(ctx: Context) -> None:
    """仅提供普通消费能力；不打开数据库或启动后台任务。"""
    _ = await ctx.provide(TURN_PROJECTION, TurnProjection())
