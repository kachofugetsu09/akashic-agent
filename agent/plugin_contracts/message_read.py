"""消息读取的公开结构合同：reader 端口与基于消息的纯判据。

`needs_reply`（来源是否应被唤醒）与 `input_origin`（原输入的渠道目的地）只读取
已提交消息。它们需要一个「消息读取器」的窄接口，而不应依赖 Core 的存储实现类，
因此本模块用 `MessageReaderPort` 描述该接口，并把两个判据放在这里。

`MessageReaderPort` 是纯 Protocol（只声明方法），因此 `isinstance` 可在运行时
区分「reader」与「消息序列」——这与原先对具体存储类做 isinstance 的语义一致：
真实的 `MessageReader` 结构上满足本 Protocol。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

from agent.plugin_contracts.conversation import check_origin
from agent.plugin_contracts.message import Control, Input, Message, Output


@runtime_checkable
class MessageReaderPort(Protocol):
    """一次会话的只读消息读取器（Core 消息服务的窄接口）。"""

    def head(self, *, source: str | None = None) -> int:
        """当前来源（或全部）的日志头 seq。"""
        ...

    def latest_input(self, source: str, *, through_seq: int) -> Message | None:
        """给定上界内该来源最近的一条 Input。"""
        ...

    def snapshot(
        self, *, after_seq: int = -1, through_seq: int | None = None
    ) -> tuple[Message, ...]:
        """按区间读取已提交消息。"""
        ...


def needs_reply(messages: Sequence[Message] | MessageReaderPort, source: str) -> bool:
    """来源从输入和控制事实决定是否唤醒；不依赖逻辑 Turn 或消费 cursor。"""
    # 最近 Input 之前的控制和终结只能覆盖更早的 seq，不影响本次唤醒。
    if isinstance(messages, MessageReaderPort):
        head = messages.head()
        latest = messages.latest_input(source, through_seq=head)
        if latest is None:
            return False
        messages = (latest, *messages.snapshot(after_seq=latest.seq, through_seq=head))
    boundary = -1
    latest_input = -1
    paused_through = -1
    for message in messages:
        if message.source != source:
            continue
        body = message.body
        if isinstance(body, Input):
            latest_input = message.seq
        elif isinstance(body, Output) and body.finish != "continue":
            boundary = message.seq
        elif isinstance(body, Control):
            if body.action == "abandon":
                boundary = max(boundary, body.through_seq)
            elif body.action in {"pause", "failure"}:
                paused_through = max(paused_through, body.through_seq)
            elif body.action == "resume" and body.through_seq >= paused_through:
                paused_through = -1
    return latest_input > max(boundary, paused_through)


def input_origin(
    reader: MessageReaderPort, source: str, *, through_seq: int
) -> tuple[str, str] | None:
    """只从原输入的已验证渠道事实读取目的地。"""
    previous = reader.latest_input(source, through_seq=through_seq)
    if previous is None:
        return None
    assert isinstance(previous.body, Input)
    parts = [part for part in previous.body.parts if part.kind == "channel.origin"]
    if not parts:
        return None
    if len(parts) != 1:
        raise ValueError("输入必须只有一个渠道来源")
    _ = check_origin(parts[0])
    value = cast(Mapping[str, str], parts[0].value)
    return value["channel"], value["chat_id"]
