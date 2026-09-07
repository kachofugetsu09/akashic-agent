from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from agent.plugin_composition import Context
from plugins.delivery.api import Sink
from plugins.delivery.execution import Deliveries
from session.log import MessageCatalog, MessageReader
from session.message import Message

logger = logging.getLogger(__name__)
Select = Callable[[MessageReader, Message], tuple[Sink, ...] | None]


@dataclass(slots=True)
class _Wake:
    changed: bool = True


async def follow(
    ctx: Context, catalog: MessageCatalog,
    execution: Callable[[], Deliveries], select: Select,
    *, settled: Callable[[str, str], None] | None = None,
) -> None:
    """按 seq 固定选路；重启追赶 prepared，各目的地与各 Session 独立结算。"""
    active: dict[str, _Wake] = {}
    recovery: dict[str, dict[str, list[str]]] = {}

    async def send(message_id: str, sink: str) -> None:
        try:
            async with ctx.runtime_scope():
                receipt = await execution().send(message_id, sink)
            if receipt.status != "delivered":
                logger.warning("发送尚未确认 message=%s sink=%s status=%s error=%s",
                               message_id, sink, receipt.status, receipt.error)
        except Exception:
            # 一个目的地失败不能取消另一处已开始的效果；回执仍由 Delivery 保留。
            logger.exception("发送失败，保留原效果等待恢复 message=%s sink=%s", message_id, sink)
        finally:
            if settled is not None:
                settled(message_id, sink)

    async def send_all(message_id: str, sinks: tuple[str, ...], attempted: set[tuple[str, str]]) -> None:
        async with asyncio.TaskGroup() as group:
            for sink in sinks:
                key = (message_id, sink)
                if key not in attempted:
                    attempted.add(key)
                    _ = group.create_task(send(message_id, sink))

    async def drive(session_id: str, wake: _Wake) -> None:
        """单个 Session 保持消息顺序；实际发送失败不阻止后续消息固定选路。"""
        try:
            attempted: set[tuple[str, str]] = set()
            reader = catalog.reader(session_id)
            # 1. 恢复效果从原消息 seq 排序，不使用随机 message_id 的字典顺序。
            pending = recovery.pop(session_id, {})
            ordered: list[tuple[int, str]] = []
            for message_id in pending:
                message = reader.get(message_id)
                if message is None:
                    raise ValueError("待恢复发送的原消息缺失")
                ordered.append((message.seq, message_id))
            for _, message_id in sorted(ordered):
                await send_all(message_id, tuple(pending[message_id]), attempted)

            # 2. 新选择与全部 prepared、cursor 同事务，I/O 才可以开始。
            while wake.changed:
                wake.changed = False
                while True:
                    async with ctx.runtime_scope():
                        delivery = execution()
                        messages = reader.read(after_seq=delivery.cursor(session_id), limit=100)
                    if not messages:
                        break
                    for message in messages:
                        async with ctx.runtime_scope():
                            delivery = execution()
                            existing = delivery.selection(message.message_id)
                            sinks = select(reader, message) if existing is None else ()
                            selected = delivery.consume(reader, message, sinks, passive=True)
                        if selected is not None:
                            await send_all(message.message_id, selected.sinks, attempted)
        except Exception:
            # 选路失败不推进该 Session cursor；其他 Session 仍可独立接纳和发送。
            logger.exception("发送消费停止，保留原消息等待修复 session=%s", session_id)
        finally:
            del active[session_id]

    previous: dict[str, int] = {}
    first = True
    async with asyncio.TaskGroup() as group:
        async for heads in catalog.follow():
            # 先建立日志订阅再取耐久 pending；恢复后不用扫描当前策略重选旧路由。
            if first:
                async with ctx.runtime_scope():
                    delivery = execution()
                    for message_id, sink in delivery.pending():
                        selection = delivery.selection(message_id)
                        if selection is None:
                            raise ValueError("待恢复发送缺少首次选择")
                        session = recovery.setdefault(selection.session_id, {})
                        session.setdefault(message_id, []).append(sink)
                first = False
            changed = {key for key, head in heads.items() if previous.get(key) != head} | set(recovery)
            previous = dict(heads)
            for session_id in sorted(changed):
                wake = active.get(session_id)
                if wake is None:
                    wake = _Wake()
                    active[session_id] = wake
                    _ = group.create_task(drive(session_id, wake))
                else:
                    wake.changed = True
