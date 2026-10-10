from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from agent.plugin_composition import Context
from plugins.ledger.contract import MessageCatalog, MessageReader
from plugins.ledger.contract import Message, Output

from .boundary import DeliveryExecution, SinkInput

logger = logging.getLogger(__name__)
Select = Callable[[MessageReader, Message], tuple[SinkInput, ...] | None]


@dataclass(slots=True)
class _Wake:
    changed: bool = True


@dataclass(slots=True)
class _Destination:
    owner: str
    position: int
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    recovery: list[tuple[int, str]] = field(default_factory=list)


async def follow(
    ctx: Context, catalog: MessageCatalog,
    execution: Callable[[], DeliveryExecution], select: Select,
    *, settled: Callable[[str, str], None] | None = None,
) -> None:
    """按 seq 固定选路；重启追赶 prepared，各目的地与各 Session 独立结算。"""
    active: dict[str, _Wake] = {}
    destinations: dict[tuple[str, str], _Destination] = {}

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

    async def send_destination(session_id: str, sink: str, destination: _Destination) -> None:
        """Read this destination's ordered backlog from durable selections."""
        reader = catalog.reader(session_id)
        for _, message_id in sorted(destination.recovery):
            await send(message_id, sink)
        destination.recovery.clear()
        while True:
            await destination.changed.wait()
            destination.changed.clear()
            while True:
                async with ctx.runtime_scope():
                    delivery = execution()
                    through = delivery.cursor(session_id)
                    messages = tuple(message for message in reader.read(after_seq=destination.position, limit=100)
                                     if message.seq <= through)
                if not messages:
                    break
                for message in messages:
                    async with ctx.runtime_scope():
                        selected = execution().selection(message.message_id)
                    if selected is not None and selected.recovery_owner == destination.owner and sink in selected.sinks:
                        await send(message.message_id, sink)
                    destination.position = message.seq
                await asyncio.sleep(0)

    def wake_destination(session_id: str, sink: str, owner: str, position: int) -> _Destination:
        key = (session_id, sink)
        destination = destinations.get(key)
        if destination is None:
            destination = destinations[key] = _Destination(owner, position)
            group.create_task(send_destination(session_id, sink, destination))
        destination.changed.set()
        return destination

    async def drive(session_id: str, wake: _Wake) -> None:
        """单个 Session 保持消息顺序；实际发送失败不阻止后续消息固定选路。"""
        try:
            reader = catalog.reader(session_id)
            # 2. 新选择与全部 prepared、cursor 同事务，I/O 才可以开始。
            while wake.changed:
                wake.changed = False
                while True:
                    async with ctx.runtime_scope():
                        delivery = execution()
                        messages = reader.read(after_seq=delivery.cursor(session_id), limit=100)
                    if not messages:
                        break
                    async with ctx.runtime_scope():
                        delivery = execution()
                        # 整批选择一次事务提交；选路事实仍按消息逐条固定。
                        batch = tuple(
                            (message, select(reader, message) if delivery.selection(message.message_id) is None else ())
                            for message in messages
                        )
                        selected_batch = await delivery.consume_batch_async(reader, batch, passive=True)
                    for message, selected in zip(messages, selected_batch):
                        if selected is not None:
                            for sink in selected.sinks:
                                wake_destination(session_id, sink, selected.recovery_owner, message.seq - 1)
                    await asyncio.sleep(0)
        except Exception:
            # 选路失败不推进该 Session cursor；其他 Session 仍可独立接纳和发送。
            logger.exception("发送消费停止，保留原消息等待修复 session=%s", session_id)
        finally:
            del active[session_id]

    previous: dict[str, int] = {}
    first = True
    async with asyncio.TaskGroup() as group:
        # 只有 Output 可能被选路；其他消息在下一次 Output 唤醒时按游标顺序一并消费。
        async for heads in catalog.follow(wake_on=Output):
            # 先建立日志订阅再取耐久 pending；恢复后不用扫描当前策略重选旧路由。
            if first:
                async with ctx.runtime_scope():
                    delivery = execution()
                    for message_id, sink in delivery.pending():
                        selection = delivery.selection(message_id)
                        if selection is None:
                            raise ValueError("待恢复发送缺少首次选择")
                        reader = catalog.reader(selection.session_id)
                        message = reader.get(message_id)
                        if message is None:
                            raise ValueError("待恢复发送的原消息缺失")
                        destination = wake_destination(selection.session_id, sink, selection.recovery_owner,
                                                       delivery.cursor(selection.session_id))
                        destination.recovery.append((message.seq, message_id))
                first = False
            changed = {key for key, head in heads.items() if previous.get(key) != head}
            previous = dict(heads)
            for session_id in sorted(changed):
                wake = active.get(session_id)
                if wake is None:
                    wake = _Wake()
                    active[session_id] = wake
                    _ = group.create_task(drive(session_id, wake))
                else:
                    wake.changed = True
