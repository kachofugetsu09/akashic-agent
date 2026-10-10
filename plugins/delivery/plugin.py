from functools import partial

from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE, OwnerStore
from agent.plugin_composition.tasks import TASKS, TaskAdmission
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START,
)

from .api import FINAL_OUTPUT_DELIVERY, FinalOutputDelivery
from .execution import Deliveries
from .history import DELIVERY_READ, DeliveryHistory
from .records import DeliveryRecords
from .senders import DELIVERY_SENDERS, Senders, open_sender

api_version = 3
name = "delivery"
version = "1.0.0"
desc = "独立发送已保存消息，固定原目的地与出站绑定并恢复真实效果"
inject = (BINDINGS, MESSAGE_CATALOG, OWNER_STATE, TASKS)

class DeliveryAdmission:
    """按实际消费者签发恢复范围，发送记录和 Task 仍由 Delivery 独占。"""

    def __init__(
        self, ctx: Context, state: OwnerStore, tasks: TaskAdmission,
    ):
        self._ctx = ctx
        self._state = state
        self._tasks = tasks

    def open(self, consumer: Context) -> Deliveries:
        owner = consumer.require_runtime_identity(DELIVERY_GUARDED_START, self).plugin_id
        ctx = self._ctx
        bindings = ctx.require(BINDINGS)
        return Deliveries(
            DeliveryRecords(self._state, owner),
            ctx.require(MESSAGE_CATALOG), self._tasks,
            partial(open_sender, bindings), task_key="delivery",
        )


async def apply(ctx: Context) -> None:
    """Bind Delivery's state and Task once under its own lifecycle owner."""
    state_service = ctx.require(OWNER_STATE)
    state = state_service.open(ctx)
    tasks = ctx.require(TASKS).open(ctx)
    _ = await ctx.provide(DELIVERY_SENDERS, Senders(ctx))
    _ = await ctx.provide(DELIVERY_GUARDED_START, DeliveryAdmission(ctx, state, tasks))
    _ = await ctx.provide(FINAL_OUTPUT_DELIVERY, FinalOutputDelivery())
    _ = await ctx.provide(DELIVERY_READ, DeliveryHistory(
        lambda: state,
        ctx.require(MESSAGE_CATALOG),
    ))
