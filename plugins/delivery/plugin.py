from functools import partial

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE, OwnerStore
from agent.plugin_composition.tasks import TASKS, TaskAdmission

from .execution import Deliveries
from .history import DELIVERY_READ, DeliveryHistory
from .records import DeliveryRecords
from .senders import DELIVERY_SENDERS, Senders, open_sender
from .api import FINAL_OUTPUT_DELIVERY, FinalOutputDelivery

api_version = 3
name = "delivery"
version = "1.0.0"
desc = "独立发送已保存消息，固定原目的地与出站绑定并恢复真实效果"
inject = (BINDINGS, MESSAGE_CATALOG, OWNER_STATE, TASKS)

class DeliveryAdmission:
    """按实际消费者签发恢复范围，发送记录和 Task 仍由 Delivery 独占。"""

    def __init__(
        self, ctx: Context, state: OwnerStore | None, tasks: TaskAdmission | None,
    ):
        self._ctx = ctx
        self._state = state
        self._tasks = tasks

    def open(self, consumer: Context) -> Deliveries:
        owner = consumer.require_runtime_owner(DELIVERY, self)
        if self._state is None or self._tasks is None:
            raise RuntimeError("candidate 验证期禁止打开正式 Delivery")
        ctx = self._ctx
        bindings = ctx.require(BINDINGS)
        return Deliveries(
            DeliveryRecords(self._state, owner),
            ctx.require(MESSAGE_CATALOG), self._tasks,
            partial(open_sender, bindings), task_key="delivery",
        )


DELIVERY = ServiceKey[DeliveryAdmission]("delivery.v1")


async def apply(ctx: Context) -> None:
    """Bind Delivery's state and Task once under its own lifecycle owner."""
    state_service = ctx.require(OWNER_STATE)
    state = state_service.open(ctx) if state_service.available else None
    tasks = ctx.require(TASKS).open(ctx) if state is not None else None
    _ = await ctx.provide(DELIVERY_SENDERS, Senders(ctx))
    _ = await ctx.provide(DELIVERY, DeliveryAdmission(ctx, state, tasks))
    _ = await ctx.provide(FINAL_OUTPUT_DELIVERY, FinalOutputDelivery())
    # 状态能力在正式生命周期中才打开，候选加载期不触碰运行库。
    _ = await ctx.provide(DELIVERY_READ, DeliveryHistory(
        lambda: state if state is not None else state_service.open(ctx),
        ctx.require(MESSAGE_CATALOG),
    ))
