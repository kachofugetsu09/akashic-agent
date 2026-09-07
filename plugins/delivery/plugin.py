from functools import partial

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE
from agent.plugin_composition.tasks import TASKS

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

    def __init__(self, ctx: Context):
        self._ctx = ctx

    def open(self, consumer: Context) -> Deliveries:
        owner = consumer.require_runtime_owner(DELIVERY, self)
        ctx = self._ctx
        return Deliveries(
            DeliveryRecords(ctx.require(OWNER_STATE).open(ctx), owner),
            ctx.require(MESSAGE_CATALOG), ctx.require(TASKS).open(ctx),
            partial(open_sender, ctx.require(BINDINGS)), task_key="delivery",
        )


DELIVERY = ServiceKey[DeliveryAdmission]("delivery.v1")


async def apply(ctx: Context, config: object) -> None:
    """注册出站能力；只有正式调用才取得发送 owner 的状态和 Task。"""
    _ = await ctx.provide(DELIVERY_SENDERS, Senders(ctx))
    _ = await ctx.provide(DELIVERY, DeliveryAdmission(ctx))
    # 状态能力在正式生命周期中才打开，候选加载期不触碰运行库。
    _ = await ctx.provide(DELIVERY_READ, DeliveryHistory(
        lambda: ctx.require(OWNER_STATE).open(ctx), ctx.require(MESSAGE_CATALOG),
    ))
