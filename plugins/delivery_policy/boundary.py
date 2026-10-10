"""delivery_policy 需要的最小普通插件边界。

这些类型只描述消费者实际使用的能力。Delivery 仍拥有持久记录、发送
回执和未知效果恢复；策略只提交目的地结构并读取结果。
"""

from __future__ import annotations

from collections.abc import Mapping

from agent.plugin_contracts.delivery import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    FINAL_OUTPUT_DELIVERY as FINAL_OUTPUT_DELIVERY,
    GuardedDeliveries as DeliveryExecution,  # noqa: F401 - 显式再导出给本插件消费者。
    FinalOutputTurn as FinalOutputTurn,
    FinalOutputWaiter as FinalOutputWaiter,
)
from plugins.reply.contract import (
    REPLY_COMPLETION as REPLY_COMPLETION,
    Completion as Completion,
)
from plugins.conversation.contract import (
    CHECK_ORIGIN as ORIGIN_CHECK,
    OriginCheck as OriginCheck,
)

SinkInput = Mapping[str, object]
