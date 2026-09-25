"""delivery_policy 需要的最小普通插件边界。

这些类型只描述消费者实际使用的能力。Delivery 仍拥有持久记录、发送
回执和未知效果恢复；策略只提交目的地结构并读取结果。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from agent.plugin_contracts.delivery import (
    DELIVERY as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    FINAL_OUTPUT_DELIVERY as FINAL_OUTPUT_DELIVERY,
    Deliveries as DeliveryExecution,  # noqa: F401 - 显式再导出给本插件消费者。
    FinalOutputTurn as FinalOutputTurn,
    FinalOutputWaiter as FinalOutputWaiter,
)
from agent.plugin_contracts.reply import (
    REPLY_COMPLETION as REPLY_COMPLETION,
    Completion as Completion,
)
from agent.plugin_contracts.sources import (
    CHECK_ORIGIN as ORIGIN_CHECK,  # noqa: F401 - 显式再导出给本插件消费者。
    OriginCheck as OriginCheck,
)

SinkInput = Mapping[str, object]


class DeliveryTask(Protocol):
    """Delivery 返回的真实发送任务句柄。"""

    def on_done(self, callback: Callable[[], None]) -> None: ...
