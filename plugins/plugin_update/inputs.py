"""plugin_update 消费的内容、工具与投递能力。"""
from __future__ import annotations

from plugins.content.contract import (
    CONTENT as CONTENT,
)
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    GuardedDeliveries as Deliveries,
    GuardedDelivery as Delivery,
    Receipt as Receipt,
    Selection as Selection,
    Senders as Senders,
)
from plugins.delivery_policy.contract import (
    INPUT_ORIGIN as INPUT_ORIGIN,
    InputOrigin as InputOrigin,
)
from plugins.tools.contract import (
    ALL_TOOLS as ALL_TOOLS,
    TOOLS as TOOLS,
    CallSource as CallSource,
    Result as Result,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
    ToolView as ToolView,
)
