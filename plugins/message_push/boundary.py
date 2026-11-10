"""message_push 依赖的局部能力边界。

ServiceKey 只按稳定名称匹配，真实实现仍由各 owner 注册；这里不复制
Delivery、Tools、TurnProjection 或 Content 的内部对象。
"""

from __future__ import annotations

from plugins.content.contract import (
    CONTENT as CONTENT,
    ContentView as ContentView,
)
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    FINAL_OUTPUT_DELIVERY as FINAL_OUTPUT_DELIVERY,
    FinalOutputDelivery as FinalOutputDelivery,
    FinalOutputTurn as FinalOutputTurn,
    FinalOutputWaiter as FinalOutputWaiter,
    Receipt as ReceiptView,
)
from plugins.tools.contract import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    BoundTool as BoundTool,
    Outcome as ToolOutcome,
    Result as ToolResult,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
)
from plugins.turn_projection.contract import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as ProjectedTurn,  # noqa: F401 - 显式再导出给本插件消费者。
    TurnProjection as TurnProjection,
)
