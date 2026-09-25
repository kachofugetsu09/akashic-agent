"""message_push 依赖的局部能力边界。

ServiceKey 只按稳定名称匹配，真实实现仍由各 owner 注册；这里不复制
Delivery、Tools、TurnProjection 或 Content 的内部对象。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_contracts import (
    ContentPart,
)
from agent.plugin_contracts.content import (
    CONTENT as CONTENT,
    ContentView as ContentView,
)
from agent.plugin_contracts.delivery import (
    DELIVERY as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    FINAL_OUTPUT_DELIVERY as FINAL_OUTPUT_DELIVERY,
    FinalOutputDelivery as FinalOutputDelivery,
    FinalOutputTurn as FinalOutputTurn,
    FinalOutputWaiter as FinalOutputWaiter,
    Receipt as ReceiptView,  # noqa: F401 - 显式再导出给本插件消费者。
)
from agent.plugin_contracts.tools import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
)
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as ProjectedTurn,  # noqa: F401 - 显式再导出给本插件消费者。
    TurnProjection as TurnProjection,
)

ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """message_push 自有的结构结果；tools owner 在执行边界重新校验。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


class BoundTool(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self,
        arguments: Mapping[str, object],
        source: CallSource | None = None,
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResult: ...

    async def query(self, key: str) -> ToolResult | None: ...
