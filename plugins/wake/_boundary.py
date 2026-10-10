"""Wake 使用的外部能力窄边界。

这里仅声明 Wake 实际消费的结构和操作。ServiceKey 按稳定名称连接真实
owner；Delivery、Tools、Content 和兴趣服务的实现与内部模型不进入 Wake。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from agent.plugin_composition import EmitEventKey, ServiceKey
from agent.plugin_contracts import (
    ContentPart,
)
from plugins.content.contract import (
    CONTENT as CONTENT,
)
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_READ as DELIVERY_READ,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    DeliveryHistory as DeliveryHistory,
)
from plugins.akasha.contract import SEMANTIC_INTEREST as SEMANTIC_INTEREST, SemanticInterest as SemanticInterest
from plugins.tools.contract import (
    ALL_TOOLS as ALL_TOOLS,
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
    ToolView as ToolView,
)


class SinkValue(TypedDict):
    """Wake 归档的发送目的地；Delivery owner 在入口重新校验。"""

    name: str
    binding_id: str
    address: str


ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResultValue:
    """Wake 工具返回的结构值；tools owner 在执行边界归一化它。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


WAKE_TOOLS_VIEW = ServiceKey[ToolView]("wake.tools.v1")


DRIFT_CHANGED = EmitEventKey[None]("drift.changed")
