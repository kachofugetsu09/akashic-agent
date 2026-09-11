from __future__ import annotations



from agent.plugin_contracts.turn_projection import Turn
from agent.plugin_composition.messages import MessageReader

# 投递值模型、Sender/FinalOutputWaiter Protocol 与 key 的拥有者已移到结构合同层；
# 这里按原路径再导出，既有调用点与对象身份不变。
from agent.plugin_contracts.delivery_api import (  # noqa: E402,F401  (再导出)
    FINAL_OUTPUT_DELIVERY,
    OpenSender,
    Receipt,
    Sender,
    Sink,
    Status,
    Text,
)
from agent.plugin_contracts.delivery_api import FinalOutputWaiter  # noqa: E402,F401













class FinalOutputDelivery:
    """按来源选择最终 Output 的普通 delivery 能力。"""

    def __init__(self) -> None:
        self._providers: dict[str, FinalOutputWaiter] = {}

    def register(self, source: str, provider: FinalOutputWaiter) -> None:
        if not source or source in self._providers:
            raise ValueError("最终 Output provider 已有 owner")
        self._providers[source] = provider

    def unregister(self, source: str, provider: FinalOutputWaiter) -> None:
        """只移除仍由同一 optional child owner 登记的 provider。"""
        if self._providers.get(source) is provider:
            del self._providers[source]

    async def wait(self, reader: MessageReader, turn: Turn) -> None:
        provider = self._providers.get(turn.source)
        if provider is None:
            raise ValueError(f"没有来源 {turn.source!r} 的最终 Output provider")
        await provider.wait(reader, turn)


