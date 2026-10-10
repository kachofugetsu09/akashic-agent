"""投递策略拥有的输入来源查询合同。"""

from __future__ import annotations

from typing import Protocol

from agent.plugin_composition import ServiceKey
from plugins.ledger.contract import MessageReader


class InputOrigin(Protocol):
    def __call__(
        self, reader: MessageReader, source: str, *, through_seq: int
    ) -> tuple[str, str] | None: ...


INPUT_ORIGIN = ServiceKey[InputOrigin]("delivery.input-origin.v1")
