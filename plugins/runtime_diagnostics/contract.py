"""诊断的只读出口；外部接入按游标拉取，不在业务调用中执行发送。"""
from __future__ import annotations

from typing import Protocol

from agent.plugin_composition.model import ServiceKey


class DiagnosticsReader(Protocol):
    def read(self, *, after: int = 0) -> dict[str, object]:
        """返回独立 JSON 值；schema、stream_id、游标与丢失范围见插件 README。"""
        ...


DIAGNOSTICS = ServiceKey[DiagnosticsReader]("runtime.diagnostics.v1")
