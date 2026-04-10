from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


@dataclass
class AgentTickContext:
    tick_id: str = field(default_factory=lambda: uuid4().hex[:8])
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str = ""
    context_as_fallback_open: bool = False

    # Gateway 预取结果（_run_loop 启动前由 DataGateway 填充）
    fetched_alerts: list[dict] = field(default_factory=list)    # 含 ack_server 字段
    fetched_contents: list[dict] = field(default_factory=list)  # 含 ack_server 字段（从 content_meta 还原）
    fetched_context: list[dict] = field(default_factory=list)
    alerts_fetched: bool = False
    contents_fetched: bool = False
    context_fetched: bool = False
    # compound_key → 正文（fetch 失败时为 ""）；tick 结束后由 agent_tick 清空
    content_store: dict[str, str] = field(default_factory=dict)

    # 过滤结果（loop 中逐步写入，均为复合键 "{ack_server}:{id}"）
    discarded_item_ids: set[str] = field(default_factory=set)   # mark_not_interesting 写入
    interesting_item_ids: set[str] = field(default_factory=set) # recall_memory 后立即写入，不可撤销

    # 终止状态（由 finish_turn 写入）
    terminal_action: Literal["reply", "skip"] | None = None
    skip_reason: str = ""
    skip_note: str = ""
    final_message: str = ""
    cited_item_ids: list[str] = field(default_factory=list)     # 复合键列表
    steps_taken: int = 0
    drift_entered: bool = False
    drift_finished: bool = False
    drift_message_sent: bool = False

    def mark_alerts_prefetched(self, alerts: list[dict]) -> None:
        self.fetched_alerts = alerts
        self.alerts_fetched = True

    def mark_contents_prefetched(
        self, contents: list[dict], content_store: dict[str, str]
    ) -> None:
        self.fetched_contents = contents
        self.content_store = content_store
        self.contents_fetched = True

    def mark_context_prefetched(self, context_rows: list[dict]) -> None:
        self.fetched_context = context_rows
        self.context_fetched = True

    @property
    def _alerts_fetched(self) -> bool:
        return self.alerts_fetched

    @_alerts_fetched.setter
    def _alerts_fetched(self, value: bool) -> None:
        self.alerts_fetched = value

    @property
    def _contents_fetched(self) -> bool:
        return self.contents_fetched

    @_contents_fetched.setter
    def _contents_fetched(self, value: bool) -> None:
        self.contents_fetched = value

    @property
    def _context_fetched(self) -> bool:
        return self.context_fetched

    @_context_fetched.setter
    def _context_fetched(self, value: bool) -> None:
        self.context_fetched = value
