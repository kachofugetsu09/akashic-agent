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
    _alerts_fetched: bool = False
    _contents_fetched: bool = False
    _context_fetched: bool = False
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
