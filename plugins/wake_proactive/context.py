from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


InitialInterest = Literal["likely_interesting", "not_interesting", "uncertain"]
InvestigationKind = Literal["none", "content", "recall", "both"]


def _event_list() -> list[dict[str, Any]]:
    return []


def _scratch_dict() -> dict[str, ScratchItem]:
    return {}


def _result_dict() -> dict[str, dict[str, Any]]:
    return {}


def _string_list() -> list[str]:
    return []


def _index_map() -> dict[int, str]:
    return {}


def _source_ref_list() -> list[dict[str, Any]]:
    return []


@dataclass(slots=True)
class ScratchItem:
    item_id: str
    initial_interest: InitialInterest
    investigate: InvestigationKind
    question: str = ""
    recall_query: str = ""


@dataclass(slots=True)
class WakeContext:
    wake_id: str = field(default_factory=lambda: uuid4().hex)
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str = ""
    content_events: list[dict[str, Any]] = field(default_factory=_event_list)
    content_backlog_count: int = 0
    scratchpad: dict[str, ScratchItem] = field(default_factory=_scratch_dict)
    screening_completed: bool = False
    investigation_results: dict[str, dict[str, Any]] = field(default_factory=_result_dict)
    investigation_completed: bool = False
    final_message: str = ""
    cited_item_ids: list[str] = field(default_factory=_string_list)
    display_event_map: dict[int, str] = field(default_factory=_index_map)
    source_refs: list[dict[str, Any]] = field(default_factory=_source_ref_list)
    terminal_action: Literal["reply", "skip"] | None = None
    steps_taken: int = 0


def event_item_id(event: dict[str, Any]) -> str:
    item_id = str(event.get("item_id") or event.get("id") or "").strip()
    ack_server = str(event.get("ack_server") or "").strip()
    if ack_server and item_id and ":" not in item_id:
        return f"{ack_server}:{item_id}"
    return item_id


def content_candidate_map(ctx: WakeContext) -> dict[str, dict[str, Any]]:
    """为本轮内容候选分配只在当前 Wake 内有效的模型引用。"""

    return {
        f"candidate_{index}": event
        for index, event in enumerate(ctx.content_events, 1)
    }


def content_event_map(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 reservoir canonical ID 索引已验证的内部事件。"""

    return {event_item_id(event): event for event in events}
