"""主动失效错误记忆条目的工具。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool

if TYPE_CHECKING:
    from memory2.store import MemoryStore2


class ForgetMemoryTool(Tool):
    name = "forget_memory"
    description = (
        "将已确认错误的记忆条目标记为失效（status='superseded'）。\n"
        "只在用户明确纠正你，且你已先用 recall_memory 确认 summary 与错误内容吻合时调用。\n"
        "禁止在未核实内容的情况下直接传 id；禁止把它当成搜索工具使用。\n"
        "执行后条目会被标记为 superseded，不可恢复。若用户同时给出正确版本，可再单独调用 memorize 写入。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要失效的 memory item id 列表",
            }
        },
        "required": ["ids"],
    }

    def __init__(self, store: "MemoryStore2") -> None:
        self._store = store

    async def execute(self, ids: list[str], **_: Any) -> str:
        clean_ids: list[str] = []
        seen: set[str] = set()
        for raw in ids or []:
            item_id = str(raw).strip()
            if item_id and item_id not in seen:
                seen.add(item_id)
                clean_ids.append(item_id)
        if not clean_ids:
            return json.dumps(
                {
                    "requested_ids": [],
                    "superseded_ids": [],
                    "missing_ids": [],
                    "count": 0,
                    "items": [],
                },
                ensure_ascii=False,
            )

        items = self._store.get_items_by_ids(clean_ids)
        found_ids = [str(item.get("id") or "") for item in items if str(item.get("id") or "")]
        if found_ids:
            self._store.mark_superseded_batch(found_ids)
        missing_ids = [item_id for item_id in clean_ids if item_id not in set(found_ids)]
        return json.dumps(
            {
                "requested_ids": clean_ids,
                "superseded_ids": found_ids,
                "missing_ids": missing_ids,
                "count": len(found_ids),
                "items": [
                    {
                        "id": item.get("id"),
                        "memory_type": item.get("memory_type"),
                        "summary": item.get("summary"),
                    }
                    for item in items
                ],
            },
            ensure_ascii=False,
        )
