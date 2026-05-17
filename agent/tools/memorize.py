from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool
from core.memory.engine import (
    MemoryMutation,
    MemoryScope,
    MemoryToolSpec,
)

if TYPE_CHECKING:
    from core.memory.engine import MemoryWriteApi

logger = logging.getLogger(__name__)


class MemorizeTool(Tool):
    name = "memorize"
    description = "由当前 memory engine 的 tool_profile 注入工具描述。"
    parameters = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def __init__(
        self,
        memory: "MemoryWriteApi",
        spec: MemoryToolSpec,
    ) -> None:
        self._memory = memory
        self._spec = spec
        self.description = self._spec.description
        self.parameters = self._spec.parameters

    async def execute(
        self,
        summary: str,
        memory_kind: str = "",
        tool_requirement: str | None = None,
        steps: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        current_user_source_ref: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **extra_kwargs: Any,
    ) -> str:
        kind = memory_kind.strip()
        extra = dict(metadata or {})
        extra.update(extra_kwargs)
        if tool_requirement is not None:
            extra["tool_requirement"] = tool_requirement
        if steps is not None:
            extra["steps"] = steps
        result = await self._memory.mutate(
            MemoryMutation(
                kind="remember",
                summary=summary,
                memory_kind=kind,
                source_ref=str(current_user_source_ref or "").strip(),
                scope=MemoryScope(
                    session_key=f"{channel}:{chat_id}" if channel and chat_id else "",
                    channel=channel or "",
                    chat_id=chat_id or "",
                ),
                metadata=extra,
            )
        )
        logger.info("memorize: engine stored memory_kind=%s", result.actual_kind)
        return _format_result(result.item_id, result.status, result.actual_kind, summary)


def _format_result(item_id: str, status: str, actual_kind: str, summary: str) -> str:
    value = (item_id or "").strip()
    write_status = (status or "new").strip()
    kind = (actual_kind or "").strip()
    if kind:
        return f"已记住（item_id={value}；kind={kind}；status={write_status}）：{summary}"
    return f"已记住（item_id={value}；status={write_status}）：{summary}"
