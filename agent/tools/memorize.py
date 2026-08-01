from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent.tools.base import Tool, get_current_tool_context
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
    ) -> str:
        kind = memory_kind.strip()
        extra = dict(metadata or {})
        if tool_requirement is not None:
            extra["tool_requirement"] = tool_requirement
        if steps is not None:
            extra["steps"] = steps
        context = get_current_tool_context()
        result = await self._memory.mutate(
            MemoryMutation(
                kind="remember",
                summary=summary,
                memory_kind=kind,
                source_ref=(context.current_user_source_ref if context else "").strip(),
                scope=MemoryScope(
                    session_key=context.origin_session_key if context else "",
                    channel=context.origin_channel if context else "",
                    chat_id=context.origin_chat_id if context else "",
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
