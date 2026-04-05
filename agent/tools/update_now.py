"""update_now 工具：agent 主动更新 NOW.md「近期进行中」状态。"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.tools.base import Tool

from core.memory.profile import ProfileMaintenanceStore

logger = logging.getLogger(__name__)


class UpdateNowTool(Tool):
    name = "update_now"
    description = (
        "更新 NOW.md「近期进行中」状态。"
        "跨对话任务状态发生变化时必须调用：开始新任务、任务完成/取消、阅读坐标推进、待确认事项产生或消解。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "新增到「近期进行中」的条目（自然语言一句话，不带 bullet 符号）",
            },
            "remove_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要从「近期进行中」删除的条目关键词（模糊匹配，命中即删）",
            },
        },
        "required": [],
    }

    def __init__(self, memory: ProfileMaintenanceStore) -> None:
        self._memory = memory

    @staticmethod
    def _coerce_list(val: Any) -> list[str]:
        """LLM 有时把数组参数序列化成 JSON 字符串传入，统一转回 list[str]。"""
        if val is None:
            return []
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("["):
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            if isinstance(val, list):
                return [s for s in val if isinstance(s, str) and s.strip()]
            return [val] if val else []
        if isinstance(val, list):
            return [s for s in val if isinstance(s, str) and s.strip()]
        return []

    async def execute(
        self,
        add: list[str] | None = None,
        remove_keywords: list[str] | None = None,
        **_: Any,
    ) -> str:
        add = self._coerce_list(add)
        remove_keywords = self._coerce_list(remove_keywords)
        if not add and not remove_keywords:
            return "无变化（add 和 remove_keywords 均为空）"
        self._memory.update_now_ongoing(add=add, remove_keywords=remove_keywords)
        parts = []
        if add:
            parts.append(f"已添加 {len(add)} 条: {add}")
        if remove_keywords:
            parts.append(f"已删除含关键词 {remove_keywords} 的条目")
        logger.info(f"update_now: {'; '.join(parts)}")
        return "NOW.md 已更新：" + "，".join(parts)
