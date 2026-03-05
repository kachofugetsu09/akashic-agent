"""
proactive/schedule.py — 用户作息配置的动态存储。

从独立的 schedule.json 读取作息相关设置，
与 config.json 解耦，允许随时修改而无需重启。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ScheduleStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """每次调用都从磁盘读取，确保修改即时生效。"""
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[schedule] 读取失败，忽略: %s", e)
            return {}
