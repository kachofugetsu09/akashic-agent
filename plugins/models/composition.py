"""上下文构成：每次成功调用按来源估算各部分的份额，界面用实报输入按份额换算。"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from .contract import ModelRequest

_KEEP_DAYS = 31

Estimate = Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], int]
# 类别 -> 来源名 -> 估算 token；来源名为空表示这一类不再细分。
Parts = dict[str, dict[str, int]]


# 装配方给了逐来源拆分就按它算；没有时（标题、看图等旁路调用）按消息角色粗分。
def measure(request: ModelRequest, estimate: Estimate) -> Parts:
    parts: Parts = {}

    def add(category: str, label: str, value: int) -> None:
        if value > 0:
            group = parts.setdefault(category, {})
            group[label] = group.get(label, 0) + value

    if request.sources:
        for source in request.sources:
            add(source.category, source.label, estimate(source.rows, source.tools))
        return parts
    for row in request.messages:
        role = row.get("role")
        add("system" if role in ("system", "developer") else "assistant" if role == "assistant"
            else "tool_results" if role == "tool" else "user", "", estimate((row,), ()))
    if request.system_prompt:
        add("system", "", estimate(({"role": "system", "content": request.system_prompt},), ()))
    if request.tools:
        add("tools", "", estimate((), request.tools))
    return parts


class CompositionLog:
    """调用 ID 到构成份额；独立文件，旧版本不读它，删掉只是丢失历史构成。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        if not self._ready:
            _ = connection.execute(
                "CREATE TABLE IF NOT EXISTS composition (call_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, parts TEXT NOT NULL)")
            self._ready = True
        return connection

    # 写入时顺带清掉超出保留期的行；失败由调用方记日志，不影响调用本身。
    def put(self, call_id: str, parts: Parts) -> None:
        now = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection, connection:
            _ = connection.execute("INSERT OR REPLACE INTO composition VALUES (?, ?, ?)",
                                   (call_id, now.isoformat(), json.dumps(parts, ensure_ascii=False)))
            _ = connection.execute("DELETE FROM composition WHERE recorded_at < ?",
                                   ((now - timedelta(days=_KEEP_DAYS)).isoformat(),))

    def get(self, call_ids: Iterable[str]) -> dict[str, Parts]:
        ids = list(dict.fromkeys(call_ids))
        if not ids or not self._path.exists():
            return {}
        found: dict[str, Parts] = {}
        with self._lock, closing(self._connect()) as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                rows = connection.execute(
                    f"SELECT call_id, parts FROM composition WHERE call_id IN ({','.join('?' * len(chunk))})", chunk)
                found.update((call_id, cast(Parts, json.loads(parts))) for call_id, parts in rows)
        return found
