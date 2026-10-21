"""上下文构成：每次成功调用按来源估算各部分的份额，界面用实报输入按份额换算。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from .contract import ModelRequest, message_size

_KEEP_DAYS = 31
_PRUNE_SECONDS = 3600.0
# 一轮明细的回包预算：每类最多列这么多来源，其余并成一项；来源名截断。
_SOURCES_PER_CATEGORY = 30
_LABEL_CHARS = 60

# 类别 -> 来源名 -> 估算 token；来源名为空表示这一类不再细分。
Parts = dict[str, dict[str, int]]


# 只求份额，用无状态的粗估；不借 driver 的估算器，免得覆盖它为在线预算留的增量缓存。
def _estimate(rows: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> int:
    chars = images = 0
    for row in rows:
        size = message_size(row)
        chars, images = chars + size[0], images + size[1]
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False, separators=(",", ":"), default=dict))
    return chars // 3 + images


# 装配方给了逐来源拆分就按它算；没有时（标题、看图等旁路调用）按消息角色粗分。
def measure(request: ModelRequest) -> Parts:
    parts: Parts = {}

    def add(category: str, label: str, value: int) -> None:
        if value > 0:
            group = parts.setdefault(category, {})
            group[label] = group.get(label, 0) + value

    if request.sources:
        for source in request.sources:
            add(source.category, source.label, _estimate(source.rows, source.tools))
        return parts
    for row in request.messages:
        role = row.get("role")
        add("system" if role in ("system", "developer") else "assistant" if role == "assistant"
            else "tool_results" if role == "tool" else "user", "", _estimate((row,), ()))
    if request.system_prompt:
        add("system", "", _estimate(({"role": "system", "content": request.system_prompt},), ()))
    if request.tools:
        add("tools", "", _estimate((), request.tools))
    return parts


# 逐轮趋势只要六类合计，来源明细按需单独查。
def totals(parts: Mapping[str, Mapping[str, int]]) -> Parts:
    return {category: {"": sum(group.values())} for category, group in parts.items()}


# 一轮明细按预算截断：每类保留最大的若干来源，其余合成「其余 N 项」，总和不变。
def bounded(parts: Mapping[str, Mapping[str, int]]) -> Parts:
    result: Parts = {}
    for category, group in parts.items():
        ranked = sorted(group.items(), key=lambda item: -item[1])
        kept: dict[str, int] = {}
        for label, value in ranked[:_SOURCES_PER_CATEGORY]:
            name = label if len(label) <= _LABEL_CHARS else label[:_LABEL_CHARS - 1] + "…"
            kept[name] = kept.get(name, 0) + value
        rest = ranked[_SOURCES_PER_CATEGORY:]
        if rest:
            kept[f"其余 {len(rest)} 项"] = sum(value for _, value in rest)
        result[category] = kept
    return result


class CompositionLog:
    """调用 ID 到构成份额；独立文件，旧版本不读它，删掉只是丢失历史构成。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._pruned: float | None = None

    # 每次连接都确认表在：文件被删掉后下次写入原地重建，不靠进程内的「已建表」标记。
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        _ = connection.execute(
            "CREATE TABLE IF NOT EXISTS composition (call_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, parts TEXT NOT NULL)")
        _ = connection.execute("CREATE INDEX IF NOT EXISTS composition_recorded ON composition (recorded_at)")
        return connection

    # 写入时每小时最多清一次超出保留期的行；失败由调用方记日志，不影响调用本身。
    def put(self, call_id: str, parts: Parts) -> None:
        now = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection, connection:
            _ = connection.execute("INSERT OR REPLACE INTO composition VALUES (?, ?, ?)",
                                   (call_id, now.isoformat(), json.dumps(parts, ensure_ascii=False)))
            if self._pruned is None or time.monotonic() - self._pruned >= _PRUNE_SECONDS:
                _ = connection.execute("DELETE FROM composition WHERE recorded_at < ?",
                                       ((now - timedelta(days=_KEEP_DAYS)).isoformat(),))
                self._pruned = time.monotonic()

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
