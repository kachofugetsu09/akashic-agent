"""有界诊断记录只保存数值与操作身份，不保存消息、参数或异常正文。"""
from __future__ import annotations

from collections import deque
import copy
import logging
import math
import os
import threading
import time
from uuid import uuid4
from typing import TypedDict


class Record(TypedDict):
    seq: int
    timestamp: float
    kind: str
    values: dict[str, object]


class Records(logging.Handler):
    """日志回调只做固定大小复制；所有消费者通过游标读取。"""

    def __init__(self, owner: str) -> None:
        super().__init__()
        self._owner = owner
        self.stream_id = uuid4().hex
        self._guard = threading.Lock()
        self._seq = 0
        # 停顿记录带线程栈，体积远大于样本，只保留最近 8 次。
        self._lanes: dict[str, deque[Record]] = {"samples": deque(maxlen=720), "events": deque(maxlen=2048),
                                                 "stalls": deque(maxlen=8)}
        self._evicted = dict.fromkeys(self._lanes, 0)
        self._last_evicted = dict.fromkeys(self._lanes, 0)

    def append(self, kind: str, values: dict[str, object], *, timestamp: float | None = None) -> None:
        """锁内登记顺序；溢出只淘汰本插件的临时诊断记录。"""
        lane = {"sample": "samples", "stall": "stalls"}.get(kind, "events")
        with self._guard:
            target = self._lanes[lane]
            if len(target) == target.maxlen:
                self._evicted[lane] += 1
                self._last_evicted[lane] = target[0]["seq"]
            self._seq += 1
            target.append({"seq": self._seq, "timestamp": time.time() if timestamp is None else timestamp,
                           "kind": kind, "values": values})

    def emit(self, record: logging.LogRecord) -> None:
        """只接纳现有结构化诊断字段；任意日志正文不进入导出。"""
        fields = record.__dict__.get("akashic_fields")
        if not isinstance(fields, dict):
            return
        if fields.get("plugin_id") == self._owner and fields.get("operation") == "dashboard.http":
            return  # 查看诊断本身不应挤走正在调查的操作。
        event = fields.get("event")
        if event not in {"plugin.operation.start", "plugin.operation.done", "plugin.operation.error",
                         "plugin.operation.cancelled", "plugin.measurement"}:
            return
        values: dict[str, object] = {"event": event}
        for key in ("plugin_id", "generation_id", "operation", "operation_id", "parent_operation_id",
                    "session_id", "turn_id", "fiber", "outcome", "measurement", "measurement_unit"):
            value = fields.get(key)
            if isinstance(value, str):
                values[key] = value[:200]
        for key in ("duration_ms", "measurement_value"):
            value = fields.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values[key] = value
        self.append("operation", values, timestamp=record.created)

    def read(self, *, after: int = 0) -> dict[str, object]:
        """返回副本；消费者停顿不会保留记录或阻塞生产者。"""
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after 必须是非负整数")
        with self._guard:
            return {"schema": "akashic.diagnostics.v1", "stream_id": self.stream_id,
                    "pid": os.getpid(), "release_commit": os.environ.get("AKASHIC_RUNTIME_COMMIT", ""),
                    "through_seq": self._seq, "reset_required": after > self._seq,
                    "sample_interval_seconds": 5,
                    "capacity": {lane: rows.maxlen for lane, rows in self._lanes.items()},
                    "evicted": dict(self._evicted),
                    "gap": {lane: after < seq for lane, seq in self._last_evicted.items()},
                    **{lane: [_row(row) for row in rows if row["seq"] > after] for lane, rows in self._lanes.items()}}


def _row(row: Record) -> dict[str, object]:
    # 停顿记录含嵌套线程栈，复制后消费者修改不会影响保留副本。
    values = copy.deepcopy(row["values"]) if row["kind"] == "stall" else row["values"]
    return {"seq": row["seq"], "timestamp": row["timestamp"], "kind": row["kind"], **values}
