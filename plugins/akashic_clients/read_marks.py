"""跨设备共享的会话已读水位；消息日志仍是 seq 的唯一事实来源。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractContextManager

from plugins.ledger.contract import OwnerRecord, OwnerStore, OwnerTransaction

# 每个会话一条记录；基线记录不落在 "read:" 区间内，扫描时天然排除。
_PREFIX = "read:"
_STOP = "read;"
_BASELINE = "read-baseline"
_SCAN_LIMIT = 1000


def _seq(record: OwnerRecord | None) -> int | None:
    if record is None:
        return None
    value = record.value.get("seq")
    if set(record.value) != {"seq"} or type(value) is not int or value < -1:
        raise ValueError("会话已读记录损坏")
    return value


class ReadMarks:
    """单调前进的已读水位；同进程订阅者在提交后被唤醒。"""

    def __init__(self, open_store: Callable[[], AbstractContextManager[OwnerStore]]) -> None:
        self._open_store = open_store
        self._changed = asyncio.Event()

    def has_baseline(self) -> bool:
        with self._open_store() as store:
            return store.read(_BASELINE) is not None

    # 首次启用时把已有会话视为已读，之后出现的新会话从头算未读。
    def ensure_baseline(self, heads: Mapping[str, int]) -> None:
        def save(transaction: OwnerTransaction) -> bool:
            if transaction.read(_BASELINE) is not None:
                return False
            for session_id, head in heads.items():
                key = _PREFIX + session_id
                if transaction.read(key) is None:
                    _ = transaction.save(key, {"seq": head}, expected_version=None)
            _ = transaction.save(_BASELINE, {"sessions": len(heads)}, expected_version=None)
            return True
        with self._open_store() as store:
            changed = store.transact(save)
        if changed:
            self._notify()

    def read_all(self) -> dict[str, int]:
        # 1. 倒序分页读完整区间；同一快照内读取，避免翻页时漏掉并发写入。
        def scan() -> dict[str, int]:
            marks: dict[str, int] = {}
            stop = _STOP
            while True:
                rows = store.scan(start=_PREFIX, stop=stop, limit=_SCAN_LIMIT)
                for key, record in rows:
                    seq = _seq(record)
                    assert seq is not None
                    marks[key.removeprefix(_PREFIX)] = seq
                if len(rows) < _SCAN_LIMIT:
                    return marks
                stop = rows[-1][0]
        with self._open_store() as store:
            return store.snapshot(scan)

    # 只前进不后退；返回提交后的水位。
    def mark(self, session_id: str, seq: int) -> int:
        key = _PREFIX + session_id

        def save(transaction: OwnerTransaction) -> tuple[int, bool]:
            current = transaction.read(key)
            previous = _seq(current)
            if previous is not None and previous >= seq:
                return previous, False
            _ = transaction.save(key, {"seq": seq}, expected_version=None if current is None else current.version)
            return seq, True
        with self._open_store() as store:
            value, changed = store.transact(save)
        if changed:
            self._notify()
        return value

    async def follow(self) -> AsyncGenerator[dict[str, int], None]:
        while True:
            changed = self._changed
            yield self.read_all()
            await changed.wait()

    def _notify(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        _ = changed.set()
