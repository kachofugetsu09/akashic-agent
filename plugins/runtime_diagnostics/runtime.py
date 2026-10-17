"""每代拥有一个采样器；记录只在内存中保留，不打开业务存储。"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import time

from core.common.file_io import run_file_io
from .records import Records
from .sampling import memory_sample


class Monitor:
    def __init__(self, records: Records) -> None:
        self.records = records
        self._gc_lock = threading.Lock()
        self._gc_started: dict[int, float] = {}
        self._gc_seconds = 0.0
        self._gc_baseline = gc.get_stats()
        self._logger = logging.getLogger("akashic.plugin.diagnostics")

    def start(self):
        """回调只累加时间；清理移除同一对象，避免热更新重复采集。"""
        self._logger.addHandler(self.records)
        gc.callbacks.append(self._gc)

        def close() -> None:
            gc.callbacks.remove(self._gc)
            self._logger.removeHandler(self.records)
            self.records.close()

        return close

    def _gc(self, phase: str, info: dict[str, int]) -> None:
        now = time.monotonic()
        generation = info["generation"]
        with self._gc_lock:
            if phase == "start":
                self._gc_started[generation] = now
            elif generation in self._gc_started:
                self._gc_seconds += now - self._gc_started.pop(generation)

    async def run(self) -> None:
        """固定低频读取 OS 数据；事件循环上只读任务数和 GC 小计。"""
        loop = asyncio.get_running_loop()
        due = loop.time()
        count = 0
        while True:
            now = loop.time()
            values = await run_file_io(lambda: memory_sample(detailed=count % 6 == 0))
            with self._gc_lock:
                gc_seconds = self._gc_seconds
            stats = gc.get_stats()
            values.update(pid=os.getpid(), task_count=len(asyncio.all_tasks()),
                          loop_lag_ms=max(0, now - due) * 1000,
                          gc_seconds=gc_seconds,
                          gc_collections=sum(row["collections"] - before["collections"] for row, before in zip(stats, self._gc_baseline)),
                          gc_collected=sum(row["collected"] - before["collected"] for row, before in zip(stats, self._gc_baseline)),
                          operations_enabled=self._logger.isEnabledFor(logging.INFO))
            self.records.append("sample", values)
            count += 1
            due = now + 5
            await asyncio.sleep(max(0, due - loop.time()))
