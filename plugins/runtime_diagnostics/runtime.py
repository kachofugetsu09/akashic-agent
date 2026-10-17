"""每代拥有一个采样器；记录只在内存中保留，不打开业务存储。"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import threading
import time
import traceback

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


STALL_THRESHOLD_SECONDS = 1.0
_BEAT_SECONDS = 0.1
_STACK_DEPTH = 30


class StallWatch:
    """事件循环停住超过阈值时，由循环外的线程抓取全部线程栈，恢复后登记一条 stall 事件。"""

    def __init__(self, records: Records, *, threshold: float = STALL_THRESHOLD_SECONDS) -> None:
        self._records = records
        self._threshold = threshold
        self._beat = time.monotonic()
        self._stopped = threading.Event()

    async def run(self) -> None:
        """心跳只在事件循环上写时间戳；取消时让看门狗线程在下一次唤醒时退出。"""
        self._beat = time.monotonic()
        threading.Thread(target=self._watch, args=(threading.get_ident(),),
                         name="diagnostics-stall-watch", daemon=True).start()
        try:
            while True:
                self._beat = time.monotonic()
                await asyncio.sleep(_BEAT_SECONDS)
        finally:
            self._stopped.set()

    # 看门狗线程：停顿越过阈值时先抓栈，心跳恢复后按实际停顿时长登记。
    def _watch(self, loop_thread: int) -> None:
        stalled_since: float | None = None
        threads: list[dict[str, object]] = []
        while not self._stopped.wait(_BEAT_SECONDS):
            # 1. 心跳被拖住：抓一次此刻的全部线程栈，之后同一次停顿不再重复抓。
            beat = self._beat
            if stalled_since is None:
                if time.monotonic() - beat >= self._threshold:
                    stalled_since, threads = beat, _thread_stacks(loop_thread)
                continue
            # 2. 心跳前进说明循环已恢复；登记停顿起点、时长和抓到的栈。
            if beat != stalled_since:
                started = time.time() - (time.monotonic() - stalled_since)
                self._records.append("stall", {
                    "stall_ms": round(max(0.0, beat - stalled_since - _BEAT_SECONDS) * 1000, 1),
                    "threshold_ms": self._threshold * 1000, "threads": threads}, timestamp=started)
                stalled_since, threads = None, []


# 每个线程只取函数、文件和行号，不读取局部变量或源码行。
def _thread_stacks(loop_thread: int) -> list[dict[str, object]]:
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    current = threading.get_ident()
    rows: list[dict[str, object]] = []
    for ident, frame in sys._current_frames().items():  # pyright: ignore[reportPrivateUsage]
        if ident == current:
            continue
        summary = traceback.StackSummary.extract(traceback.walk_stack(frame), limit=_STACK_DEPTH,
                                                 lookup_lines=False)
        rows.append({"thread": names.get(ident, str(ident)), "event_loop": ident == loop_thread,
                     "frames": [f"{item.filename}:{item.lineno} {item.name}" for item in summary]})
    return rows
