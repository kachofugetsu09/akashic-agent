"""事件循环停顿现场：独立看门狗线程在停顿期间抓取线程栈，只保留代码位置，不读取变量或对象值。"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
import threading
import time
from types import FrameType

from .profiling import _export_locator  # pyright: ignore[reportPrivateUsage]
from .records import Records

# 正常回调远低于 100ms，大堆完整 GC 与偶发同步文件 I/O 可达数百毫秒；2 秒避开这些噪声，
# 又远早于 Web 守卫 10 秒超时，阻塞代码仍在执行时就能抓到现场。
STALL_SECONDS = 2.0
BEAT_SECONDS = 0.5
CHECK_SECONDS = 0.25
# 迟到超过半个阈值的心跳单独留底，后续健康心跳覆盖 _pulse 时仍能找回停顿真正的恢复时长。
LATE_SECONDS = STALL_SECONDS / 2
# 首次抓取后按约 3 倍间隔补抓两次，覆盖 #1455 的 10～31 秒停顿，区分单个长调用与接连阻塞。
FOLLOW_UP_SECONDS = (6.0, 18.0)
# 基线只用于对比停顿窗口内的 CPU 时间差；4 秒刷新一次，几十个线程时约 0.15ms/秒。
CPU_BASELINE_SECONDS = 4.0
MAX_THREADS = 48
MAX_FRAMES = 20
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


@dataclass
class _Cpu:
    at: float
    seconds: dict[int, float]


@dataclass
class _ThreadStack:
    name: str
    native_id: int | None
    loop: bool
    cpu_ms: float | None
    frames: list[str]
    frames_omitted: int
    hot: bool = False

    def value(self) -> dict[str, object]:
        return {"name": self.name, "native_id": self.native_id, "loop": self.loop, "hot": self.hot,
                "cpu_ms": self.cpu_ms, "frames": self.frames, "frames_omitted": self.frames_omitted}


@dataclass
class _Capture:
    elapsed_ms: float
    watchdog_delay_ms: float
    cpu_window_ms: float
    threads: list[_ThreadStack]

    def value(self) -> dict[str, object]:
        return {"elapsed_ms": self.elapsed_ms, "watchdog_delay_ms": self.watchdog_delay_ms,
                "cpu_window_ms": self.cpu_window_ms, "threads": [row.value() for row in self.threads[:MAX_THREADS]],
                "threads_omitted": max(0, len(self.threads) - MAX_THREADS)}

    def top(self, *, loop: bool) -> str | None:
        return next((row.frames[0] for row in self.threads
                     if (row.loop if loop else row.hot) and row.frames), None)


@dataclass
class _Stall:
    due: float
    started_at: float
    cpu: _Cpu
    captures: list[_Capture] = field(default_factory=lambda: [])


# 线程 CPU 时间来自 /proc/self/task/<tid>/stat 的 utime+stime；非 Linux 返回空表。
def _thread_cpu(native_ids: Iterable[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for native_id in native_ids:
        try:
            fields = Path(f"/proc/self/task/{native_id}/stat").read_text().rsplit(")", 1)[1].split()
            result[native_id] = (int(fields[11]) + int(fields[12])) / _CLOCK_TICKS
        except (OSError, IndexError, ValueError):
            continue
    return result


def _python_threads() -> dict[int, threading.Thread]:
    return {thread.ident: thread for thread in threading.enumerate() if thread.ident is not None}


# 由内向外保留最多 MAX_FRAMES 层「文件:行 in 函数」；只读代码对象与行号。
def _frames(frame: FrameType | None, locators: dict[str, str]) -> tuple[list[str], int]:
    rows: list[str] = []
    depth = 0
    while frame is not None:
        if len(rows) < MAX_FRAMES:
            code = frame.f_code
            locator = locators.get(code.co_filename)
            if locator is None:
                locator = locators[code.co_filename] = _export_locator(code.co_filename)
            rows.append(f"{locator}:{frame.f_lineno or 0} in {code.co_qualname}")
        depth += 1
        frame = frame.f_back
    return rows, depth - len(rows)


class StallWatch:
    """心跳运行在事件循环上；判断、抓栈与记录在独立线程中完成，不依赖循环推进。"""

    def __init__(self, records: Records) -> None:
        self._records = records
        self._logger = logging.getLogger("akashic.plugin.diagnostics")
        self._stop = threading.Event()
        self._loop_ident = 0
        self._watch_ident = 0
        self._pulse = (time.monotonic(), 0.0)
        self._late: deque[tuple[float, float]] = deque(maxlen=8)
        self._cpu = _Cpu(time.monotonic(), {})
        self._open: _Stall | None = None
        self._reported_due = 0.0

    async def run(self) -> None:
        """本任务拥有看门狗线程；线程确实退出后任务才结束。"""
        self._loop_ident = threading.get_ident()
        self._pulse = (time.monotonic(), 0.0)
        thread = threading.Thread(target=self._watch, name="diagnostics-stall-watch", daemon=True)
        thread.start()
        try:
            while True:
                due = time.monotonic() + BEAT_SECONDS
                await asyncio.sleep(BEAT_SECONDS)
                now = time.monotonic()
                self._beat(now, max(0.0, now - due))
        finally:
            self._stop.set()
            await self._drain(thread)

    # 先留迟到边界再更新心跳：看门狗读到新心跳时，这一拍的迟到量一定已在队列里。
    def _beat(self, now: float, lag: float) -> None:
        if lag >= LATE_SECONDS:
            self._late.append((now, lag))
        self._pulse = (now, lag)

    # 在循环上轮询等线程退出，不阻塞循环、不占线程池；重复取消也继续等，不以超时冒充卸载成功。
    async def _drain(self, thread: threading.Thread) -> None:
        started = time.monotonic()
        warned = False
        while thread.is_alive():
            if not warned and time.monotonic() - started >= STALL_SECONDS:
                warned = True
                self._logger.warning("停顿看门狗线程 %.0f 秒仍未退出，继续等待", STALL_SECONDS)
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    def _watch(self) -> None:
        self._watch_ident = threading.get_ident()
        self._cpu = self._read_cpu(time.monotonic())
        tick = time.monotonic()
        while not self._stop.wait(CHECK_SECONDS):
            now = time.monotonic()
            delay, tick = max(0.0, now - tick - CHECK_SECONDS), now
            try:
                self._check(now, delay)
            except Exception:
                # 诊断自身故障不能影响业务；记录一次后停止本代看门狗。
                self._logger.exception("停顿看门狗异常退出")
                return

    # 每次检查只读两个时间戳；仅在停顿时抓栈，健康时低频刷新 CPU 基线。
    def _check(self, now: float, delay: float) -> None:
        beat, lag = self._pulse
        due = beat + BEAT_SECONDS
        stall = self._open
        # 1. 停顿进行中：心跳恢复即按恢复那一拍的迟到量结束；否则到补抓时间点再抓一次。
        if stall is not None:
            if beat >= stall.due:
                recovered = next((row for row in self._late.copy() if row[0] >= stall.due), (beat, lag))
                self._close(stall, *recovered)
            elif len(stall.captures) <= len(FOLLOW_UP_SECONDS) and \
                    now - stall.due >= FOLLOW_UP_SECONDS[len(stall.captures) - 1]:
                self._capture(stall, now, delay)
            return
        # 2. 心跳迟到超过阈值：循环仍被占住，现场就在此刻。
        if now - due >= STALL_SECONDS:
            self._open = _Stall(due=due, started_at=time.time() - (now - due), cpu=self._cpu)
            self._capture(self._open, now, delay)
            return
        # 3. 看门狗自己也被饿住（某线程长时间持有 GIL）且循环先恢复：事后补记，栈在 GIL 释放后抓取。
        #    从迟到留底里找，循环恢复后又走过几拍健康心跳也不漏。
        missed = next((row for row in self._late.copy() if row[1] >= STALL_SECONDS
                       and row[0] - row[1] > self._reported_due), None) if delay >= STALL_SECONDS else None
        if missed is not None:
            beat, lag = missed
            stall = _Stall(due=beat - lag, started_at=time.time() - (now - beat + lag), cpu=self._cpu)
            self._capture(stall, now, delay)
            self._close(stall, beat, lag)
            return
        if now - self._cpu.at >= CPU_BASELINE_SECONDS:
            self._cpu = self._read_cpu(now)

    def _read_cpu(self, now: float) -> _Cpu:
        threads = _python_threads().values()
        return _Cpu(now, _thread_cpu(thread.native_id for thread in threads if thread.native_id is not None))

    # 抓所有 Python 线程的栈：循环线程在前，其余按停顿窗口内 CPU 时间排序。
    def _capture(self, stall: _Stall, now: float, delay: float) -> None:
        frames = sys._current_frames()  # pyright: ignore[reportPrivateUsage]
        threads = _python_threads()
        cpu = self._read_cpu(now)
        locators: dict[str, str] = {}
        rows: list[_ThreadStack] = []
        for ident, frame in frames.items():
            if ident == self._watch_ident:
                continue
            thread = threads.get(ident)
            native = thread.native_id if thread is not None else None
            used = None
            if native is not None and native in cpu.seconds:
                # 基线之后才启动的线程，全部 CPU 时间都落在本窗口内。
                used = round((cpu.seconds[native] - stall.cpu.seconds.get(native, 0.0)) * 1000, 1)
            stack, omitted = _frames(frame, locators)
            rows.append(_ThreadStack(thread.name if thread is not None else "unregistered", native,
                                     ident == self._loop_ident, used, stack, omitted))
        del frames
        rows.sort(key=lambda row: (not row.loop, -(row.cpu_ms or 0.0)))
        window = round((now - stall.cpu.at) * 1000, 1)
        elapsed = round((now - stall.due) * 1000, 1)
        hottest = max(rows, key=lambda row: row.cpu_ms or 0.0, default=None)
        # 至少占满停顿期一半的 CPU 才标记，避免把空闲时的微小差值当作肇事线程。
        if hottest is not None and (hottest.cpu_ms or 0.0) >= min(window, elapsed + BEAT_SECONDS * 1000) / 2:
            hottest.hot = True
        stall.cpu = cpu
        stall.captures.append(_Capture(elapsed, round(delay * 1000, 1), window, rows))

    # 停顿结束才写入一条不可变记录，游标读者不会看到之后被改写的旧记录。
    def _close(self, stall: _Stall, beat: float, lag: float) -> None:
        self._open = None
        self._reported_due = beat - lag
        self._cpu = self._read_cpu(time.monotonic())
        self._records.append("stall", {"started_at": stall.started_at, "duration_ms": round(lag * 1000, 1),
                                       "threshold_ms": STALL_SECONDS * 1000,
                                       "captures": [capture.value() for capture in stall.captures]})
        first = stall.captures[0]
        self._logger.warning("事件循环停顿 %.1f 秒；循环线程位于 %s，占用 CPU 的线程位于 %s",
                             lag, first.top(loop=True) or "未知", first.top(loop=False) or "无")
