"""手动限时 Python 分配追踪或 CPU 采样；只停止由本实例启动的追踪器。"""
from __future__ import annotations

import asyncio
import copy
import logging
import os
from pathlib import Path
from collections.abc import Sequence
import re
import threading
import time
import tracemalloc
from collections.abc import Callable
from typing import Literal

from agent.plugin_composition.model import ServiceKey
from core.common.file_io import run_file_io
from .sampling import allocator_sample


class ProfileBusy(RuntimeError):
    pass


_INSTALL_ROOT = Path(__file__).resolve().parents[2]

# 安全虚拟标签：tracemalloc 对非文件来源给出的形如 <frozen importlib._bootstrap> 的名字。
# 它不含路径分隔符，不是文件系统路径，原样保留不构成身份泄漏。
_VIRTUAL_LABEL = re.compile(r"^<[^/\\]*>$")


def _export_locator(
    filename: str,
    *,
    roots: Sequence[tuple[Path, str]] | None = None,
) -> str:
    """导出用定位路径：只保留确认安全的相对定位，其他一律退化为基名。"""
    raw = str(filename)
    if _VIRTUAL_LABEL.match(raw):
        return raw
    # 词法折叠 `..`：/home/alice/../bob/x.py 必须先变成 /home/bob/x.py，
    # 否则按 home 前缀匹配会通过，却在输出里带出另一个身份目录。
    path = Path(os.path.normpath(raw))
    if not path.is_absolute():
        # 相对路径同样可能带出身份目录（home/alice/private/app.py），只留基名。
        return path.name
    effective = roots if roots is not None else ((_INSTALL_ROOT, "<install>"), (Path.home(), "~"))
    for root, label in effective:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return label if str(relative) == "." else f"{label}/{relative.as_posix()}"
    # 未知根：不保留任何父目录身份，避免导出文件名泄漏用户名或安装路径。
    return path.name


type CpuSampler = Callable[[threading.Event], dict[str, object]]


class _CpuRun:
    """后台采样线程与停止标记；finish 在 worker 中等线程汇总完成。"""

    def __init__(self, sample: CpuSampler) -> None:
        self._stop = threading.Event()
        self._summary: dict[str, object] | None = None
        self._thread = threading.Thread(target=self._run, args=(sample,), name="diagnostics-cpu-profile", daemon=True)
        self._thread.start()

    def _run(self, sample: CpuSampler) -> None:
        try:
            self._summary = sample(self._stop)
        except Exception:
            logging.getLogger(__name__).exception("CPU 采样线程异常退出")

    def cancel(self) -> None:
        self._stop.set()

    def finish(self) -> dict[str, object]:
        self._stop.set()
        self._thread.join()
        if self._summary is None:
            raise RuntimeError("CPU 采样线程异常退出")
        return self._summary


class Profiles:
    def __init__(self, *, sample_cpu: CpuSampler) -> None:
        self._sample_cpu = sample_cpu
        self._cpu: _CpuRun | None = None
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task[None] | None = None
        self._owned = False
        self._closed = False
        self._result: dict[str, object] = {"state": "idle"}

    def status(self) -> dict[str, object]:
        return copy.deepcopy(self._result)

    async def start(self, seconds: int, kind: Literal["memory", "cpu"] = "memory") -> dict[str, object]:
        """一个进程只开启一轮；其他 owner 的追踪器明确拒绝占用。"""
        if isinstance(seconds, bool) or not isinstance(seconds, int) or not 10 <= seconds <= 120:
            raise ValueError("剖析时长必须是 10～120 秒的整数")
        async with self._lock:
            if self._closed:
                raise ProfileBusy("诊断服务正在关闭")
            if self._cpu is not None:
                raise ProfileBusy("已有 CPU 采样正在运行")
            if kind == "cpu":
                if self._owned:
                    raise ProfileBusy("已有内存剖析正在运行")
                self._cpu = _CpuRun(self._sample_cpu)
                self._result = {"state": "running", "kind": "cpu", "started_at": time.time(),
                                "ends_at": time.time() + seconds, "seconds": seconds}
                self._timer = asyncio.create_task(self._finish_after(seconds), name="diagnostics-profile")
                return self.status()
            if self._owned or tracemalloc.is_tracing():
                raise ProfileBusy("已有内存剖析正在运行")
            # mallinfo2 会持有分配器锁，只在运维发起的剖析首尾各读一次；先读，基线不含追踪器自身分配。
            allocator = await run_file_io(allocator_sample)
            tracemalloc.start(5)
            self._owned = True
            self._result = {"state": "running", "kind": "memory", "started_at": time.time(),
                            "ends_at": time.time() + seconds, "seconds": seconds, "allocator_start": allocator}
            self._timer = asyncio.create_task(self._finish_after(seconds), name="diagnostics-profile")
            return self.status()

    async def _finish_after(self, seconds: int) -> None:
        await asyncio.sleep(seconds)
        await self.stop()

    async def stop(self) -> dict[str, object]:
        """在线程中汇总快照；取消请求仍等待线程完成并关闭追踪。"""
        async with self._lock:
            if not self._owned and self._cpu is None:
                return self.status()
            timer = self._timer
            if timer is not None and timer is not asyncio.current_task():
                timer.cancel()
                try:
                    await timer
                except asyncio.CancelledError:
                    pass
            self._result = {**self._result, "state": "finishing"}
            if self._cpu is not None:
                await self._finish_cpu(self._cpu)
                return self.status()
            try:
                summary = await run_file_io(self._summarize)
                self._result = {**self._result, **summary, "state": "complete", "ended_at": time.time()}
            except asyncio.CancelledError:
                self._result = {**self._result, "state": "cancelled", "ended_at": time.time()}
                raise
            except (MemoryError, RuntimeError) as error:
                # 内存不足或其他进程级探针停止了追踪时，释放本轮并明确报告失败。
                self._result = {**self._result, "state": "failed", "error": type(error).__name__, "ended_at": time.time()}
                logging.getLogger(__name__).exception("内存剖析失败，已停止本轮追踪")
            finally:
                tracemalloc.stop()
                self._owned = False
                if self._result["state"] == "finishing":
                    self._result = {**self._result, "state": "failed", "error": "剖析汇总未完成"}
            self._timer = None
            return self.status()

    # 等采样线程退出再汇总；取消也先等线程结束，失败明确显示 failed。
    async def _finish_cpu(self, run: _CpuRun) -> None:
        try:
            summary = await run_file_io(run.finish)
            self._result = {**self._result, **summary, "state": "complete", "ended_at": time.time()}
        except asyncio.CancelledError:
            self._result = {**self._result, "state": "cancelled", "ended_at": time.time()}
            raise
        except RuntimeError as error:
            self._result = {**self._result, "state": "failed", "error": str(error), "ended_at": time.time()}
        finally:
            self._cpu = None
            self._timer = None

    def _summarize(self) -> dict[str, object]:
        """仅保留统计与文件位置；不读取对象值、源码正文或局部变量。"""
        # 先读分配器视图再拍快照，结束值不含快照自身的分配。
        allocator = allocator_sample()
        current, peak = tracemalloc.get_traced_memory()
        overhead = tracemalloc.get_tracemalloc_memory()
        snapshot = tracemalloc.take_snapshot()
        rows = []
        for stat in snapshot.statistics("traceback")[:30]:
            frames = []
            for frame in stat.traceback:
                frames.append({"file": _export_locator(frame.filename), "line": frame.lineno})
            rows.append({"bytes": stat.size, "count": stat.count, "frames": frames})
        return {"traced_bytes": current, "peak_bytes": peak, "overhead_bytes": overhead, "allocations": rows,
                "allocator_end": allocator}

    async def close(self) -> None:
        """卸载取消定时器并等待正在汇总的线程，不终止其他追踪器。"""
        self._closed = True
        timer = self._timer
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
            try:
                await timer
            except asyncio.CancelledError:
                pass
        async with self._lock:
            if self._owned:
                tracemalloc.stop()
                self._owned = False
            if self._cpu is not None:
                self._cpu.cancel()
                self._cpu = None
            self._result = {"state": "closed"}


PROFILES = ServiceKey[Profiles]("runtime.diagnostics.profiles.v1")
