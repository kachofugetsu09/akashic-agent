"""只读操作系统计数；缺席和权限拒绝保留原因，不填造零值。"""
from __future__ import annotations

from collections.abc import Callable
import ctypes
from pathlib import Path
import sys
import time


def _fields(path: Path, *, kib: bool = False) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(":")] = int(parts[1]) * (1024 if kib else 1)
    return values


class _MallInfo2(ctypes.Structure):
    _fields_ = [(field, ctypes.c_size_t) for field in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


# glibc 2.33+ 才有 mallinfo2；musl、macOS 与 Windows 没有该符号，按不支持处理。
def _load_mallinfo2() -> Callable[[], _MallInfo2] | None:
    try:
        function = ctypes.CDLL(None).mallinfo2
    except (OSError, AttributeError, TypeError):
        return None
    function.argtypes = []
    function.restype = _MallInfo2
    return function


_MALLINFO2 = _load_mallinfo2() if sys.platform == "linux" else None


def allocator_sample() -> dict[str, object]:
    """glibc 视图区分在用与已释放未归还；其余分配器只标记不支持。"""
    if _MALLINFO2 is None:
        return {"malloc_status": "unsupported"}
    # glibc 逐个 arena 加锁遍历空闲块，期间同一 arena 上的 malloc 要排队，耗时随碎片增长；
    # 释放 GIL 也挡不住这把锁，所以只在运维显式发起的剖析开始和结束时各读一次，不进周期采样。
    started = time.perf_counter()
    info = _MALLINFO2()
    return {"malloc_status": "ok", "malloc_arena_bytes": info.arena, "malloc_mmap_bytes": info.hblkhd,
            "malloc_in_use_bytes": info.uordblks, "malloc_free_bytes": info.fordblks,
            "malloc_top_releasable_bytes": info.keepcost,
            "malloc_scan_ms": round((time.perf_counter() - started) * 1000, 3)}


def memory_sample(*, detailed: bool) -> dict[str, object]:
    """采集当前 Python 进程；cgroup v2 数据始终与进程字段分开。"""
    if sys.platform != "linux":
        return {"process_status": "unsupported", "process_note": "当前平台没有 Linux /proc 指标",
                "cgroup_status": "unsupported"}
    result: dict[str, object] = {}
    # 1. 常规驻留指标便宜；PSS 只在低频样本中读取。
    try:
        status = _fields(Path("/proc/self/status"), kib=True)
        result.update(process_status="ok", rss_bytes=status["VmRSS"],
                      anonymous_bytes=status["RssAnon"], swap_bytes=status["VmSwap"],
                      peak_rss_bytes=status["VmHWM"])
    except (OSError, KeyError) as error:
        result.update(process_status="unavailable", process_note=type(error).__name__)
    if detailed:
        try:
            result["pss_bytes"] = _fields(Path("/proc/self/smaps_rollup"), kib=True)["Pss"]
            result["pss_status"] = "ok"
        except (OSError, KeyError) as error:
            result.update(pss_status="unavailable", pss_note=type(error).__name__)
    # 2. 容器内通常挂载自身 cgroup 根；宿主运行时沿实际 membership 定位。
    try:
        membership = next((line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                           if line.startswith("0::")), None)
        if membership is None:
            result["cgroup_status"] = "unsupported"
            return result
        root = Path("/sys/fs/cgroup")
        group = root / membership.lstrip("/")
        if (root / "memory.current").exists() and membership == "/":
            group = root
        values = _fields(group / "memory.stat")
        limit = (group / "memory.max").read_text().strip()
        result.update(cgroup_status="ok", cgroup_bytes=int((group / "memory.current").read_text()),
                      cgroup_anon_bytes=values["anon"], cgroup_file_bytes=values["file"],
                      cgroup_limit_bytes=None if limit == "max" else int(limit),
                      cgroup_swap_bytes=int((group / "memory.swap.current").read_text()),
                      cgroup_oom_kills=_fields(group / "memory.events")["oom_kill"])
    except (OSError, KeyError, ValueError) as error:
        result.update(cgroup_status="unavailable", cgroup_note=type(error).__name__)
    return result
