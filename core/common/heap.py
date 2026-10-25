from __future__ import annotations

import asyncio
import ctypes
import logging
import time
from collections.abc import Callable
from typing import cast

logger = logging.getLogger(__name__)

# 启动追平、大会话读取等临时峰值释放后，glibc 常因碎片把空闲页留在进程里；
# 定期 malloc_trim 把它们还给系统。prod 副本实测启动后 RSS 532MB → 171MB，单次 < 20ms。
TRIM_INTERVAL_S = 300.0
_FIRST_TRIM_DELAY_S = 60.0
_SLOW_TRIM_MS = 50.0


# 只有 glibc 提供 malloc_trim；musl、macOS、Windows 返回 None，后台任务即为空操作。
def _load_malloc_trim() -> Callable[[int], int] | None:
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return None
    trim = getattr(libc, "malloc_trim", None)
    if trim is None:
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return cast(Callable[[int], int], trim)


# 首次在启动追平之后执行，之后按固定间隔；trim 可能等待 arena 锁，放到工作线程（ctypes 调用释放 GIL），慢时记日志。
async def trim_free_heap_forever(interval_s: float = TRIM_INTERVAL_S) -> None:
    trim = _load_malloc_trim()
    if trim is None:
        return
    delay = _FIRST_TRIM_DELAY_S
    while True:
        await asyncio.sleep(delay)
        delay = interval_s
        started = time.perf_counter()
        _ = await asyncio.to_thread(trim, 0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms >= _SLOW_TRIM_MS:
            logger.info("heap.trim elapsed_ms=%.1f", elapsed_ms)
