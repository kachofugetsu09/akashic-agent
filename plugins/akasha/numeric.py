"""Akasha 唯一的 numpy 入口：小矩阵运算使用单线程 OpenBLAS。"""

from __future__ import annotations

import os

# OpenBLAS 只在库加载时读取线程数，默认按核数起线程并逐个自旋，约耗一核秒 CPU。
# Akasha 的向量运算规模很小，单线程即可；用户显式设置时保持不变。
# 加载后恢复环境，不把这项选择传给其他插件启动的子进程。
_THREADS_ENV = "OPENBLAS_NUM_THREADS"
_configured = _THREADS_ENV in os.environ
if not _configured:
    os.environ[_THREADS_ENV] = "1"
try:
    import numpy as np
finally:
    if not _configured:
        _ = os.environ.pop(_THREADS_ENV, None)

__all__ = ["np"]
