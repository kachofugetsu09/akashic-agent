"""提供不依赖 Python 小版本的 Linux pidfd 边界。"""

from __future__ import annotations

import ctypes
import os
import sys

_LIBC = ctypes.CDLL(None, use_errno=True)


def open_pidfd(pid: int) -> int:
    """打开稳定的 Linux 进程引用，并原样暴露内核错误。"""

    if not sys.platform.startswith("linux"):
        raise RuntimeError("pidfd 仅支持 Linux")
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    function = _LIBC.pidfd_open
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return fd
