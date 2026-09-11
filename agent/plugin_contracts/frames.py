"""控制帧的公开异常词汇。

`FrameRouteReleased` 表示「短生命周期路由已释放，不再接受投递等待」。消费者
（例如 `agent_restart` 工具）必须能捕获它，因此由合同层拥有；实现
`agent/control/frame_book.py` 按原路径再导出。
"""

from __future__ import annotations


class FrameRouteReleased(RuntimeError):
    """短生命周期路由已释放，不再接受投递等待。"""
