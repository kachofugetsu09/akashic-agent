"""普通 runtime_inspection 能力的客户端投影。

客户端只依赖这组请求级只读方法；如何从当前 generation 取得它们由宿主
绑定，不把 Core Snapshot、ComposablePlugin 或私有快照对象带进插件。
"""

from __future__ import annotations

from .services import RuntimeInspectionError, RuntimeInspectionService

__all__ = ["RuntimeInspectionError", "RuntimeInspectionService"]
