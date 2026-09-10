"""兼容入口；控制协议值模型由 `agent.plugin_contracts.control_models` 拥有。"""

from agent.plugin_contracts.control_models import *  # noqa: F401,F403
from agent.plugin_contracts.control_models import __dict__ as _impl
from agent.plugin_contracts.control_models import METHOD_PARAMS  # noqa: F401

__all__ = [name for name in _impl if not name.startswith("_") and name not in {"annotations"}]
