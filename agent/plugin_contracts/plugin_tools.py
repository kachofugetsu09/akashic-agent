"""插件之间共享的工具 view key。

多个插件（akasha、standard_web、tool_search）各自把一组工具暴露给消费者。消费
者不应 import 对方实现模块，因此这些 key 由合同层拥有；它们都只承载
`ToolView`（见 `agent.plugin_contracts.tools`），实现留在各自插件。
"""

from __future__ import annotations

from agent.plugin_composition.model import ServiceKey
from collections.abc import Callable

from agent.plugin_contracts.tools import ToolView
from agent.plugin_contracts.tool_api import ToolPresentation

AKASHA_TOOLS = ServiceKey[ToolView]("akasha.tools.v1")
STANDARD_WEB_TOOLS = ServiceKey[ToolView]("standard-web.tools.v1")
TOOL_SEARCH_TOOLS = ServiceKey[ToolView]("tool-search.tools.v1")
TOOL_SEARCH_PRESENTATION = ServiceKey[Callable[[ToolView], ToolPresentation]](
    "tool-search.presentation.v1"
)
