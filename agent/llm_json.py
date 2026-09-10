"""兼容入口；LLM JSON 解析由 `agent.plugin_contracts.llm_json` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts.llm_json`。
"""

from agent.plugin_contracts.llm_json import load_json_object_loose

__all__ = ["load_json_object_loose"]
