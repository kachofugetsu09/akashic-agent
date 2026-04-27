import logging
from collections.abc import Set as AbstractSet
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from agent.tools.base import Tool, ToolResult
from agent.tools.search_backend import KeywordSearchBackend, SearchBackend

logger = logging.getLogger(__name__)

# 元工具（不参与搜索结果，也不出现在 deferred 工具目录里）
_META_TOOLS: frozenset[str] = frozenset({"tool_search"})
_PROGRESS_DESCRIPTION_FIELD = "description"
_PROGRESS_DESCRIPTION_SCHEMA: dict[str, str] = {
    "type": "string",
    "description": (
        "用 5-12 个字说明这次工具调用的意图，只写给用户看的短语。"
        "不要复述工具名，不要粘贴长参数。例如：查看目录、读取配置、搜索健康数据。"
    ),
}


def _schema_properties(parameters: dict[str, Any]) -> dict[str, Any]:
    raw_properties = parameters.get("properties")
    if isinstance(raw_properties, dict):
        return cast(dict[str, Any], raw_properties)
    properties: dict[str, Any] = {}
    parameters["properties"] = properties
    return properties


def _tool_defines_parameter(tool: Tool, name: str) -> bool:
    parameters: dict[str, Any] = tool.parameters or {}
    properties = parameters.get("properties")
    return isinstance(properties, dict) and name in properties


def _with_progress_description(schema: dict[str, Any], tool: Tool) -> dict[str, Any]:
    cloned = cast(dict[str, Any], deepcopy(schema))
    function = cloned.get("function")
    if not isinstance(function, dict):
        return cloned
    function = cast(dict[str, Any], function)
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return cloned
    parameters = cast(dict[str, Any], parameters)
    if _tool_defines_parameter(tool, _PROGRESS_DESCRIPTION_FIELD):
        return cloned
    properties = _schema_properties(parameters)
    properties[_PROGRESS_DESCRIPTION_FIELD] = dict(_PROGRESS_DESCRIPTION_SCHEMA)
    required = parameters.get("required")
    if isinstance(required, list):
        if _PROGRESS_DESCRIPTION_FIELD not in required:
            cast(list[Any], required).append(_PROGRESS_DESCRIPTION_FIELD)
    else:
        parameters["required"] = [_PROGRESS_DESCRIPTION_FIELD]
    return cloned


# ── ToolMeta ──────────────────────────────────────────────────────────────────


@dataclass
class ToolMeta:
    risk: str = "read-only"  # "read-only" | "write" | "external-side-effect"
    always_on: bool = False
    # 可选：3–10 词短语，补充工具名和描述中没有的别名或口语化表达。
    # 不需要重复名称或描述里已有的词——搜索后端自动索引 name + description。
    search_hint: str | None = None


# ── ToolDocument ──────────────────────────────────────────────────────────────


@dataclass
class ToolDocument:
    """工具的索引态视图，派生自 Tool + ToolMeta，供搜索后端使用。

    搜索后端自动索引：name、description。
    search_hint 是可选补充，仅在名称和描述无法覆盖某些口语别名时填写。
    """

    name: str
    description: str
    risk: str
    always_on: bool
    search_hint: str | None
    source_type: str  # "builtin" | "mcp"
    source_name: str  # mcp server 名，builtin 为空字符串

    @classmethod
    def from_tool_and_meta(
        cls,
        tool: "Tool",
        meta: ToolMeta,
        source_type: str = "builtin",
        source_name: str = "",
    ) -> "ToolDocument":
        return cls(
            name=tool.name,
            description=tool.description,
            risk=meta.risk,
            always_on=meta.always_on,
            search_hint=meta.search_hint,
            source_type=source_type,
            source_name=source_name,
        )


# ── ToolRegistry ──────────────────────────────────────────────────────────────


class ToolRegistry:
    """管理所有可用工具"""

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMeta] = {}
        self._documents: dict[str, ToolDocument] = {}
        self._context: dict[str, str] = {}
        self._backend: SearchBackend = backend or KeywordSearchBackend()

    def set_context(self, **kwargs: str) -> None:
        """设置当前会话上下文（channel、chat_id 等），供工具按需读取。"""
        self._context.update(kwargs)

    def get_context(self) -> dict[str, str]:
        return self._context

    def register(
        self,
        tool: Tool,
        *,
        risk: str = "read-only",
        always_on: bool = False,
        search_hint: str | None = None,
        source_type: str = "builtin",
        source_name: str = "",
    ) -> None:
        self._tools[tool.name] = tool
        meta = ToolMeta(
            risk=risk,
            always_on=always_on,
            search_hint=search_hint,
        )
        self._metadata[tool.name] = meta
        doc = ToolDocument.from_tool_and_meta(
            tool, meta, source_type=source_type, source_name=source_name
        )
        self._documents[tool.name] = doc
        self._backend.add(doc)
        logger.debug(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        _ = self._tools.pop(name, None)
        _ = self._metadata.pop(name, None)
        _ = self._documents.pop(name, None)
        self._backend.remove(name)
        logger.debug(f"注销工具: {name}")

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> "Tool | None":
        return self._tools.get(name)

    def get_registered_names(self) -> set[str]:
        """返回当前已注册工具名集合。"""
        return set(self._tools.keys())

    def get_schemas(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        """返回 OpenAI function calling 格式的工具定义列表。

        names 为 None 时返回全量；否则只返回指定名称的工具。
        """
        if names is None:
            return [
                _with_progress_description(t.to_schema(), t)
                for t in self._tools.values()
            ]
        return [
            _with_progress_description(t.to_schema(), t)
            for name, t in self._tools.items()
            if name in names
        ]

    def get_always_on_names(self) -> set[str]:
        """返回标记为 always_on 的工具名称集合。"""
        return {name for name, meta in self._metadata.items() if meta.always_on}

    def get_documents(self) -> list[ToolDocument]:
        """返回所有已注册工具的索引文档列表。"""
        return list(self._documents.values())

    def get_deferred_names(
        self, visible: set[str] | None = None
    ) -> dict[str, object]:
        """返回所有 deferred 工具名，按来源分组。

        visible: 当前 turn 已可见工具名（always_on + preloaded），从结果中排除。
        deferred = 全量注册工具 - always_on - meta_tools - visible
        格式: {"builtin": [...], "mcp": {"server_name": [...], ...}}
        """
        always_on = self.get_always_on_names()
        excluded = always_on | _META_TOOLS | (visible or set())
        builtin: list[str] = []
        mcp: dict[str, list[str]] = {}

        for name, doc in self._documents.items():
            if name in excluded:
                continue
            if doc.source_type == "mcp":
                mcp.setdefault(doc.source_name, []).append(name)
            else:
                builtin.append(name)

        return {
            "builtin": sorted(builtin),
            "mcp": {k: sorted(v) for k, v in sorted(mcp.items())},
        }

    async def execute(self, name: str, arguments: dict[str, Any]) -> str | ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return f"工具 '{name}' 不存在"
        try:
            # 将会话上下文（channel、chat_id）作为低优先级默认值合并进 kwargs，
            # 工具可按需读取，不感知此机制的工具会直接忽略多余的 key。
            merged: dict[str, Any] = {**self._context, **arguments}
            if not _tool_defines_parameter(tool, _PROGRESS_DESCRIPTION_FIELD):
                merged.pop(_PROGRESS_DESCRIPTION_FIELD, None)
            return await tool.execute(**merged)
        except Exception as e:
            logger.error(f"工具 {name} 执行出错: {e}", exc_info=True)
            return f"工具执行出错: {e}"

    def get_schemas_as_doc_results(self, names: list[str]) -> list[dict[str, Any]]:
        """将工具名列表转为与 search() 相同格式的结果列表。

        供 select: 精确加载路径使用，why_matched 固定为"名称:精确匹配"。
        """
        results: list[dict[str, Any]] = []
        for name in names:
            doc = self._documents.get(name)
            if doc:
                results.append(
                    {
                        "name": doc.name,
                        "summary": doc.description[:120],
                        "why_matched": ["名称:精确匹配"],
                        "risk": doc.risk,
                        "always_on": doc.always_on,
                    }
                )
        return results

    def get_mcp_server_names(self) -> set[str]:
        """返回当前已注册的所有 MCP server 名称。"""
        return {
            doc.source_name
            for doc in self._documents.values()
            if doc.source_type == "mcp"
        }

    def get_tool_names_by_source(self, source_type: str, source_name: str) -> set[str]:
        """返回指定来源的所有工具名。"""
        return {
            name
            for name, doc in self._documents.items()
            if doc.source_type == source_type and doc.source_name == source_name
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: AbstractSet[str] | None = None,
    ) -> list[dict[str, Any]]:
        """关键词搜索工具目录，返回匹配的工具信息列表。

        excluded_names: 调用方（当前 turn）传入的排除集合，通常为已可见工具名。
        meta_tools 始终被排除。搜索逻辑委托给 SearchBackend。
        """
        excluded = _META_TOOLS | (excluded_names or set())
        return cast(
            list[dict[str, Any]],
            self._backend.search(
                query=query,
                top_k=top_k,
                allowed_risk=allowed_risk,
                excluded_names=excluded,
            ),
        )
