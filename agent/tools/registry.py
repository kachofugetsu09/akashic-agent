import asyncio
import logging
from collections.abc import Iterable, Set as AbstractSet
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, TypedDict, cast

from agent.tools.base import (
    Tool,
    ToolExecutionContext,
    ToolResult,
    normalize_tool_parameters,
    tool_execution_context_scope,
)
from agent.tools.search_backend import KeywordSearchBackend, SearchBackend
from agent.control.ids import new_operation_id

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


class DeferredToolNames(TypedDict):
    builtin: list[str]
    mcp: dict[str, list[str]]


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


def _validate_structure(
    value: Any,
    schema: dict[str, Any],
    path: str = "",
) -> list[str]:
    """Validate only object shape, leaving domain values to the tool."""

    if schema.get("type") == "object":
        if not isinstance(value, dict):
            return []
        errors: list[str] = []
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"缺少必填字段：{path + '.' + name if path else name}")
        additional = schema.get("additionalProperties")
        for name, child in value.items():
            if name in properties and isinstance(properties[name], dict):
                errors.extend(
                    _validate_structure(
                        child,
                        cast(dict[str, Any], properties[name]),
                        f"{path}.{name}" if path else name,
                    )
                )
            elif additional is False:
                errors.append(f"不允许额外字段：{path + '.' + name if path else name}")
            elif isinstance(additional, dict):
                errors.extend(
                    _validate_structure(
                        child,
                        cast(dict[str, Any], additional),
                        f"{path}.{name}" if path else name,
                    )
                )
        return errors
    if schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            errors: list[str] = []
            for index, item in enumerate(value):
                errors.extend(
                    _validate_structure(item, cast(dict[str, Any], item_schema), f"{path}[{index}]")
                )
            return errors
    return []


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
    preloadable: bool = True
    requires_turn_search: bool = False
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


@dataclass
class _TurnSearchScope:
    turn_id: str
    session_key: str
    attempt: int
    granted: set[str] = field(default_factory=lambda: set[str]())


_TURN_SEARCH_SCOPE: ContextVar[_TurnSearchScope | None] = ContextVar(
    "akashic_turn_search_scope",
    default=None,
)


def begin_turn_search_scope(
    *,
    turn_id: str,
    session_key: str,
    attempt: int,
) -> Token[_TurnSearchScope | None]:
    """为一次 reasoner attempt 建立独立 search 授权域。"""

    if not session_key or attempt < 0:
        raise ValueError("turn search scope 参数无效")
    effective_turn_id = turn_id or f"local:{id(asyncio.current_task())}"
    return _TURN_SEARCH_SCOPE.set(
        _TurnSearchScope(effective_turn_id, session_key, attempt)
    )


def end_turn_search_scope(token: Token[_TurnSearchScope | None]) -> None:
    _TURN_SEARCH_SCOPE.reset(token)


# ── ToolRegistry ──────────────────────────────────────────────────────────────


class ToolRegistry:
    """管理所有可用工具"""

    def __init__(
        self,
        backend: SearchBackend | None = None,
        *,
        follow_runtime_snapshot: bool = True,
        validate_semantic_schema: bool = True,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMeta] = {}
        self._documents: dict[str, ToolDocument] = {}
        self._execution_context: ContextVar[ToolExecutionContext | None] = ContextVar(
            f"akashic_tool_registry_context_{id(self)}",
            default=None,
        )
        self._backend: SearchBackend = backend or KeywordSearchBackend()
        self._snapshot_view = not follow_runtime_snapshot
        self._validate_semantic_schema = validate_semantic_schema

    def fork(
        self,
        *,
        excluded_source_types: set[str] | None = None,
        excluded_sources: set[tuple[str, str]] | None = None,
    ) -> "ToolRegistry":
        backend = deepcopy(self._backend)
        cloned = ToolRegistry(
            backend=backend,
            validate_semantic_schema=self._validate_semantic_schema,
        )
        excluded_types = excluded_source_types or set()
        excluded_pairs = excluded_sources or set()
        names = [
            name
            for name, document in self._documents.items()
            if document.source_type not in excluded_types
            and (document.source_type, document.source_name) not in excluded_pairs
        ]
        cloned._tools = {name: self._tools[name] for name in names}
        cloned._metadata = {name: self._metadata[name] for name in names}
        cloned._documents = {name: self._documents[name] for name in names}
        _ = cloned._execution_context.set(self._execution_context.get())
        cloned._backend.rebuild(list(cloned._documents.values()))
        cloned._snapshot_view = True
        return cloned

    def _runtime_view(self) -> "ToolRegistry":
        if self._snapshot_view:
            return self
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        if snapshot is None or snapshot.tool_registry is None:
            return self
        return snapshot.tool_registry

    def set_context(self, **kwargs: str) -> None:
        """为当前 async task 绑定不可变 runtime provenance。"""

        view = self._runtime_view()
        if view is not self:
            view.set_context(**kwargs)
            return

        allowed = {
            "channel",
            "chat_id",
            "session_key",
            "turn_id",
            "current_timestamp",
            "current_user_source_ref",
            "origin_channel",
            "origin_chat_id",
            "origin_session_key",
        }
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"工具上下文包含未知字段: {', '.join(unknown)}")
        legacy_fields = {"channel", "chat_id", "session_key"}
        origin_fields = {
            "origin_channel",
            "origin_chat_id",
            "origin_session_key",
        }
        if legacy_fields.intersection(kwargs) and origin_fields.intersection(kwargs):
            raise TypeError(
                "工具上下文不能同时使用 legacy channel/chat_id/session_key "
                "和 origin_* 字段"
            )
        self_context = ToolExecutionContext(
            origin_channel=str(
                kwargs.get("origin_channel", kwargs.get("channel", ""))
                or ""
            ),
            origin_chat_id=str(
                kwargs.get("origin_chat_id", kwargs.get("chat_id", ""))
                or ""
            ),
            origin_session_key=str(
                kwargs.get(
                    "origin_session_key",
                    kwargs.get("session_key", ""),
                )
                or ""
            ),
            turn_id=str(kwargs.get("turn_id", "") or ""),
            current_timestamp=str(
                kwargs.get("current_timestamp", "") or ""
            ),
            current_user_source_ref=str(
                kwargs.get(
                    "current_user_source_ref", ""
                )
                or ""
            ),
        )
        # ContextVar is task-local; no registry-owned mutable context remains.
        _ = self._execution_context.set(self_context)

    def get_context(self) -> dict[str, str]:
        """Return a compatibility view for internal callers, never model input."""

        view = self._runtime_view()
        if view is not self:
            return view.get_context()
        context = self._execution_context.get()
        if context is None:
            return {}
        return {
            "channel": context.origin_channel,
            "chat_id": context.origin_chat_id,
            "session_key": context.origin_session_key,
            "turn_id": context.turn_id,
            "current_timestamp": context.current_timestamp,
            "current_user_source_ref": context.current_user_source_ref,
        }

    def get_execution_context(self) -> ToolExecutionContext | None:
        """Return the immutable runtime provenance for internal tool adapters."""

        view = self._runtime_view()
        if view is not self:
            return view.get_execution_context()
        return self._execution_context.get()

    def begin_turn_search_scope(
        self,
        *,
        turn_id: str,
        session_key: str,
        attempt: int,
    ) -> Token[_TurnSearchScope | None]:
        """为一次 reasoner attempt 建立独立 search 授权域。"""

        return begin_turn_search_scope(
            turn_id=turn_id,
            session_key=session_key,
            attempt=attempt,
        )

    def end_turn_search_scope(
        self,
        token: Token[_TurnSearchScope | None],
    ) -> None:
        end_turn_search_scope(token)

    def grant_current_turn_search(self, names: list[str]) -> None:
        """只在当前 attempt 内授权 tool_search 的真实结果。"""

        scope = _TURN_SEARCH_SCOPE.get()
        if scope is not None:
            scope.granted.update(names)

    def register(
        self,
        tool: Tool,
        *,
        risk: str = "read-only",
        always_on: bool = False,
        preloadable: bool = True,
        requires_turn_search: bool = False,
        search_hint: str | None = None,
        source_type: str = "builtin",
        source_name: str = "",
    ) -> None:
        self._tools[tool.name] = tool
        meta = ToolMeta(
            risk=risk,
            always_on=always_on,
            preloadable=preloadable,
            requires_turn_search=requires_turn_search,
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
        view = self._runtime_view()
        if view is not self:
            return view.has_tool(name)
        return name in self._tools

    def get_tool(self, name: str) -> "Tool | None":
        view = self._runtime_view()
        if view is not self:
            return view.get_tool(name)
        return self._tools.get(name)

    def get_registered_names(self) -> set[str]:
        """返回当前已注册工具名集合。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_registered_names()
        return set(self._tools.keys())

    def get_source_tool_names(
        self,
        source_type: str,
        source_name: str,
        *,
        risk: str | None = None,
    ) -> set[str]:
        """按来源与 risk 返回工具名，跟随 runtime snapshot 视图。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_source_tool_names(source_type, source_name, risk=risk)
        names: set[str] = set()
        for name, document in self._documents.items():
            if document.source_type != source_type:
                continue
            if document.source_name != source_name:
                continue
            if risk is not None and document.risk != risk:
                continue
            names.add(name)
        return names

    def get_non_read_only_source_tool_names(
        self,
        source_type: str,
        source_name: str,
    ) -> set[str]:
        """返回指定来源中具有写入或外部副作用的工具名。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_non_read_only_source_tool_names(source_type, source_name)
        return {
            name
            for name, document in self._documents.items()
            if document.source_type == source_type
            and document.source_name == source_name
            and document.risk != "read-only"
        }

    def get_schemas(
        self,
        names: AbstractSet[str] | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """返回 OpenAI function calling 格式的工具定义列表。

        names 为 None 时返回全量；传 set 时按注册顺序过滤；传 list/tuple 时按调用方顺序返回。
        """
        view = self._runtime_view()
        if view is not self:
            return view.get_schemas(names)
        if names is None:
            return [
                _with_progress_description(t.to_schema(), t)
                for t in self._tools.values()
            ]
        if not isinstance(names, AbstractSet):
            return [
                _with_progress_description(tool.to_schema(), tool)
                for name in names
                if (tool := self._tools.get(name)) is not None
            ]
        return [
            _with_progress_description(t.to_schema(), t)
            for name, t in self._tools.items()
            if name in names
        ]

    def get_registered_order(self, names: AbstractSet[str] | None = None) -> list[str]:
        view = self._runtime_view()
        if view is not self:
            return view.get_registered_order(names)
        if names is None:
            return list(self._tools.keys())
        return [name for name in self._tools.keys() if name in names]

    def get_always_on_names(self) -> set[str]:
        """返回标记为 always_on 的工具名称集合。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_always_on_names()
        return {name for name, meta in self._metadata.items() if meta.always_on}

    def get_non_preloadable_names(self) -> set[str]:
        """返回每个新 turn 都必须重新发现的工具。"""

        view = self._runtime_view()
        if view is not self:
            return view.get_non_preloadable_names()
        return {
            name for name, meta in self._metadata.items() if not meta.preloadable
        }

    def get_documents(self) -> list[ToolDocument]:
        """返回所有已注册工具的索引文档列表。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_documents()
        return list(self._documents.values())

    def get_document(self, name: str) -> ToolDocument | None:
        """返回指定工具的索引文档。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_document(name)
        return self._documents.get(name)

    def get_deferred_names(
        self, visible: set[str] | None = None
    ) -> DeferredToolNames:
        """返回所有 deferred 工具名，按来源分组。

        visible: 当前 turn 实际已投影给 provider 的工具名，从结果中排除。
        deferred = 全量注册工具 - meta_tools - visible。provider 上限挤出的
        always_on 工具也必须可发现，否则模型无法通过 tool_search 重新加载。
        格式: {"builtin": [...], "mcp": {"server_name": [...], ...}}
        """
        view = self._runtime_view()
        if view is not self:
            return view.get_deferred_names(visible)
        excluded = _META_TOOLS | (visible or set())
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

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        internal_arguments: dict[str, Any] | None = None,
        raise_errors: bool = False,
        execution_timeout: float | None = None,
    ) -> str | ToolResult:
        view = self._runtime_view()
        if view is not self:
            return await view.execute(
                name,
                arguments,
                internal_arguments=internal_arguments,
                raise_errors=raise_errors,
                execution_timeout=execution_timeout,
            )
        tool = self._tools.get(name)
        if tool is None:
            if raise_errors:
                raise RuntimeError(f"工具 '{name}' 不存在")
            return f"工具 '{name}' 不存在"
        meta = self._metadata[name]
        if not isinstance(arguments, dict):
            message = "工具参数必须是对象"
            if raise_errors:
                raise TypeError(message)
            return message

        validation_arguments = dict(arguments)
        if not _tool_defines_parameter(tool, _PROGRESS_DESCRIPTION_FIELD):
            validation_arguments.pop(_PROGRESS_DESCRIPTION_FIELD, None)
        source_type = self._documents[name].source_type
        parameter_schema = normalize_tool_parameters(
            tool.parameters or {},
            open_object=source_type == "mcp",
        )
        properties = parameter_schema.get("properties", {})
        known_names = set(properties) if isinstance(properties, dict) else set()
        additional = parameter_schema.get("additionalProperties", False)
        unknown_names = (
            []
            if additional is not False
            else sorted(
                str(key) for key in validation_arguments if key not in known_names
            )
        )
        if unknown_names:
            message = f"工具参数无效: 不允许额外字段：{', '.join(unknown_names)}"
            if raise_errors:
                raise ValueError(message)
            return message

        if self._validate_semantic_schema:
            validation_errors = tool.validate_params(
                validation_arguments,
                schema=parameter_schema,
            )
        else:
            validation_errors = _validate_structure(
                validation_arguments,
                parameter_schema,
            )
        if validation_errors:
            message = "; ".join(validation_errors)
            if raise_errors:
                raise ValueError(message)
            return f"工具参数无效: {message}"

        if meta.requires_turn_search:
            scope = _TURN_SEARCH_SCOPE.get()
            from agent.control.context import running_turn_id
            from core.error_context import current_session_key

            active_turn_id = running_turn_id.get()
            active_session_key = current_session_key.get()
            scope_matches_caller = (
                scope is not None
                and bool(active_turn_id)
                and scope.turn_id == active_turn_id
                and scope.session_key == active_session_key
            )
            if (
                not scope_matches_caller
                or scope is None
                or name not in scope.granted
            ):
                message = (
                    f"工具 '{name}' 必须在当前 turn 的当前 attempt 中先通过 "
                    f'tool_search(query="select:{name}") 解锁'
                )
                if raise_errors:
                    raise RuntimeError(message)
                return message
        try:
            merged = dict(validation_arguments)
            if internal_arguments:
                merged.update(internal_arguments)
            base_context = self._execution_context.get()
            execution_context = replace(
                base_context or ToolExecutionContext(),
                execution_id=new_operation_id(),
            )
            with tool_execution_context_scope(execution_context):
                return await tool.execute_with_timeout(
                    merged,
                    execution_timeout=execution_timeout,
                )
        except Exception as e:
            logger.error(f"工具 {name} 执行出错: {e}", exc_info=True)
            if raise_errors:
                raise
            return f"工具执行出错: {e}"

    def get_schemas_as_doc_results(self, names: list[str]) -> list[dict[str, Any]]:
        """将工具名列表转为与 search() 相同格式的结果列表。

        供 select: 精确加载路径使用，why_matched 固定为"名称:精确匹配"。
        """
        view = self._runtime_view()
        if view is not self:
            return view.get_schemas_as_doc_results(names)
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
        view = self._runtime_view()
        if view is not self:
            return view.get_mcp_server_names()
        return {
            doc.source_name
            for doc in self._documents.values()
            if doc.source_type == "mcp"
        }

    def get_tool_names_by_source(self, source_type: str, source_name: str) -> set[str]:
        """返回指定来源的所有工具名。"""
        view = self._runtime_view()
        if view is not self:
            return view.get_tool_names_by_source(source_type, source_name)
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
        view = self._runtime_view()
        if view is not self:
            return view.search(query, top_k, allowed_risk, excluded_names)
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
