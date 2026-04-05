import json
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool
from agent.tools.registry import _META_TOOLS

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry


class ToolSearchTool(Tool):
    """在工具目录中搜索可用工具，帮助模型发现并解锁需要的工具。

    调用此工具后，匹配到的工具将在本轮对话中解锁，可直接调用。
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self._registry = registry
        self._excluded_names: set[str] | None = None

    def set_excluded_names(self, names: set[str] | None) -> None:
        """Explicitly set which tool names are already visible to the LLM.

        Called by the Reasoner before dispatching each tool_search call.
        Replaces the ContextVar mechanism (_excluded_names_ctx) which is now
        kept only as a fallback for callers that have not yet migrated.
        """
        self._excluded_names = names

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "在工具目录中搜索可用工具。搜索结果中的工具将立即解锁，之后可直接调用。\n\n"
            "调用时机：\n"
            "- 需要某类功能，但不知道工具名称 → 必须调用\n"
            "- 知道工具名且已可见 → 直接调用，不要先搜索\n"
            "- 知道工具名但不可见 → 用 select: 前缀精确加载（见下）\n"
            "- 收到'工具不存在'错误 → 必须调用，用错误中的建议关键词搜索\n"
            "- 纯对话/推理，不涉及工具能力 → 不调用\n\n"
            "查询形式：\n"
            "- \"select:工具名\" → 精确加载已知工具，支持逗号分隔多个：\"select:A,B,C\"\n"
            "- \"关键词\" → 模糊搜索，例如：\"定时提醒\"、\"RSS订阅管理\"、\"Fitbit健康数据\"\n\n"
            "正确流程：tool_search(query) → 从结果中选择工具 → 立即调用（不需二次搜索）"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索查询。两种形式：\n"
                        "1. \"select:工具名\" 精确加载（支持逗号分隔多个）\n"
                        "2. 关键词描述功能，例如：\"定时任务\"、\"文件读取\"、\"订阅管理\""
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "关键词搜索时返回的最大工具数量，默认 5，最大 10",
                    "default": 5,
                },
                "allowed_risk": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["read-only", "write", "external-side-effect"],
                    },
                    "description": "允许的风险等级，不填则不过滤。read-only=只读，write=写操作，external-side-effect=外部副作用",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        **_: Any,
    ) -> str:
        # Consume-once: Reasoner calls set_excluded_names() before each dispatch.
        excluded_names = self._excluded_names
        self._excluded_names = None

        query = (query or "").strip()
        if not query:
            return json.dumps(
                {"matched": [], "tip": "query 不能为空，请描述你需要的功能"},
                ensure_ascii=False,
            )

        # ── select: 精确加载路径 ──────────────────────────────────────────
        if query.lower().startswith("select:"):
            return self._handle_select(
                query[7:],
                allowed_risk=allowed_risk,
                excluded_names=excluded_names,
            )

        # ── 关键词搜索路径 ────────────────────────────────────────────────
        top_k = min(max(1, int(top_k)), 10)
        results = self._registry.search(
            query=query,
            top_k=top_k,
            allowed_risk=allowed_risk,
            excluded_names=excluded_names,
        )
        if not results:
            return json.dumps(
                {"matched": [], "tip": "没有找到匹配工具，请换个关键词重试"},
                ensure_ascii=False,
            )
        return json.dumps({"matched": results}, ensure_ascii=False, indent=2)

    def _handle_select(
        self,
        names_str: str,
        *,
        allowed_risk: list[str] | None = None,
        excluded_names: set[str] | None = None,
    ) -> str:
        """处理 select:A,B,C 精确加载路径。

        与 search() 使用相同的过滤语义：
        - excluded_names 中的工具已可见，无需加载（返回 tip 提示直接调用）
        - allowed_risk 不为空时，风险等级不符的工具不返回
        """
        requested = [n.strip() for n in names_str.split(",") if n.strip()]
        if not requested:
            return json.dumps(
                {"matched": [], "tip": "select: 后面需要提供工具名"},
                ensure_ascii=False,
            )

        excluded = _META_TOOLS | (set(excluded_names) if excluded_names else set())
        risk_filter = set(allowed_risk) if allowed_risk else None

        already_loaded: list[str] = []
        found: list[str] = []
        missing: list[str] = []
        risk_blocked: list[str] = []

        for name in requested:
            if name in excluded:
                already_loaded.append(name)
            elif not self._registry.has_tool(name):
                missing.append(name)
            else:
                doc = self._registry._documents.get(name)
                if risk_filter and doc and doc.risk not in risk_filter:
                    risk_blocked.append(name)
                else:
                    found.append(name)

        matched = self._registry.get_schemas_as_doc_results(found)
        result: dict[str, Any] = {"matched": matched}

        tip_parts: list[str] = []
        if already_loaded:
            tip_parts.append(f"已加载可直接调用: {', '.join(already_loaded)}")
        if missing:
            tip_parts.append(
                f"未找到工具: {', '.join(missing)}，请用关键词搜索确认正确名称"
            )
        if risk_blocked:
            tip_parts.append(
                f"风险等级不符（allowed_risk={allowed_risk}）: {', '.join(risk_blocked)}"
            )
        if tip_parts:
            result["tip"] = "; ".join(tip_parts)

        return json.dumps(result, ensure_ascii=False, indent=2)
