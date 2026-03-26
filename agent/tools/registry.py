import logging
from dataclasses import dataclass, field

from agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

# token 级：单个词命中后追加同义词
_TOKEN_SYNONYMS: dict[str, list[str]] = {
    # 文件系统
    "目录": ["list_dir", "dir", "ls"],
    "文件": ["file", "read_file"],
    "写入": ["write", "write_file"],
    "编辑": ["edit", "edit_file"],
    "修改": ["edit", "edit_file"],
    "读取": ["read", "read_file"],
    "保存": ["write", "write_file"],
    # 网络
    "搜索": ["search", "web_search"],
    "查询": ["query", "search"],
    "浏览": ["fetch", "web_fetch"],
    "网页": ["web", "fetch", "web_fetch"],
    # 定时
    "定时": ["schedule", "cron"],
    "提醒": ["schedule", "remind"],
    "计划": ["schedule"],
    "任务": ["task", "schedule", "skill"],
    # 消息推送
    "推送": ["push", "message_push"],
    "通知": ["notify", "message_push", "push"],
    "发送": ["send", "push"],
    "消息": ["message", "push"],
    "对话": ["message", "search_messages"],
    # 订阅
    "订阅": ["feed", "rss", "subscribe"],
    "rss": ["feed", "rss"],
    # 健康
    "健康": ["health", "fitbit"],
    "睡眠": ["sleep", "fitbit"],
    "步数": ["fitbit", "health"],
    "心率": ["fitbit", "health"],
    # 记忆
    "记忆": ["memory", "memorize"],
    "记录": ["memorize", "write"],
    "备忘": ["memorize", "memory"],
    # 技能
    "技能": ["skill"],
    # 系统
    "命令": ["shell", "command"],
    "终端": ["shell", "terminal"],
    "脚本": ["shell", "script"],
    "bash": ["shell"],
    # MCP
    "mcp": ["mcp"],
    # 更新
    "更新": ["update", "edit"],
    "刷新": ["update"],
}


def _expand_query(query: str) -> set[str]:
    """将查询字符串展开为搜索词集合（原词 + token 同义词）。

    处理顺序：
    1. 空格切分得到 tokens（含原始词）
    2. token 级：token 命中 _TOKEN_SYNONYMS → 追加对应词
    各工具领域同义词通过 search_keywords 注册时指定，不在此处硬编码。
    """
    query_lower = query.lower().strip()
    tokens = [t for t in query_lower.split() if t]
    expanded: set[str] = set(tokens)

    for token in tokens:
        # token 级匹配：token 本身，或 token 中包含某个 token_key
        for tk, syns in _TOKEN_SYNONYMS.items():
            if tk == token or tk in token:
                expanded.update(syns)

    return expanded


# ── ToolMeta ──────────────────────────────────────────────────────────────────


@dataclass
class ToolMeta:
    tags: list[str] = field(default_factory=list)
    risk: str = "read-only"  # "read-only" | "write" | "external-side-effect"
    always_on: bool = False
    search_keywords: list[str] = field(default_factory=list)


# ── ToolRegistry ──────────────────────────────────────────────────────────────


class ToolRegistry:
    """管理所有可用工具"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMeta] = {}
        self._context: dict[str, str] = {}

    def set_context(self, **kwargs: str) -> None:
        """设置当前会话上下文（channel、chat_id 等），供工具按需读取。"""
        self._context.update(kwargs)

    def get_context(self) -> dict[str, str]:
        return self._context

    def register(
        self,
        tool: Tool,
        *,
        tags: list[str] | None = None,
        risk: str = "read-only",
        always_on: bool = False,
        search_keywords: list[str] | None = None,
    ) -> None:
        self._tools[tool.name] = tool
        self._metadata[tool.name] = ToolMeta(
            tags=tags or [],
            risk=risk,
            always_on=always_on,
            search_keywords=search_keywords or [],
        )
        logger.debug(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._metadata.pop(name, None)
        logger.debug(f"注销工具: {name}")

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_schemas(self, names: set[str] | None = None) -> list[dict]:
        """返回 OpenAI function calling 格式的工具定义列表。

        names 为 None 时返回全量；否则只返回指定名称的工具。
        """
        if names is None:
            return [t.to_schema() for t in self._tools.values()]
        return [t.to_schema() for name, t in self._tools.items() if name in names]

    def get_always_on_names(self) -> set[str]:
        """返回标记为 always_on 的工具名称集合。"""
        return {name for name, meta in self._metadata.items() if meta.always_on}

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
    ) -> list[dict]:
        """关键词搜索工具目录，返回匹配的工具信息列表。"""
        keywords = _expand_query(query)
        risk_filter = set(allowed_risk) if allowed_risk else None

        results = []
        for name, tool in self._tools.items():
            meta = self._metadata.get(name, ToolMeta())

            if name in ("tool_search", "list_tools"):
                continue

            if risk_filter and meta.risk not in risk_filter:
                continue

            score, matched_reasons = self._score_tool(tool, meta, keywords)
            if score > 0:
                params = tool.parameters or {}
                key_params = list((params.get("properties") or {}).keys())[:5]
                results.append(
                    {
                        "name": name,
                        "summary": tool.description[:120],
                        "why_matched": matched_reasons,
                        "key_params": key_params,
                        "tags": meta.tags,
                        "risk": meta.risk,
                        "_score": score,
                    }
                )

        results.sort(key=lambda x: x["_score"], reverse=True)
        for r in results:
            del r["_score"]
        return results[:top_k]

    @staticmethod
    def _score_tool(
        tool: Tool, meta: ToolMeta, keywords: set[str]
    ) -> tuple[int, list[str]]:
        """给单个工具打分，返回 (score, matched_reasons)。

        权重：name=3, search_keywords=3, tags=2, description=2, param_names=1
        每个 keyword 只取最高命中字段（elif），避免重复计分。
        """
        name_lower = tool.name.lower()
        kw_str = " ".join(meta.search_keywords).lower()
        tag_str = " ".join(meta.tags).lower()
        desc_lower = tool.description.lower()
        param_names = " ".join(
            (tool.parameters or {}).get("properties", {}).keys()
        ).lower()

        score = 0
        reasons: list[str] = []

        for kw in keywords:
            if kw in name_lower:
                score += 3
                reasons.append(f"名称:{kw}")
            elif kw in kw_str:
                score += 3
                reasons.append(f"关键词:{kw}")
            elif kw in tag_str:
                score += 2
                reasons.append(f"标签:{kw}")
            elif kw in desc_lower:
                score += 2
                reasons.append(f"描述:{kw}")
            elif kw in param_names:
                score += 1
                reasons.append(f"参数:{kw}")

        # 去重保序
        seen: set[str] = set()
        deduped = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return score, deduped

    async def execute(self, name: str, arguments: dict) -> str | ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return f"工具 '{name}' 不存在"
        try:
            # 将会话上下文（channel、chat_id）作为低优先级默认值合并进 kwargs，
            # 工具可按需读取，不感知此机制的工具会直接忽略多余的 key。
            merged = {**self._context, **arguments}
            return await tool.execute(**merged)
        except Exception as e:
            logger.error(f"工具 {name} 执行出错: {e}", exc_info=True)
            return f"工具执行出错: {e}"
