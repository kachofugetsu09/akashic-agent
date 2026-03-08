import logging
from dataclasses import dataclass, field

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

# ── 同义词扩展表 ──────────────────────────────────────────────────────────────
# 短语级：整个短语（或作为 token 的子串）命中后，追加对应搜索词
_PHRASE_SYNONYMS: dict[str, list[str]] = {
    "查看目录": ["list_dir", "ls", "目录"],
    "列出文件": ["list_dir", "ls", "目录"],
    "浏览目录": ["list_dir", "ls", "目录"],
    "文件写入": ["write_file", "write", "写入"],
    "写入文件": ["write_file", "write", "写入"],
    "新建文件": ["write_file", "create", "写入"],
    "创建文件": ["write_file", "create", "写入"],
    "保存文件": ["write_file", "write", "保存"],
    "编辑文件": ["edit_file", "edit", "patch"],
    "修改文件": ["edit_file", "edit", "patch"],
    "更新文件": ["edit_file", "edit", "patch"],
    "读取文件": ["read_file", "read"],
    "查看文件": ["read_file", "read"],
    "定时任务": ["schedule", "cron", "timer"],
    "设置提醒": ["schedule", "remind", "timer"],
    "计划任务": ["schedule", "cron"],
    "延时执行": ["schedule", "delay"],
    "查看提醒": ["list_schedules", "schedule"],
    "定时列表": ["list_schedules", "schedule"],
    "取消定时": ["cancel_schedule", "cancel"],
    "取消提醒": ["cancel_schedule", "cancel"],
    "健康数据": ["fitbit", "health", "步数", "心率"],
    "睡眠报告": ["fitbit", "sleep"],
    "睡眠数据": ["fitbit", "sleep"],
    "推送消息": ["message_push", "push", "通知"],
    "发送消息": ["message_push", "push", "send"],
    "通知用户": ["message_push", "push", "notify"],
    "rss订阅": ["feed", "rss", "subscribe"],
    "订阅管理": ["feed_manage", "subscribe", "rss"],
    "订阅查询": ["feed_query", "feed", "rss"],
    "技能列表": ["skill_action_list", "skill"],
    "技能状态": ["skill_action_status", "skill", "task"],
    "注册技能": ["skill_action_register", "skill"],
    "添加技能": ["skill_action_register", "skill"],
    "删除技能": ["skill_action_unregister", "skill"],
    "记忆存储": ["memorize", "memory"],
    "存储知识": ["memorize", "memory"],
    "网络搜索": ["web_search", "search"],
    "搜索网络": ["web_search", "search"],
    "读取网页": ["web_fetch", "fetch"],
    "抓取网页": ["web_fetch", "fetch"],
    "执行命令": ["shell", "bash", "command"],
    "运行脚本": ["shell", "bash", "script"],
    "更新记忆": ["update_now", "memory"],
    "mcp服务器": ["mcp"],
}

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
    """将查询字符串展开为搜索词集合（原词 + 两层同义词）。

    处理顺序：
    1. 空格切分得到 tokens（含原始词）
    2. 短语级：phrase_key 是 query 或任意 token 的子串 → 追加短语同义词
    3. token 级：token 命中 _TOKEN_SYNONYMS → 追加对应词
    """
    query_lower = query.lower().strip()
    tokens = [t for t in query_lower.split() if t]
    expanded: set[str] = set(tokens)

    for token in tokens:
        # 短语匹配：phrase key 是整个 query 的子串，或是单个 token 的子串
        for phrase, synonyms in _PHRASE_SYNONYMS.items():
            if phrase in query_lower or phrase in token:
                expanded.update(synonyms)
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

    def get_schemas(self, names: set[str] | None = None) -> list[dict]:
        """返回 OpenAI function calling 格式的工具定义列表。

        names 为 None 时返回全量；否则只返回指定名称的工具。
        """
        if names is None:
            return [t.to_schema() for t in self._tools.values()]
        return [
            t.to_schema()
            for name, t in self._tools.items()
            if name in names
        ]

    def get_always_on_names(self) -> set[str]:
        """返回标记为 always_on 的工具名称集合。"""
        return {
            name
            for name, meta in self._metadata.items()
            if meta.always_on
        }

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
                results.append({
                    "name": name,
                    "summary": tool.description[:120],
                    "why_matched": matched_reasons,
                    "key_params": key_params,
                    "tags": meta.tags,
                    "risk": meta.risk,
                    "_score": score,
                })

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

    async def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"工具 '{name}' 不存在"
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            logger.error(f"工具 {name} 执行出错: {e}", exc_info=True)
            return f"工具执行出错: {e}"
