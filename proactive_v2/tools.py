"""
proactive_v2/tools.py — Tool schemas + execute dispatcher

外部依赖通过 ToolDeps 注入，便于测试和替换。
所有内部函数返回 JSON string（传给 LLM messages）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from proactive_v2.context import AgentTickContext

logger = logging.getLogger(__name__)

_VALID_SKIP_REASONS = frozenset(["no_content", "user_busy", "already_sent_similar", "other"])


# ── 依赖容器 ──────────────────────────────────────────────────────────────

@dataclass
class ToolDeps:
    """所有工具的外部依赖，通过构造注入。"""
    web_fetch_tool: Any = None          # WebFetchTool instance
    memory: Any = None                  # MemoryPort instance
    alert_fn: Any = None                # async () -> list[dict]
    feed_fn: Any = None                 # async (limit) -> list[dict]
    context_fn: Any = None              # async () -> list[dict]
    recent_chat_fn: Any = None          # async (n) -> list[dict]
    ack_fn: Any = None                  # async (compound_key: str, ttl_hours: int) -> None
    alert_ack_fn: Any = None            # async (compound_key: str) -> None  (no TTL, alerts manage own expiry)
    max_chars: int = 8_000


# ── Tool Schemas ──────────────────────────────────────────────────────────

def _schema(name: str, description: str, parameters: dict) -> dict:
    """构造 OpenAI Chat Completions 风格的 function tool schema。"""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOL_SCHEMAS: list[dict] = [
    _schema("get_alert_events",
            "获取当前待处理的 alert 事件列表。首次调用后结果缓存，重复调用直接返回缓存。",
            {"type": "object", "properties": {}, "required": []}),
    _schema("get_content_events",
            "获取 feed 内容事件列表（36h 内未 ACK，按 published_at DESC）。首次调用后缓存。",
            {"type": "object", "properties": {
                "limit": {"type": "integer", "description": "最多返回条数，默认 5", "default": 5},
            }, "required": []}),
    _schema("get_context_data",
            "获取背景上下文数据（Steam 等持续状态）。最多调用一次，重复调用返回缓存。",
            {"type": "object", "properties": {}, "required": []}),
    _schema("web_fetch",
            "抓取指定 URL 的正文内容（text 格式，截断至 max_chars）。失败时返回 error 字段。",
            {"type": "object", "properties": {
                "url": {"type": "string", "description": "要抓取的完整 URL，必须以 http:// 或 https:// 开头"},
            }, "required": ["url"]}),
    _schema("recall_memory",
            "从向量库检索与 query 相关的用户偏好和 profile 记忆。返回 {result: str, hits: int}。",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "检索关键词，建议包含文章标题和核心主题"},
            }, "required": ["query"]}),
    _schema("get_recent_chat",
            "获取最近 n 条聊天记录，用于判断用户当前是否在忙。",
            {"type": "object", "properties": {
                "n": {"type": "integer", "description": "返回条数，默认 20", "default": 20},
            }, "required": []}),
    _schema("mark_interesting",
            (
                "将指定 item 明确标记为「感兴趣」（在 web_fetch + recall_memory 后调用）。"
                "被标记的条目若未被 send_message 引用，将在 ACK 层得到 24h TTL（uncited interesting）。"
                "仅用于明确确认感兴趣的内容；不确定时不要调用。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{id}\"，如 [\"feed-mcp:abc123\"]",
                },
            }, "required": ["item_ids"]}),
    _schema("mark_not_interesting",
            (
                "将指定 item 标记为「本质上不感兴趣」（720h ACK）。"
                "仅用于内容本身无价值的情况；时机问题、抓取失败等不得调用此工具。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{id}\"，如 [\"feed-mcp:abc123\"]",
                },
            }, "required": ["item_ids"]}),
    _schema("send_message",
            (
                "【终止工具】向用户发送消息。调用后 loop 立即结束。"
                "cited_ids 只填实际引用的条目，格式 \"{ack_server}:{id}\"。"
            ),
            {"type": "object", "properties": {
                "text": {"type": "string", "description": "要发送的消息正文"},
                "cited_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "引用的内容复合键列表，格式 \"{ack_server}:{id}\"",
                },
            }, "required": ["text"]}),
    _schema("skip",
            (
                "【终止工具】决定本轮不发送消息。调用后 loop 立即结束。"
                "reason 枚举：no_content / user_busy / already_sent_similar / other"
            ),
            {"type": "object", "properties": {
                "reason": {
                    "type": "string",
                    "description": "跳过原因：no_content | user_busy | already_sent_similar | other",
                    "enum": ["no_content", "user_busy", "already_sent_similar", "other"],
                },
                "note": {"type": "string", "description": "可选补充说明（写入日志）"},
            }, "required": ["reason"]}),
]


# ── 工具实现 ──────────────────────────────────────────────────────────────

async def _get_alert_events(ctx: AgentTickContext, args: dict, *, alert_fn) -> str:
    if ctx._alerts_fetched:
        return json.dumps(ctx.fetched_alerts, ensure_ascii=False)
    events = await alert_fn() if alert_fn else []
    ctx.fetched_alerts = events or []
    ctx._alerts_fetched = True
    return json.dumps(ctx.fetched_alerts, ensure_ascii=False)


async def _get_content_events(ctx: AgentTickContext, args: dict, *, feed_fn, limit: int) -> str:
    if ctx._contents_fetched:
        return json.dumps(ctx.fetched_contents, ensure_ascii=False)
    events = await feed_fn(limit=limit) if feed_fn else []
    ctx.fetched_contents = events or []
    ctx._contents_fetched = True
    return json.dumps(ctx.fetched_contents, ensure_ascii=False)


async def _get_context_data(ctx: AgentTickContext, args: dict, *, context_fn) -> str:
    if ctx._context_fetched:
        return json.dumps(ctx.fetched_context, ensure_ascii=False)
    items = await context_fn() if context_fn else []
    ctx.fetched_context = items or []
    ctx._context_fetched = True
    return json.dumps(ctx.fetched_context, ensure_ascii=False)


async def _web_fetch(ctx: AgentTickContext, args: dict, *, web_fetch_tool, max_chars: int) -> str:
    result_json = await web_fetch_tool.execute(url=args["url"], format="text")
    result = json.loads(result_json)
    if "error" in result:
        return result_json
    text = result.get("text", "")
    truncated_now = len(text) > max_chars
    result["text"] = text[:max_chars]
    result["truncated"] = truncated_now or result.get("truncated", False)
    return json.dumps(result, ensure_ascii=False)


async def _recall_memory(ctx: AgentTickContext, args: dict, *, memory) -> str:
    query = args["query"]
    hits: list[dict] = await memory.retrieve_related(
        query,
        memory_types=["preference", "profile"],
        top_k=5,
    ) or []
    if not hits:
        return json.dumps({"result": "", "hits": 0}, ensure_ascii=False)
    texts = [h.get("text", "") for h in hits if h.get("text")]
    result_text = "\n---\n".join(texts)
    return json.dumps({"result": result_text, "hits": len(hits)}, ensure_ascii=False)


async def _get_recent_chat(ctx: AgentTickContext, args: dict, *, recent_chat_fn) -> str:
    n = args.get("n", 20)
    messages = await recent_chat_fn(n=n) if recent_chat_fn else []
    return json.dumps(messages or [], ensure_ascii=False)


def _mark_interesting(ctx: AgentTickContext, args: dict) -> str:
    """将 item_ids 加入 interesting_set（若未被 discarded）。"""
    item_ids: list[str] = args.get("item_ids", [])
    for key in item_ids:
        if key not in ctx.discarded_item_ids:
            ctx.interesting_item_ids.add(key)
    return json.dumps({"ok": True}, ensure_ascii=False)


def _mark_not_interesting(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    ctx.discarded_item_ids.update(item_ids)
    # 若之前已 mark_interesting，discard 覆盖（两者互斥）
    ctx.interesting_item_ids -= set(item_ids)
    return json.dumps({"ok": True}, ensure_ascii=False)


def _send_message(ctx: AgentTickContext, args: dict) -> str:
    cited: list[str] = args.get("cited_ids", [])
    ctx.final_message = args["text"]
    ctx.cited_item_ids = cited
    ctx.terminal_action = "send"
    # cited 与 discarded 冲突时 cited 优先；cited 加入 interesting_set
    for key in cited:
        ctx.interesting_item_ids.add(key)
        ctx.discarded_item_ids.discard(key)
    return json.dumps({"ok": True}, ensure_ascii=False)


def _skip(ctx: AgentTickContext, args: dict) -> str:
    reason = args["reason"]
    if reason not in _VALID_SKIP_REASONS:
        raise ValueError(f"invalid skip reason: {reason!r}. must be one of {sorted(_VALID_SKIP_REASONS)}")
    ctx.skip_reason = reason
    ctx.skip_note = args.get("note", "")
    ctx.terminal_action = "skip"
    return json.dumps({"ok": True}, ensure_ascii=False)


# ── execute 分发 ──────────────────────────────────────────────────────────

async def execute(tool_name: str, args: dict, ctx: AgentTickContext, deps: ToolDeps) -> str:
    """统一入口：分发工具调用，递增 steps_taken，返回 JSON string。"""
    ctx.steps_taken += 1

    if tool_name == "get_alert_events":
        return await _get_alert_events(ctx, args, alert_fn=deps.alert_fn)

    if tool_name == "get_content_events":
        limit = args.get("limit", 5)
        return await _get_content_events(ctx, args, feed_fn=deps.feed_fn, limit=limit)

    if tool_name == "get_context_data":
        return await _get_context_data(ctx, args, context_fn=deps.context_fn)

    if tool_name == "web_fetch":
        return await _web_fetch(ctx, args, web_fetch_tool=deps.web_fetch_tool, max_chars=deps.max_chars)

    if tool_name == "recall_memory":
        return await _recall_memory(ctx, args, memory=deps.memory)

    if tool_name == "get_recent_chat":
        return await _get_recent_chat(ctx, args, recent_chat_fn=deps.recent_chat_fn)

    if tool_name == "mark_interesting":
        return _mark_interesting(ctx, args)

    if tool_name == "mark_not_interesting":
        return _mark_not_interesting(ctx, args)

    if tool_name == "send_message":
        return _send_message(ctx, args)

    if tool_name == "skip":
        return _skip(ctx, args)

    raise ValueError(f"unknown tool: {tool_name!r}")
