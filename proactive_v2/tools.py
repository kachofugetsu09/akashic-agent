"""
proactive_v2/tools.py — Tool schemas + execute dispatcher

数据层已由 DataGateway 预取，agent 只需：
  recall_memory  — 检索偏好记忆（HyDE 正/负假设）
  get_content    — 按需取预 fetch 正文（批量，失败时可降级 web_fetch）
  web_fetch      — 降级/兜底（content_store 为空时使用）
  get_recent_chat / mark_interesting / mark_not_interesting / send_message / skip
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
    web_fetch_tool: Any = None          # WebFetchTool（降级用）
    memory: Any = None                  # MemoryPort instance
    recent_chat_fn: Any = None          # async (n) -> list[dict]
    ack_fn: Any = None                  # async (compound_key: str, ttl_hours: int) -> None
    alert_ack_fn: Any = None            # async (compound_key: str) -> None
    max_chars: int = 8_000
    # Gateway 数据源（用于 DataGateway 构建，不直接暴露给 agent 工具）
    alert_fn: Any = None
    feed_fn: Any = None
    context_fn: Any = None


# ── Tool Schemas ──────────────────────────────────────────────────────────

def _schema(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOL_SCHEMAS: list[dict] = [
    _schema("recall_memory",
            (
                "从向量库检索用户偏好/profile 记忆。用于判断某条内容是否触碰雷点或符合兴趣。\n"
                "【使用方式】对每条你想评估的内容，写两条 query 分别调用：\n"
                "  1. 负向假设：「如果《标题》是用户完全不感兴趣的内容，用户会怎么评价它」\n"
                "     → 命中雷点记忆 → mark_not_interesting\n"
                "  2. 正向假设：「如果《标题》对用户很有价值，用户为什么会关心它」\n"
                "     → 命中兴趣记忆 → mark_interesting 并准备 get_content\n"
                "返回 {result: str, hits: int}。hits=0 表示无相关记忆，不等于不感兴趣。"
            ),
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "假设性陈述，描述用户对这条内容的正面或负面评价"},
            }, "required": ["query"]}),

    _schema("get_content",
            (
                "从预取缓存中批量获取内容正文。传入 item_ids 列表，返回 {id: text} 映射。\n"
                "text 为空字符串表示预取失败，此时可选择用 web_fetch 降级获取，或凭标题+recall判断。\n"
                "仅对 mark_interesting 的条目调用，雷点无需读正文。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{event_id}\"，如 [\"feed:fmcp_abc123\"]",
                },
            }, "required": ["item_ids"]}),

    _schema("web_fetch",
            "【降级工具】抓取指定 URL 的正文（get_content 返回空时使用）。失败时返回 error 字段。",
            {"type": "object", "properties": {
                "url": {"type": "string", "description": "要抓取的完整 URL"},
            }, "required": ["url"]}),

    _schema("get_recent_chat",
            "获取最近 n 条聊天记录，用于判断用户当前是否在忙。",
            {"type": "object", "properties": {
                "n": {"type": "integer", "description": "返回条数，默认 20", "default": 20},
            }, "required": []}),

    _schema("mark_interesting",
            (
                "将指定 item 明确标记为「感兴趣」。被标记但未被 send_message 引用的条目将得到 24h ACK。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{event_id}\"",
                },
            }, "required": ["item_ids"]}),

    _schema("mark_not_interesting",
            (
                "将指定 item 标记为「本质上不感兴趣」（720h ACK，30天内不再出现）。\n"
                "仅用于内容本身无价值；时机问题、抓取失败不得调用。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{event_id}\"",
                },
            }, "required": ["item_ids"]}),

    _schema("send_message",
            (
                "【终止工具】向用户发送消息，调用后 loop 立即结束。\n"
                "cited_ids 只填实际引用的条目。"
            ),
            {"type": "object", "properties": {
                "text": {"type": "string", "description": "要发送的消息正文"},
                "cited_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "引用的内容复合键列表",
                },
            }, "required": ["text"]}),

    _schema("skip",
            (
                "【终止工具】决定本轮不发送消息，调用后 loop 立即结束。\n"
                "reason 枚举：no_content | user_busy | already_sent_similar | other"
            ),
            {"type": "object", "properties": {
                "reason": {
                    "type": "string",
                    "enum": ["no_content", "user_busy", "already_sent_similar", "other"],
                },
                "note": {"type": "string", "description": "可选补充说明（写入日志）"},
            }, "required": ["reason"]}),
]


# ── 工具实现 ──────────────────────────────────────────────────────────────

async def _recall_memory(ctx: AgentTickContext, args: dict, *, memory) -> str:
    query = args["query"]
    hits: list[dict] = await memory.retrieve_related(
        query,
        memory_types=["preference", "profile"],
        top_k=2,
    ) or []
    if not hits:
        return json.dumps({"result": "", "hits": 0}, ensure_ascii=False)
    texts = [h.get("text", "") for h in hits if h.get("text")]
    return json.dumps({"result": "\n---\n".join(texts), "hits": len(hits)}, ensure_ascii=False)


def _get_content(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    result = {}
    for item_id in item_ids:
        result[item_id] = ctx.content_store.get(item_id, "")
    return json.dumps(result, ensure_ascii=False)


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


async def _get_recent_chat(ctx: AgentTickContext, args: dict, *, recent_chat_fn) -> str:
    n = args.get("n", 20)
    messages = await recent_chat_fn(n=n) if recent_chat_fn else []
    return json.dumps(messages or [], ensure_ascii=False)


def _mark_interesting(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    for key in item_ids:
        if key not in ctx.discarded_item_ids:
            ctx.interesting_item_ids.add(key)
    return json.dumps({"ok": True}, ensure_ascii=False)


def _mark_not_interesting(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    ctx.discarded_item_ids.update(item_ids)
    ctx.interesting_item_ids -= set(item_ids)
    return json.dumps({"ok": True}, ensure_ascii=False)


def _send_message(ctx: AgentTickContext, args: dict) -> str:
    cited: list[str] = args.get("cited_ids", [])
    ctx.final_message = args["text"]
    ctx.cited_item_ids = cited
    ctx.terminal_action = "send"
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
    ctx.steps_taken += 1

    if tool_name == "recall_memory":
        return await _recall_memory(ctx, args, memory=deps.memory)

    if tool_name == "get_content":
        return _get_content(ctx, args)

    if tool_name == "web_fetch":
        return await _web_fetch(ctx, args, web_fetch_tool=deps.web_fetch_tool, max_chars=deps.max_chars)

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
