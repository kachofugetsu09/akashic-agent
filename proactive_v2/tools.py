"""
proactive_v2/tools.py — Tool schemas + execute dispatcher

数据层已由 DataGateway 预取，agent 只需：
  recall_memory  — 检索偏好记忆（HyDE 正/负假设）
  get_content    — 按需取预 fetch 正文（批量，失败时可降级 web_fetch）
  web_fetch      — 补正文/核实来源页面
  get_recent_chat / mark_interesting / mark_not_interesting / finish_turn
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from agent.prompting import is_context_frame
from proactive_v2.context import AgentTickContext
from proactive_v2.outbound_text import normalize_outbound_text

logger = logging.getLogger(__name__)

_VALID_SKIP_REASONS = frozenset(["no_content", "user_busy", "already_sent_similar", "other"])


# ── 依赖容器 ──────────────────────────────────────────────────────────────

@dataclass
class ToolDeps:
    """所有工具的外部依赖，通过构造注入。"""
    web_fetch_tool: Any = None          # WebFetchTool（降级用）
    web_search_tool: Any = None         # WebSearchTool（可选）
    memory: Any = None                  # MemoryRuntimeFacade instance
    recent_chat_fn: Any = None          # async (n) -> list[dict]
    ack_fn: Any = None                  # async (compound_key: str, ttl_hours: int) -> None
    alert_ack_fn: Any = None            # async (compound_key: str) -> None
    max_chars: int = 8_000


# ── Tool Schemas ──────────────────────────────────────────────────────────

def _schema(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOL_SCHEMAS: list[dict] = [
    _schema("get_alert_events",
            "获取告警事件列表（本 tick 内缓存）。",
            {"type": "object", "properties": {}, "required": []}),

    _schema("get_content_events",
            "获取内容事件列表（本 tick 内缓存）。",
            {"type": "object", "properties": {}, "required": []}),

    _schema("get_context_data",
            "获取上下文数据列表（本 tick 内最多调用一次）。",
            {"type": "object", "properties": {}, "required": []}),

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
            (
                "【优先工具】抓取指定 URL 的正文或直接来源页面。"
                "当当前候选条目已经有明确 URL，且你需要补正文、核实细节、核实规则时，优先使用它。"
                "失败时返回 error 字段。"
            ),
            {"type": "object", "properties": {
                "url": {"type": "string", "description": "要抓取的完整 URL"},
            }, "required": ["url"]}),

    _schema("web_search",
            "搜索网页结果。用于需要额外外部信息时的补充搜索。",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "type": {"type": "string", "description": "可选搜索模式"},
            }, "required": ["query"]}),

    _schema("get_recent_chat",
            "获取最近 n 条聊天记录，用于判断用户当前是否在忙。",
            {"type": "object", "properties": {
                "n": {"type": "integer", "description": "返回条数，默认 20", "default": 20},
            }, "required": []}),

    _schema("message_push",
            (
                "暂存本轮要发送给用户的消息草稿。调用后 loop 不终止，"
                "必须随后调用 finish_turn(decision=reply) 提交。\n"
                "调用过 message_push 后，禁止 finish_turn(decision=skip, ...)，否则报错。\n"
                "每轮只能调用一次，重复调用报错。\n"
                "evidence 只填本轮实际引用的 alert/content 条目复合键。"
            ),
            {"type": "object", "properties": {
                "message": {"type": "string", "description": "要发送给用户的消息内容，必须非空"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "引用的内容复合键列表，格式 \"{ack_server}:{event_id}\"",
                },
            }, "required": ["message"]}),

    _schema("mark_interesting",
            (
                "将指定 item 明确标记为「感兴趣」。只用于你已单独评估且明确相关的条目，"
                "不能因为其中一条相关就把整批不同主题内容一起标记。"
                "被标记但未被 message_push.evidence 引用的条目将得到 24h ACK。"
                "可选传 reason，简短说明为什么 interesting。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{event_id}\"",
                },
                "reason": {
                    "type": "string",
                    "description": "可选，简短说明原因，例如“命中用户关注的 G2/FaZe 动态”",
                },
            }, "required": ["item_ids"]}),

    _schema("mark_not_interesting",
            (
                "将指定 item 标记为「本质上不感兴趣」（720h ACK，30天内不再出现）。\n"
                "仅用于内容本身无价值；时机问题、抓取失败不得调用。"
                "可选传 reason，简短说明为什么 not_interesting。"
            ),
            {"type": "object", "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "复合键列表，格式 \"{ack_server}:{event_id}\"",
                },
                "reason": {
                    "type": "string",
                    "description": "可选，简短说明原因，例如“未命中用户关注白名单”",
                },
            }, "required": ["item_ids"]}),

    _schema("finish_turn",
            (
                "【终止工具】提交本轮决策，调用后 loop 立即结束。\n"
                "decision=reply：要求之前已调用 message_push 暂存草稿，否则报错。\n"
                "decision=skip：本轮不发送，reason 必填。"
            ),
            {"type": "object", "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["reply", "skip"],
                },
                "reason": {
                    "type": "string",
                    "enum": ["no_content", "user_busy", "already_sent_similar", "other"],
                    "description": "decision=skip 时必填",
                },
                "note": {"type": "string", "description": "可选补充说明（写入日志）"},
            }, "required": ["decision"], "allOf": [
                {
                    "if": {
                        "properties": {"decision": {"const": "skip"}},
                        "required": ["decision"],
                    },
                    "then": {"required": ["reason"]},
                }
            ]}),
]

TERMINAL_TOOL_SCHEMAS: list[dict] = [
    schema
    for schema in TOOL_SCHEMAS
    if schema.get("function", {}).get("name") in {"finish_turn"}
]


# ── 工具实现 ──────────────────────────────────────────────────────────────

async def _recall_memory(ctx: AgentTickContext, args: dict, *, memory) -> str:
    query = args["query"]
    hits = await _retrieve_interest_hits(memory=memory, query=query)
    if not hits:
        return json.dumps({"result": "", "hits": 0}, ensure_ascii=False)
    texts = [h.get("text", "") for h in hits if h.get("text")]
    return json.dumps({"result": "\n---\n".join(texts), "hits": len(hits)}, ensure_ascii=False)


async def _retrieve_interest_hits(*, memory, query: str) -> list[dict]:
    if memory is None:
        return []

    from core.memory.runtime_facade import InterestRetrievalRequest

    result = memory.retrieve_interest_block(
        InterestRetrievalRequest(
            query=query,
            top_k=2,
        )
    )
    if asyncio.iscoroutine(result):
        result = await result
    return list(getattr(result, "hits", None) or [])


def _valid_content_ids(ctx: AgentTickContext) -> set[str]:
    """返回本轮真实存在的 compound ID 集合。content=0 时为空集。"""
    if ctx.content_store:
        return set(ctx.content_store.keys())
    # content_store 为空时退回 fetched_contents 元数据（fetch 失败但 meta 存在的情况）
    return {
        f"{e['ack_server']}:{e['event_id']}"
        for e in ctx.fetched_contents
        if e.get("ack_server") and e.get("event_id")
    }


def _get_content(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    valid = _valid_content_ids(ctx)
    unknown = [i for i in item_ids if i not in valid]
    if unknown:
        logger.warning("[proactive_v2] get_content: unknown item_ids=%s (valid=%s)", unknown, sorted(valid))
        return json.dumps(
            {"error": f"以下 id 不在本轮候选列表中，禁止操作：{unknown}。本轮有效 id：{sorted(valid)}"},
            ensure_ascii=False,
        )
    result = {}
    for item_id in item_ids:
        result[item_id] = ctx.content_store.get(item_id, "")
    return json.dumps(result, ensure_ascii=False)


async def _web_fetch(ctx: AgentTickContext, args: dict, *, web_fetch_tool, max_chars: int) -> str:
    if web_fetch_tool is None:
        return json.dumps({"error": "web_fetch tool not configured"}, ensure_ascii=False)
    result_json = await web_fetch_tool.execute(url=args["url"], format="text")
    result = json.loads(result_json)
    if "error" in result:
        return result_json
    text = result.get("text", "")
    truncated_now = len(text) > max_chars
    result["text"] = text[:max_chars]
    result["truncated"] = truncated_now or result.get("truncated", False)
    return json.dumps(result, ensure_ascii=False)


async def _web_search(ctx: AgentTickContext, args: dict, *, web_search_tool) -> str:
    if web_search_tool is None:
        return json.dumps({"error": "web_search tool not configured"}, ensure_ascii=False)
    return await web_search_tool.execute(**args)


async def _get_alert_events(ctx: AgentTickContext, args: dict) -> str:
    ctx.alerts_fetched = True
    return json.dumps(ctx.fetched_alerts, ensure_ascii=False)


async def _get_content_events(ctx: AgentTickContext, args: dict) -> str:
    ctx.contents_fetched = True
    return json.dumps(ctx.fetched_contents, ensure_ascii=False)


async def _get_context_data(ctx: AgentTickContext, args: dict) -> str:
    ctx.context_fetched = True
    return json.dumps(ctx.fetched_context, ensure_ascii=False)


async def _get_recent_chat(ctx: AgentTickContext, args: dict, *, recent_chat_fn) -> str:
    n = args.get("n", 20)
    messages = await recent_chat_fn(n=n) if recent_chat_fn else []
    # 过滤规则：
    #   role=user           → 保留（判断用户是否在忙、最近关心什么）
    #   role=assistant, proactive 为假 → 保留（被动回复，代表用户主动发起的对话上下文）
    #   role=assistant, proactive=True → 过滤（主动推送，不能当成事实被循环引用；
    #                                           去重逻辑由 recent_proactive_fn 独立负责）
    filtered = []
    for m in messages or []:
        content = str(m.get("content") or "")
        if is_context_frame(content):
            continue
        if m.get("role") == "user" or (
            m.get("role") == "assistant" and not m.get("proactive")
        ):
            filtered.append(m)
    return json.dumps(filtered, ensure_ascii=False)


def _mark_interesting(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    reason = str(args.get("reason", "") or "").strip()
    valid = _valid_content_ids(ctx)
    unknown = [i for i in item_ids if i not in valid]
    if unknown and valid:
        logger.warning("[proactive_v2] mark_interesting: unknown item_ids=%s", unknown)
        return json.dumps(
            {"error": f"以下 id 不在本轮候选列表中，无法标记：{unknown}。本轮有效 id：{sorted(valid)}"},
            ensure_ascii=False,
        )
    for key in item_ids:
        if key not in ctx.discarded_item_ids:
            ctx.interesting_item_ids.add(key)
    logger.info(
        "[proactive_v2] classified interesting ids=%s reason=%s",
        item_ids,
        reason or "(none)",
    )
    return json.dumps({"ok": True}, ensure_ascii=False)


def _mark_not_interesting(ctx: AgentTickContext, args: dict) -> str:
    item_ids: list[str] = args.get("item_ids", [])
    reason = str(args.get("reason", "") or "").strip()
    valid = _valid_content_ids(ctx)
    unknown = [i for i in item_ids if i not in valid]
    if unknown and valid:
        logger.warning("[proactive_v2] mark_not_interesting: unknown item_ids=%s", unknown)
        return json.dumps(
            {"error": f"以下 id 不在本轮候选列表中，无法标记：{unknown}。本轮有效 id：{sorted(valid)}"},
            ensure_ascii=False,
        )
    ctx.discarded_item_ids.update(item_ids)
    ctx.interesting_item_ids -= set(item_ids)
    logger.info(
        "[proactive_v2] classified not_interesting ids=%s reason=%s",
        item_ids,
        reason or "(none)",
    )
    return json.dumps({"ok": True}, ensure_ascii=False)


def _valid_evidence_ids(ctx: AgentTickContext) -> set[str]:
    alert_ids = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_alerts
        if e.get("ack_server") and (e.get("event_id") or e.get("id"))
    }
    return _valid_content_ids(ctx) | alert_ids


def _parse_evidence(ctx: AgentTickContext, evidence_raw: object) -> list[str]:
    evidence = (
        [str(item_id) for item_id in evidence_raw if str(item_id).strip()]
        if isinstance(evidence_raw, list)
        else []
    )
    valid_ids = _valid_evidence_ids(ctx)
    if not valid_ids:
        if evidence:
            raise ValueError(f"invalid evidence ids: {evidence}")
        return evidence
    unknown = [item_id for item_id in evidence if item_id not in valid_ids]
    if unknown:
        raise ValueError(f"invalid evidence ids: {unknown}")
    return evidence


def _finish_reply(ctx: AgentTickContext, args: dict) -> str:
    content = normalize_outbound_text(str(args.get("content", "") or ""))
    if not content.strip():
        raise ValueError("finish_reply requires non-empty content")
    evidence = _parse_evidence(ctx, args.get("evidence", []))
    ctx.final_message = content
    ctx.cited_item_ids = evidence
    ctx.terminal_action = "reply"
    for key in evidence:
        ctx.interesting_item_ids.add(key)
        ctx.discarded_item_ids.discard(key)
    return json.dumps({"ok": True}, ensure_ascii=False)


def _finish_skip(ctx: AgentTickContext, args: dict) -> str:
    content = str(args.get("content", "") or "")
    if content.strip():
        raise ValueError("finish_skip does not accept content")
    note = str(args.get("note", "") or "")
    reason = str(args.get("reason", "other") or "other")
    if reason not in _VALID_SKIP_REASONS:
        raise ValueError(
            f"invalid skip reason: {reason!r}. must be one of {sorted(_VALID_SKIP_REASONS)}"
        )
    evidence = _parse_evidence(ctx, args.get("evidence", []))
    if evidence:
        raise ValueError("finish_skip does not accept evidence")
    ctx.skip_reason = reason
    ctx.skip_note = note
    ctx.terminal_action = "skip"
    ctx.cited_item_ids = []
    return json.dumps({"ok": True}, ensure_ascii=False)


def _finish_turn(ctx: AgentTickContext, args: dict) -> str:
    decision = str(args.get("decision", "") or "").strip()
    note = str(args.get("note", "") or "")
    if decision == "reply":
        if not ctx.draft_message.strip():
            raise ValueError("finish_turn(decision=reply) requires prior message_push call")
        ctx.final_message = ctx.draft_message
        ctx.cited_item_ids = list(ctx.draft_evidence)
        for key in ctx.cited_item_ids:
            ctx.interesting_item_ids.add(key)
            ctx.discarded_item_ids.discard(key)
        ctx.draft_message = ""
        ctx.draft_evidence = []
        ctx.terminal_action = "reply"
        return json.dumps({"ok": True}, ensure_ascii=False)
    if decision == "skip":
        if ctx.draft_message.strip():
            raise ValueError(
                "finish_turn(decision=skip) must not follow message_push; call finish_turn(decision=reply) instead"
            )
        reason = str(args.get("reason", "") or "").strip()
        if not reason:
            raise ValueError("finish_turn(decision=skip) requires non-empty reason")
        if reason not in _VALID_SKIP_REASONS:
            raise ValueError(f"invalid skip reason: {reason!r}")
        ctx.skip_reason = reason
        ctx.skip_note = note
        ctx.terminal_action = "skip"
        ctx.cited_item_ids = []
        return json.dumps({"ok": True}, ensure_ascii=False)
    raise ValueError("finish_turn.decision must be one of: reply, skip")


def _message_push(ctx: AgentTickContext, args: dict) -> str:
    if ctx.draft_message.strip():
        raise ValueError("message_push already called this turn; cannot overwrite draft")
    message = normalize_outbound_text(str(args.get("message", "") or ""))
    if not message.strip():
        raise ValueError("message_push requires non-empty message")
    evidence = _parse_evidence(ctx, args.get("evidence", []))
    ctx.draft_message = message
    ctx.draft_evidence = evidence
    return json.dumps({"ok": True}, ensure_ascii=False)


# ── execute 分发 ──────────────────────────────────────────────────────────

async def dispatch(tool_name: str, args: dict, ctx: AgentTickContext, deps: ToolDeps) -> str:
    if tool_name == "get_alert_events":
        return await _get_alert_events(ctx, args)

    if tool_name == "get_content_events":
        return await _get_content_events(ctx, args)

    if tool_name == "get_context_data":
        return await _get_context_data(ctx, args)

    if tool_name == "recall_memory":
        return await _recall_memory(ctx, args, memory=deps.memory)

    if tool_name == "get_content":
        return _get_content(ctx, args)

    if tool_name == "web_fetch":
        return await _web_fetch(ctx, args, web_fetch_tool=deps.web_fetch_tool, max_chars=deps.max_chars)

    if tool_name == "web_search":
        return await _web_search(ctx, args, web_search_tool=deps.web_search_tool)

    if tool_name == "get_recent_chat":
        return await _get_recent_chat(ctx, args, recent_chat_fn=deps.recent_chat_fn)

    if tool_name == "message_push":
        return _message_push(ctx, args)

    if tool_name == "mark_interesting":
        return _mark_interesting(ctx, args)

    if tool_name == "mark_not_interesting":
        return _mark_not_interesting(ctx, args)

    if tool_name == "finish_turn":
        return _finish_turn(ctx, args)

    raise ValueError(f"unknown tool: {tool_name!r}")


async def execute(tool_name: str, args: dict, ctx: AgentTickContext, deps: ToolDeps) -> str:
    ctx.steps_taken += 1
    return await dispatch(tool_name, args, ctx, deps)
