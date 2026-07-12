from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from core.memory.engine import MemoryQuery
from plugins.wake_proactive.context import ScratchItem, WakeContext, event_item_aliases
from plugins.wake_proactive.renderer import render_share

if TYPE_CHECKING:
    from core.memory.engine import MemoryRetrievalApi
    from plugins.wake_proactive.state import WakeStateStore


logger = logging.getLogger(__name__)
MAX_INVESTIGATION_CANDIDATES = 8
MAX_SHARE_ITEMS = 5


@dataclass
class ToolDeps:
    web_fetch_tool: Any = None
    memory: "MemoryRetrievalApi | None" = None
    state_store: "WakeStateStore | None" = None
    max_chars: int = 8_000
    max_concurrency: int = 6


def _schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "scratchpad",
        "只记录需要查正文或确认用户兴趣的候选。未列出的标题视为本轮不调查，不产生用户反馈或训练标签。",
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": MAX_INVESTIGATION_CANDIDATES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "initial_interest": {
                                "type": "string",
                                "enum": ["likely_interesting", "uncertain"],
                            },
                            "question": {"type": "string"},
                            "recall_query": {"type": "string"},
                        },
                        "required": ["item_id", "initial_interest"],
                    },
                }
            },
            "required": ["items"],
        },
    ),
    _schema(
        "investigate_candidates",
        "按 scratchpad 一次并发完成全部正文抓取和兴趣记忆查询，结果按 item_id 合并。",
        {"type": "object", "properties": {}, "required": []},
    ),
    _schema(
        "share_content",
        "把最终选中的内容渲染成一条自然消息，并保存稳定序号到 event id 的映射。",
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "基于已验证正文写成的一条自然主动消息，不使用固定资讯模板。",
                },
                "opening": {"type": "string"},
                "items": {
                    "type": "array",
                    "maxItems": MAX_SHARE_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "summary": {"type": "string"},
                            "why_it_matters": {"type": "string"},
                        },
                        "required": ["item_id", "summary"],
                    },
                },
                "closing": {"type": "string"},
            },
            "required": ["items"],
        },
    ),
    _schema(
        "skip_content",
        "调查完成后确认本轮没有值得分享的内容；只消费本轮窗口，不产生兴趣反馈标签。",
        {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    ),
]


def _event_map(ctx: WakeContext) -> dict[str, dict[str, Any]]:
    return {
        alias: event
        for event in ctx.content_events
        for alias in event_item_aliases(event)
    }


def _save(ctx: WakeContext, deps: ToolDeps) -> None:
    if deps.state_store is not None:
        deps.state_store.save(ctx)


def _scratchpad(ctx: WakeContext, args: dict[str, Any], deps: ToolDeps) -> str:
    if ctx.screening_completed:
        raise ValueError("scratchpad already recorded for this wake")
    valid_ids = set(_event_map(ctx))
    raw_items = list(args.get("items") or [])
    if len(raw_items) > MAX_INVESTIGATION_CANDIDATES:
        raise ValueError(
            f"scratchpad supports at most {MAX_INVESTIGATION_CANDIDATES} candidates"
        )
    item_ids = [str(item.get("item_id") or "").strip() for item in raw_items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("scratchpad contains duplicate item_id")
    unknown = sorted(set(item_ids) - valid_ids)
    if unknown:
        raise ValueError(f"scratchpad contains unknown item_id: {unknown}")

    allowed_interest = {"likely_interesting", "uncertain"}
    planned: dict[str, ScratchItem] = {}
    for raw in raw_items:
        item_id = str(raw["item_id"]).strip()
        interest = str(raw["initial_interest"])
        if interest == "not_interesting":
            continue
        if interest not in allowed_interest:
            raise ValueError(f"invalid scratchpad decision for {item_id}")
        recall_query = str(raw.get("recall_query") or "").strip()
        if interest == "uncertain" and not recall_query:
            recall_query = str(_event_map(ctx)[item_id].get("title") or item_id)
        planned[item_id] = ScratchItem(
            item_id=item_id,
            initial_interest=cast(Any, interest),
            investigate="content" if interest == "likely_interesting" else "both",
            question=str(raw.get("question") or "").strip(),
            recall_query=recall_query,
        )
    ctx.scratchpad = planned
    ctx.screening_completed = True
    _save(ctx, deps)
    return json.dumps(
        {
            "ok": True,
            "screened": len(valid_ids),
            "planned": len(ctx.scratchpad),
            "to_investigate": len(ctx.scratchpad),
        },
        ensure_ascii=False,
    )


async def _fetch_content(
    event: dict[str, Any], *, deps: ToolDeps, semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    url = str(event.get("url") or "").strip()
    if not url:
        inline = str(event.get("content") or event.get("body") or "")
        return {"text": inline[: deps.max_chars], "url": "", "truncated": len(inline) > deps.max_chars}
    if deps.web_fetch_tool is None:
        return {"error": "web_fetch tool not configured", "url": url}
    try:
        async with semaphore:
            raw = await deps.web_fetch_tool.execute(url=url, format="text")
        result = json.loads(raw)
        if "error" in result:
            return result
        text = str(result.get("text") or "")
        result["text"] = text[: deps.max_chars]
        result["truncated"] = bool(result.get("truncated")) or len(text) > deps.max_chars
        return result
    except Exception as exc:
        logger.warning("wake proactive web fetch failed url=%s error=%s", url, exc)
        return {"error": str(exc), "url": url}


async def _recall(
    query: str, *, ctx: WakeContext, deps: ToolDeps, semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    if deps.memory is None:
        return {"hits": 0, "result": ""}
    try:
        async with semaphore:
            result = await deps.memory.query(
                MemoryQuery(
                    text=query,
                    intent="interest",
                    effect="read_only",
                    limit=2,
                    timestamp=ctx.now_utc,
                )
            )
        records = list(result.records)
        return {
            "hits": len(records),
            "result": "\n---\n".join(
                str(record.summary) for record in records if str(record.summary).strip()
            ),
        }
    except Exception as exc:
        logger.warning("wake proactive recall failed query=%r error=%s", query, exc)
        return {"hits": 0, "result": "", "error": str(exc)}


async def _investigate_candidates(ctx: WakeContext, deps: ToolDeps) -> str:
    if not ctx.screening_completed:
        raise ValueError("investigate_candidates requires scratchpad first")
    if ctx.investigation_completed:
        raise ValueError("investigate_candidates already called this wake")
    events = _event_map(ctx)
    semaphore = asyncio.Semaphore(max(1, deps.max_concurrency))

    async def investigate(item: ScratchItem) -> tuple[str, dict[str, Any]]:
        result: dict[str, Any] = {
            "initial_interest": item.initial_interest,
            "question": item.question,
        }
        operations: list[tuple[str, Any]] = []
        if item.investigate in {"content", "both"}:
            operations.append(("content", _fetch_content(events[item.item_id], deps=deps, semaphore=semaphore)))
        if item.investigate in {"recall", "both"}:
            operations.append(("memory", _recall(item.recall_query, ctx=ctx, deps=deps, semaphore=semaphore)))
        if operations:
            values = await asyncio.gather(*(operation for _, operation in operations))
            result.update({name: value for (name, _), value in zip(operations, values)})
        return item.item_id, result

    pairs = await asyncio.gather(*(investigate(item) for item in ctx.scratchpad.values()))
    ctx.investigation_results = dict(pairs)
    ctx.investigation_completed = True
    _save(ctx, deps)
    verified_results = {
        item_id: result
        for item_id, result in ctx.investigation_results.items()
        if isinstance(result.get("content"), dict)
        and not result["content"].get("error")
        and str(result["content"].get("text") or "").strip()
    }
    return json.dumps(
        {"items": verified_results, "count": len(verified_results)},
        ensure_ascii=False,
    )


def _share_content(ctx: WakeContext, args: dict[str, Any], deps: ToolDeps) -> str:
    if ctx.terminal_action is not None:
        raise ValueError("wake already finished")
    if not ctx.screening_completed or not ctx.investigation_completed:
        raise ValueError("share_content requires scratchpad and investigate_candidates first")
    items = list(args.get("items") or [])
    if not items:
        raise ValueError("share_content requires at least one item")
    if len(items) > MAX_SHARE_ITEMS:
        raise ValueError("share_content supports at most 5 items")
    valid_ids = set(_event_map(ctx))
    item_ids = [str(item.get("item_id") or "").strip() for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("share_content contains duplicate item_id")
    unknown = sorted(set(item_ids) - valid_ids)
    if unknown:
        raise ValueError(f"share_content contains unknown item_id: {unknown}")
    with_evidence: list[dict[str, Any]] = []
    for item_id in item_ids:
        planned = ctx.scratchpad[item_id]
        investigated = ctx.investigation_results.get(item_id) or {}
        content = investigated.get("content")
        if planned.initial_interest == "not_interesting":
            continue
        if not isinstance(content, dict):
            continue
        typed_content = cast(dict[str, Any], content)
        if typed_content.get("error") or not str(
            typed_content.get("text") or ""
        ).strip():
            continue
        with_evidence.append(items[item_ids.index(item_id)])
    if not with_evidence:
        ctx.terminal_action = "skip"
        _save(ctx, deps)
        return json.dumps(
            {"ok": True, "decision": "skip", "reason": "没有可验证的正文证据"},
            ensure_ascii=False,
        )
    rendered = render_share(
        message=str(args.get("message") or ""),
        opening=str(args.get("opening") or ""),
        items=with_evidence,
        closing=str(args.get("closing") or ""),
        events=ctx.content_events,
    )
    ctx.final_message = rendered.message
    ctx.cited_item_ids = rendered.evidence
    ctx.display_event_map = rendered.display_event_map
    ctx.source_refs = rendered.source_refs
    ctx.terminal_action = "reply"
    _save(ctx, deps)
    return json.dumps(
        {
            "ok": True,
            "message": ctx.final_message,
            "display_event_map": ctx.display_event_map,
        },
        ensure_ascii=False,
    )


def _skip_content(ctx: WakeContext, args: dict[str, Any], deps: ToolDeps) -> str:
    if ctx.terminal_action is not None:
        raise ValueError("wake already finished")
    if not ctx.screening_completed or not ctx.investigation_completed:
        raise ValueError("skip_content requires scratchpad and investigate_candidates first")
    reason = str(args.get("reason") or "").strip()
    if not reason:
        raise ValueError("skip_content requires reason")
    ctx.terminal_action = "skip"
    _save(ctx, deps)
    return json.dumps({"ok": True, "decision": "skip", "reason": reason}, ensure_ascii=False)


async def execute(
    tool_name: str, args: dict[str, Any], ctx: WakeContext, deps: ToolDeps
) -> str:
    ctx.steps_taken += 1
    if tool_name == "scratchpad":
        return _scratchpad(ctx, args, deps)
    if tool_name == "investigate_candidates":
        return await _investigate_candidates(ctx, deps)
    if tool_name == "share_content":
        return _share_content(ctx, args, deps)
    if tool_name == "skip_content":
        return _skip_content(ctx, args, deps)
    raise ValueError(f"unknown wake proactive tool: {tool_name!r}")
