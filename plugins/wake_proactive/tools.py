from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from core.memory.engine import MemoryQuery, MemoryQueryFilters
from plugins.wake_proactive.context import (
    PreferenceProbe,
    ScratchItem,
    WakeContext,
    content_candidate_map,
    content_event_map,
    event_item_id,
)
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
                            "item_id": {
                                "type": "string",
                                "description": "本轮标题页中的 candidate_N 引用。",
                            },
                            "initial_interest": {
                                "type": "string",
                                "enum": ["likely_interesting", "uncertain"],
                            },
                            "question": {"type": "string"},
                        },
                        "required": ["item_id", "initial_interest"],
                    },
                },
                "preference_probe": {
                    "type": "object",
                    "description": (
                        "可选且每轮最多一个。入选候选的价值取决于用户对一种内容形态或打扰"
                        "类型的态度，且固定上下文没有直接证据时可以填写。主题兴趣和内容形态"
                        "偏好是不同维度；query 查询真实态度和打扰价值，不复述新闻标题。"
                    ),
                    "properties": {
                        "candidate_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": MAX_INVESTIGATION_CANDIDATES,
                        },
                        "topic": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["candidate_ids", "topic", "query"],
                },
            },
            "required": ["items"],
        },
    ),
    _schema(
        "investigate_candidates",
        "按 scratchpad 并发抓取全部正文，并在存在 preference_probe 时只执行一次只读兴趣查询。",
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
                            "item_id": {
                                "type": "string",
                                "description": "本轮标题页中的 candidate_N 引用。",
                            },
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


def _canonical_item_id(
    candidate_map: dict[str, dict[str, Any]], candidate_ref: str
) -> str:
    return event_item_id(candidate_map[candidate_ref])


def _save(ctx: WakeContext, deps: ToolDeps) -> None:
    if deps.state_store is not None:
        deps.state_store.save(ctx)


def _scratchpad(ctx: WakeContext, args: dict[str, Any], deps: ToolDeps) -> str:
    if ctx.screening_completed:
        raise ValueError("scratchpad already recorded for this wake")
    candidate_map = content_candidate_map(ctx)
    valid_ids = set(candidate_map)
    raw_items = cast(list[dict[str, Any]], list(args.get("items") or []))
    if len(raw_items) > MAX_INVESTIGATION_CANDIDATES:
        raise ValueError(
            f"scratchpad supports at most {MAX_INVESTIGATION_CANDIDATES} candidates"
        )
    raw_item_ids = [str(item.get("item_id") or "").strip() for item in raw_items]
    unknown = sorted(set(raw_item_ids) - valid_ids)
    if unknown:
        raise ValueError(f"scratchpad contains unknown item_id: {unknown}")
    item_ids = [
        _canonical_item_id(candidate_map, raw_item_id) for raw_item_id in raw_item_ids
    ]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("scratchpad contains duplicate item_id")

    allowed_interest = {"likely_interesting", "uncertain"}
    planned: dict[str, ScratchItem] = {}
    for raw, candidate_ref, item_id in zip(
        raw_items, raw_item_ids, item_ids, strict=True
    ):
        interest = str(raw["initial_interest"])
        if interest == "not_interesting":
            continue
        if interest not in allowed_interest:
            raise ValueError(f"invalid scratchpad decision for {item_id}")
        planned[item_id] = ScratchItem(
            item_id=item_id,
            initial_interest=cast(Any, interest),
            question=str(raw.get("question") or "").strip(),
        )
    ctx.scratchpad = planned
    planned_candidate_refs = {
        candidate_ref
        for candidate_ref, item_id in zip(raw_item_ids, item_ids, strict=True)
        if item_id in planned
    }
    ctx.preference_probe = _preference_probe(
        args.get("preference_probe"),
        planned_candidate_refs=planned_candidate_refs,
        candidate_map=candidate_map,
    )
    ctx.screening_completed = True
    _save(ctx, deps)
    return json.dumps(
        {
            "ok": True,
            "screened": len(valid_ids),
            "planned": len(ctx.scratchpad),
            "to_investigate": len(ctx.scratchpad),
            "preference_probe": ctx.preference_probe is not None,
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


def _preference_probe(
    raw_probe: object,
    *,
    planned_candidate_refs: set[str],
    candidate_map: dict[str, dict[str, Any]],
) -> PreferenceProbe | None:
    """校验一次批次级偏好探针，并转换候选引用。"""

    # 1. 没有关键偏好歧义时不触发任何记忆查询
    if raw_probe is None:
        return None
    if not isinstance(raw_probe, dict):
        raise ValueError("preference_probe must be an object")
    probe_payload = cast(dict[str, object], raw_probe)

    # 2. 探针只能引用本轮已经入选调查的候选
    raw_candidate_ids = probe_payload.get("candidate_ids")
    if not isinstance(raw_candidate_ids, list):
        raise ValueError("preference_probe candidate_ids must be an array")
    raw_ids = [
        str(item).strip()
        for item in cast(list[object], raw_candidate_ids)
    ]
    if not raw_ids or len(raw_ids) != len(set(raw_ids)):
        raise ValueError("preference_probe candidate_ids must be unique and non-empty")
    unknown = sorted(set(raw_ids) - planned_candidate_refs)
    if unknown:
        raise ValueError(
            f"preference_probe contains unplanned candidate_id: {unknown}"
        )
    topic = str(probe_payload.get("topic") or "").strip()
    query = str(probe_payload.get("query") or "").strip()
    if not topic or not query:
        raise ValueError("preference_probe requires topic and query")
    return PreferenceProbe(
        candidate_ids=tuple(
            _canonical_item_id(candidate_map, candidate_id)
            for candidate_id in raw_ids
        ),
        topic=topic,
        query=query,
    )


async def _recall_preference(
    query: str, *, ctx: WakeContext, deps: ToolDeps, semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    if deps.memory is None:
        return {
            "hits": 0,
            "records": [],
            "trace": {},
            "error": "memory not configured",
        }
    try:
        async with semaphore:
            result = await deps.memory.query(
                MemoryQuery(
                    text=query,
                    intent="interest",
                    effect="read_only",
                    filters=MemoryQueryFilters(relevance_floor="strong"),
                    limit=12,
                    timestamp=ctx.now_utc,
                )
            )
        records = list(result.records)
        return {
            "hits": len(records),
            "records": [
                {
                    "id": record.id,
                    "summary": str(record.summary)[:600],
                    "engine": record.engine_kind,
                }
                for record in records
                if str(record.summary).strip()
            ],
            "trace": {
                key: result.trace.get(key)
                for key in (
                    "engine",
                    "relevance_floor",
                    "native_dense_threshold",
                    "native_score_threshold",
                )
                if key in result.trace
            },
        }
    except Exception as exc:
        logger.warning("wake proactive recall failed query=%r error=%s", query, exc)
        return {"hits": 0, "records": [], "trace": {}, "error": str(exc)}


async def _investigate_candidates(ctx: WakeContext, deps: ToolDeps) -> str:
    if not ctx.screening_completed:
        raise ValueError("investigate_candidates requires scratchpad first")
    if ctx.investigation_completed:
        raise ValueError("investigate_candidates already called this wake")
    events = content_event_map(ctx.content_events)
    candidate_refs = {
        event_item_id(event): candidate_ref
        for candidate_ref, event in content_candidate_map(ctx).items()
    }
    semaphore = asyncio.Semaphore(max(1, deps.max_concurrency))

    async def investigate(item: ScratchItem) -> tuple[str, dict[str, Any]]:
        result: dict[str, Any] = {
            "initial_interest": item.initial_interest,
            "question": item.question,
        }
        result["content"] = await _fetch_content(
            events[item.item_id], deps=deps, semaphore=semaphore
        )
        return item.item_id, result

    item_task = asyncio.gather(*(investigate(item) for item in ctx.scratchpad.values()))
    probe = ctx.preference_probe
    if probe is None:
        pairs = await item_task
        ctx.preference_evidence = {}
    else:
        pairs, evidence = await asyncio.gather(
            item_task,
            _recall_preference(
                probe.query,
                ctx=ctx,
                deps=deps,
                semaphore=semaphore,
            ),
        )
        ctx.preference_evidence = {
            "topic": probe.topic,
            "candidate_ids": list(probe.candidate_ids),
            "query": probe.query,
            **evidence,
        }
    ctx.investigation_results = dict(pairs)
    ctx.investigation_completed = True
    _save(ctx, deps)
    verified_results = {
        candidate_refs[item_id]: result
        for item_id, result in ctx.investigation_results.items()
        if isinstance(result.get("content"), dict)
        and not result["content"].get("error")
        and str(result["content"].get("text") or "").strip()
    }
    return json.dumps(
        {
            "items": verified_results,
            "count": len(verified_results),
            "preference_evidence": ctx.preference_evidence,
        },
        ensure_ascii=False,
    )


def _share_content(ctx: WakeContext, args: dict[str, Any], deps: ToolDeps) -> str:
    if ctx.terminal_action is not None:
        raise ValueError("wake already finished")
    if not ctx.screening_completed or not ctx.investigation_completed:
        raise ValueError("share_content requires scratchpad and investigate_candidates first")
    raw_items = cast(list[dict[str, Any]], list(args.get("items") or []))
    if not raw_items:
        raise ValueError("share_content requires at least one item")
    if len(raw_items) > MAX_SHARE_ITEMS:
        raise ValueError("share_content supports at most 5 items")
    candidate_map = content_candidate_map(ctx)
    raw_item_ids = [
        str(item.get("item_id") or "").strip() for item in raw_items
    ]
    unknown = sorted(set(raw_item_ids) - set(candidate_map))
    if unknown:
        raise ValueError(f"share_content contains unknown item_id: {unknown}")
    item_ids = [
        _canonical_item_id(candidate_map, raw_item_id) for raw_item_id in raw_item_ids
    ]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("share_content contains duplicate item_id")
    items: list[dict[str, Any]] = []
    for raw_item, item_id in zip(raw_items, item_ids, strict=True):
        item = dict(raw_item)
        item["item_id"] = item_id
        items.append(item)
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
