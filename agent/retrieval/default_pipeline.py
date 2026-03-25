from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.looping.memory_gate import (
    _decide_history_route,
    _format_gate_history,
    _trace_memory_retrieve,
    _trace_route_reason,
)
from agent.looping.ports import LLMServices, MemoryConfig, MemoryServices
from agent.looping.turn_types import HistoryMessage, RetrievalTrace
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from memory2.injection_planner import retrieve_episodic, retrieve_procedure_items
from memory2.query_rewriter import GateDecision

if TYPE_CHECKING:
    from core.observe.events import RagItemTrace, RagTrace

logger = logging.getLogger("agent.retrieval")
_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class DefaultMemoryRetrievalPipeline(MemoryRetrievalPipeline):
    def __init__(
        self,
        memory: MemoryServices,
        memory_config: MemoryConfig,
        llm: LLMServices,
        workspace: Path,
        light_model: str,
    ) -> None:
        self._memory = memory
        self._config = memory_config
        self._llm = llm
        self._workspace = workspace
        self._light_model = light_model
        self._gate_resolver = _GateResolver(
            memory=memory,
            config=memory_config,
            llm=llm,
            light_model=light_model,
        )
        self._episodic_retriever = _EpisodicRetriever(memory=memory, config=memory_config)
        self._finalizer = _MemoryRetrievalFinalizer(
            memory=memory,
            config=memory_config,
            workspace=workspace,
        )

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        block, rag_trace = await _retrieve_memory_block_impl(
            message=request.message,
            session_key=request.session_key,
            channel=request.channel,
            chat_id=request.chat_id,
            history=request.history,
            session_metadata=request.session_metadata,
            gate_resolver=self._gate_resolver,
            episodic_retriever=self._episodic_retriever,
            finalizer=self._finalizer,
        )
        if rag_trace is None:
            return RetrievalResult(block=block)
        trace = RetrievalTrace(
            gate_type=rag_trace.gate_type,
            route_decision=rag_trace.route_decision,
            rewritten_query=rag_trace.query,
            injected_count=sum(
                1 for item in (rag_trace.items or []) if bool(getattr(item, "injected", False))
            ),
            raw=rag_trace,
        )
        return RetrievalResult(block=block, trace=trace)


class _GateResolver:
    def __init__(
        self,
        *,
        memory: MemoryServices,
        config: MemoryConfig,
        llm: LLMServices,
        light_model: str,
    ) -> None:
        self._memory = memory
        self._config = config
        self._llm = llm
        self._light_model = light_model

    @property
    def memory_window(self) -> int:
        return self._config.window

    async def resolve(
        self,
        *,
        message: str,
        session_metadata: dict[str, object],
        recent_turns: str,
    ) -> tuple[dict[str, object], list[dict]]:
        return await _resolve_memory_gate(
            message=message,
            session_metadata=session_metadata,
            recent_turns=recent_turns,
            memory=self._memory,
            config=self._config,
            llm=self._llm,
            light_model=self._light_model,
        )


class _EpisodicRetriever:
    def __init__(
        self,
        *,
        memory: MemoryServices,
        config: MemoryConfig,
    ) -> None:
        self._memory = memory
        self._config = config

    async def retrieve(
        self,
        *,
        message: str,
        recent_turns: str,
        route_decision: str,
        rewritten_query: str,
        history_memory_types: list[str],
        procedure_items: list[dict],
        hyde_context: str,
        sufficiency_trace: dict[str, object],
    ) -> tuple[list[dict], str, str | None, list[dict], str, list[str]]:
        history_items, history_scope_mode, hyde_hypothesis = await _retrieve_episodic_items(
            route_decision=route_decision,
            rewritten_query=rewritten_query,
            history_memory_types=history_memory_types,
            hyde_context=hyde_context,
            memory=self._memory,
            config=self._config,
        )
        selected_items, retrieved_block, injected_item_ids = _build_injection_payload(
            procedure_items=procedure_items,
            history_items=history_items,
            memory=self._memory,
        )
        history_items, history_scope_mode, selected_items, retrieved_block, injected_item_ids = (
            await _retry_empty_episodic_block(
                message=message,
                recent_turns=recent_turns,
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                history_memory_types=history_memory_types,
                procedure_items=procedure_items,
                history_items=history_items,
                history_scope_mode=history_scope_mode,
                selected_items=selected_items,
                retrieved_block=retrieved_block,
                injected_item_ids=injected_item_ids,
                sufficiency_trace=sufficiency_trace,
                memory=self._memory,
                config=self._config,
            )
        )
        return (
            history_items,
            history_scope_mode,
            hyde_hypothesis,
            selected_items,
            retrieved_block,
            injected_item_ids,
        )


class _MemoryRetrievalFinalizer:
    def __init__(
        self,
        *,
        memory: MemoryServices,
        config: MemoryConfig,
        workspace: Path,
    ) -> None:
        self._memory = memory
        self._config = config
        self._workspace = workspace

    def finalize(
        self,
        *,
        session_key: str,
        message: str,
        channel: str,
        chat_id: str,
        gate_type: str,
        route_decision: str,
        rewritten_query: str,
        route_ms: int,
        fallback_reason: str,
        gate_latency_ms: dict[str, int],
        p_items: list[dict],
        h_items: list[dict],
        h_scope_mode: str,
        hyde_hypothesis: str | None,
        selected_items: list[dict],
        retrieved_block: str,
        injected_item_ids: list[str],
        sufficiency_trace: dict[str, object],
    ) -> RagTrace:
        return _finalize_memory_retrieval(
            session_key=session_key,
            message=message,
            channel=channel,
            chat_id=chat_id,
            gate_type=gate_type,
            route_decision=route_decision,
            rewritten_query=rewritten_query,
            route_ms=route_ms,
            fallback_reason=fallback_reason,
            gate_latency_ms=gate_latency_ms,
            p_items=p_items,
            h_items=h_items,
            h_scope_mode=h_scope_mode,
            hyde_hypothesis=hyde_hypothesis,
            selected_items=selected_items,
            retrieved_block=retrieved_block,
            injected_item_ids=injected_item_ids,
            sufficiency_trace=sufficiency_trace,
            workspace=self._workspace,
            config=self._config,
        )

    def trace_exception(
        self,
        *,
        session_key: str,
        message: str,
        channel: str,
        chat_id: str,
        gate_type: str,
        sufficiency_trace: dict[str, object],
        error: Exception,
    ) -> RagTrace:
        _trace_memory_retrieve(
            self._workspace,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            user_msg=message,
            items=[],
            injected_block="",
            gate_type=gate_type,
            fallback_reason="retrieve_exception",
            sufficiency_check=sufficiency_trace,
            error=str(error),
        )
        return _build_agent_rag_trace(
            session_key=session_key,
            user_msg=message,
            rewritten_query=message,
            gate_type=gate_type,
            route_decision=None,
            route_latency_ms=None,
            h_scope_mode=None,
            p_items=[],
            h_items=[],
            hyde_hypothesis=None,
            injected_id_set=set(),
            injected_block="",
            sufficiency_check=sufficiency_trace,
            fallback_reason="retrieve_exception",
            error=str(error),
        )


async def _retrieve_memory_block_impl(
    *,
    message: str,
    session_key: str,
    channel: str,
    chat_id: str,
    history: list[HistoryMessage],
    session_metadata: dict[str, object],
    gate_resolver: _GateResolver,
    episodic_retriever: _EpisodicRetriever,
    finalizer: _MemoryRetrievalFinalizer,
) -> tuple[str, RagTrace | None]:
    retrieved_block = ""
    rag_trace: RagTrace | None = None
    gate_type = "history_route"
    sufficiency_trace: dict[str, object] = _empty_sufficiency_state()
    try:
        main_history = _to_history_dicts(history[-gate_resolver.memory_window :])
        recent_turns = _format_gate_history(main_history, max_turns=3)
        hyde_context = _build_hyde_context(main_history)
        gate_result, p_items = await gate_resolver.resolve(
            message=message,
            session_metadata=session_metadata,
            recent_turns=recent_turns,
        )
        gate_type = str(gate_result["gate_type"])
        rewritten_query = str(gate_result["episodic_query"])
        route_decision = str(gate_result["route_decision"])
        route_ms = int(gate_result["route_latency_ms"])
        fallback_reason = str(gate_result["fallback_reason"])
        history_memory_types = list(gate_result["history_memory_types"])
        gate_latency_ms = {"route": route_ms}

        h_items, h_scope_mode, hyde_hypothesis, selected_items, retrieved_block, injected_item_ids = (
            await episodic_retriever.retrieve(
                message=message,
                recent_turns=recent_turns,
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                history_memory_types=history_memory_types,
                procedure_items=p_items,
                hyde_context=hyde_context,
                sufficiency_trace=sufficiency_trace,
            )
        )
        rag_trace = finalizer.finalize(
            session_key=session_key,
            message=message,
            channel=channel,
            chat_id=chat_id,
            gate_type=gate_type,
            route_decision=route_decision,
            rewritten_query=rewritten_query,
            route_ms=route_ms,
            fallback_reason=fallback_reason,
            gate_latency_ms=gate_latency_ms,
            p_items=p_items,
            h_items=h_items,
            h_scope_mode=h_scope_mode,
            hyde_hypothesis=hyde_hypothesis,
            selected_items=selected_items,
            retrieved_block=retrieved_block,
            injected_item_ids=injected_item_ids,
            sufficiency_trace=sufficiency_trace,
        )
    except Exception as e:
        logger.warning("memory2 retrieve 失败，跳过: %s", e)
        rag_trace = finalizer.trace_exception(
            session_key=session_key,
            message=message,
            channel=channel,
            chat_id=chat_id,
            gate_type=gate_type,
            sufficiency_trace=sufficiency_trace,
            error=e,
        )
    return retrieved_block, rag_trace


def _to_history_dicts(history: list[HistoryMessage]) -> list[dict]:
    out: list[dict] = []
    for msg in history:
        out.append(
            {
                "role": msg.role,
                "content": msg.content,
                "tools_used": list(msg.tools_used),
                "tool_chain": [
                    {
                        "text": group.text,
                        "calls": [
                            {
                                "call_id": call.call_id,
                                "name": call.name,
                                "arguments": call.arguments,
                                "result": call.result,
                            }
                            for call in group.calls
                        ],
                    }
                    for group in msg.tool_chain
                ],
            }
        )
    return out


def _empty_sufficiency_state() -> dict[str, object]:
    return {
        "triggered": False,
        "result": "",
        "refined_query": "",
        "retry_count": 0,
    }


def _build_hyde_context(main_history: list[dict]) -> str:
    now = datetime.now()
    date_str = now.strftime(f"%Y-%m-%d {_WEEKDAY_CN[now.weekday()]} %H:%M")
    hyde_turns = _format_gate_history(
        main_history,
        max_turns=3,
        max_content_len=None,
    )
    return f"当前时间：{date_str}\n{hyde_turns}" if hyde_turns else f"当前时间：{date_str}"


async def _resolve_memory_gate(
    *,
    message: str,
    session_metadata: dict[str, object],
    recent_turns: str,
    memory: MemoryServices,
    config: MemoryConfig,
    llm: LLMServices,
    light_model: str,
) -> tuple[dict[str, object], list[dict]]:
    if memory.query_rewriter is not None:
        decision: GateDecision = await memory.query_rewriter.decide(
            user_msg=message,
            recent_history=recent_turns,
        )
        p_items = await retrieve_procedure_items(
            memory.port,
            query=message,
            top_k=config.top_k_procedure,
        )
        return {
            "gate_type": "query_rewriter",
            "episodic_query": decision.episodic_query,
            "route_decision": "RETRIEVE" if decision.needs_episodic else "NO_RETRIEVE",
            "route_latency_ms": decision.latency_ms,
            "fallback_reason": "",
            "history_memory_types": ["event", "profile"],
        }, p_items
    return await _resolve_fallback_memory_gate(
        message=message,
        session_metadata=session_metadata,
        recent_turns=recent_turns,
        memory=memory,
        config=config,
        llm=llm,
        light_model=light_model,
    )


async def _resolve_fallback_memory_gate(
    *,
    message: str,
    session_metadata: dict[str, object],
    recent_turns: str,
    memory: MemoryServices,
    config: MemoryConfig,
    llm: LLMServices,
    light_model: str,
) -> tuple[dict[str, object], list[dict]]:
    p_task = asyncio.create_task(
        retrieve_procedure_items(
            memory.port,
            query=message,
            top_k=config.top_k_procedure,
        )
    )
    route_task = asyncio.create_task(
        _decide_history_route(
            user_msg=message,
            metadata=session_metadata,
            recent_history=recent_turns,
            light_provider=llm.light_provider,
            light_model=light_model,
            route_intention_enabled=config.route_intention_enabled,
            gate_llm_timeout_ms=config.gate_llm_timeout_ms,
            gate_max_tokens=config.gate_max_tokens,
        )
    )
    p_items, route_decision_obj = await asyncio.gather(p_task, route_task)
    route_reason = _trace_route_reason(route_decision_obj)
    return {
        "gate_type": "history_route",
        "episodic_query": route_decision_obj.rewritten_query,
        "route_decision": "RETRIEVE" if route_decision_obj.needs_history else "NO_RETRIEVE",
        "route_latency_ms": route_decision_obj.latency_ms,
        "fallback_reason": "" if route_reason == "ok" else route_reason,
        "history_memory_types": ["event", "profile"],
    }, p_items


async def _retrieve_episodic_items(
    *,
    route_decision: str,
    rewritten_query: str,
    history_memory_types: list[str],
    hyde_context: str,
    memory: MemoryServices,
    config: MemoryConfig,
) -> tuple[list[dict], str, str | None]:
    if route_decision != "RETRIEVE" or not history_memory_types:
        return [], "disabled", None
    return await retrieve_episodic(
        memory.port,
        rewritten_query,
        memory_types=history_memory_types,
        top_k=config.top_k_history,
        context=hyde_context,
        hyde_enhancer=memory.hyde_enhancer,
    )


def _build_injection_payload(
    *,
    procedure_items: list[dict],
    history_items: list[dict],
    memory: MemoryServices,
) -> tuple[list[dict], str, list[str]]:
    merged = _merge_memory_items(procedure_items + history_items)
    selected_items = memory.port.select_for_injection(merged)
    block, item_ids = memory.port.build_injection_block(merged)
    return selected_items, block, item_ids


async def _retry_empty_episodic_block(
    *,
    message: str,
    recent_turns: str,
    route_decision: str,
    rewritten_query: str,
    history_memory_types: list[str],
    procedure_items: list[dict],
    history_items: list[dict],
    history_scope_mode: str,
    selected_items: list[dict],
    retrieved_block: str,
    injected_item_ids: list[str],
    sufficiency_trace: dict[str, object],
    memory: MemoryServices,
    config: MemoryConfig,
) -> tuple[list[dict], str, list[dict], str, list[str]]:
    checker = memory.sufficiency_checker
    if route_decision != "RETRIEVE" or checker is None or retrieved_block:
        return (
            history_items,
            history_scope_mode,
            selected_items,
            retrieved_block,
            injected_item_ids,
        )
    sufficiency_trace["triggered"] = True
    result = await checker.check(
        query=rewritten_query or message,
        items=selected_items,
        context=recent_turns,
    )
    sufficiency_trace["result"] = result.reason
    sufficiency_trace["refined_query"] = result.refined_query or ""
    if result.is_sufficient or not result.refined_query or not history_memory_types:
        if not history_memory_types and not result.is_sufficient and result.refined_query:
            logger.debug("sufficiency check: no history_memory_types, skip retry")
        return (
            history_items,
            history_scope_mode,
            selected_items,
            retrieved_block,
            injected_item_ids,
        )
    extra_h_items, extra_scope_mode, _retry_hypothesis = await retrieve_episodic(
        memory.port,
        result.refined_query,
        memory_types=history_memory_types,
        top_k=config.top_k_history,
        context=recent_turns,
        hyde_enhancer=memory.hyde_enhancer,
    )
    sufficiency_trace["retry_count"] = 1
    history_items = history_items + extra_h_items
    history_scope_mode = extra_scope_mode or history_scope_mode
    selected_items, retrieved_block, injected_item_ids = _build_injection_payload(
        procedure_items=procedure_items,
        history_items=history_items,
        memory=memory,
    )
    return history_items, history_scope_mode, selected_items, retrieved_block, injected_item_ids


def _merge_memory_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for item in items:
        item_id = str(item.get("id", "") or "")
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        merged.append(item)
    return merged


def _finalize_memory_retrieval(
    *,
    session_key: str,
    message: str,
    channel: str,
    chat_id: str,
    gate_type: str,
    route_decision: str,
    rewritten_query: str,
    route_ms: int,
    fallback_reason: str,
    gate_latency_ms: dict[str, int],
    p_items: list[dict],
    h_items: list[dict],
    h_scope_mode: str,
    hyde_hypothesis: str | None,
    selected_items: list[dict],
    retrieved_block: str,
    injected_item_ids: list[str],
    sufficiency_trace: dict[str, object],
    workspace: Path,
    config: MemoryConfig,
) -> RagTrace:
    logger.info(
        "memory2 retrieve: route=%s scope=%s query=%r p=%d h=%d 命中，选出 %d 条，注入 %d 条%s",
        route_decision,
        h_scope_mode,
        rewritten_query[:50],
        len(p_items),
        len(h_items),
        len(selected_items),
        len(injected_item_ids),
        "" if retrieved_block else "（无内容注入）",
    )
    _log_memory_injection(selected_items)
    sop_guard_applied = _has_sop_guard_hit(
        procedure_items=p_items,
        injected_item_ids=injected_item_ids,
        config=config,
    )
    _trace_memory_retrieve(
        workspace,
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
        user_msg=message,
        items=selected_items,
        injected_block=retrieved_block,
        gate_type=gate_type,
        route_decision=route_decision,
        rewritten_query=rewritten_query,
        fallback_reason=fallback_reason,
        sop_guard_applied=sop_guard_applied,
        procedure_hits=len(p_items),
        history_hits=len(h_items),
        injected_item_ids=injected_item_ids,
        gate_latency_ms=gate_latency_ms,
        sufficiency_check=sufficiency_trace,
    )
    return _build_agent_rag_trace(
        session_key=session_key,
        user_msg=message,
        rewritten_query=rewritten_query,
        gate_type=gate_type,
        route_decision=route_decision,
        route_latency_ms=route_ms,
        h_scope_mode=h_scope_mode,
        p_items=p_items,
        h_items=h_items,
        hyde_hypothesis=hyde_hypothesis,
        injected_id_set=set(injected_item_ids),
        injected_block=retrieved_block,
        sufficiency_check=sufficiency_trace,
        fallback_reason=fallback_reason,
    )


def _log_memory_injection(selected_items: list[dict]) -> None:
    if selected_items:
        injected_preview = " | ".join(
            f"{str(item.get('memory_type', ''))}:{str(item.get('summary', ''))[:40]}"
            for item in selected_items[:4]
            if isinstance(item, dict)
        )
        logger.info("memory2 injected_summary: %s", injected_preview)
    for item in selected_items:
        logger.debug(
            "memory2 injected: id=%s score=%.3f type=%s summary=%s",
            item.get("id", ""),
            float(item.get("score", 0.0)),
            item.get("memory_type", ""),
            str(item.get("summary", ""))[:60],
        )


def _has_sop_guard_hit(
    *,
    procedure_items: list[dict],
    injected_item_ids: list[str],
    config: MemoryConfig,
) -> bool:
    protected_ids = {
        str(item.get("id", ""))
        for item in procedure_items
        if isinstance(item, dict)
        and item.get("memory_type") == "procedure"
        and (item.get("extra_json") or {}).get("tool_requirement")
        and item.get("id")
    }
    return bool(
        config.sop_guard_enabled
        and any(item_id in protected_ids for item_id in injected_item_ids)
    )


def _build_agent_rag_trace(
    *,
    session_key: str,
    user_msg: str,
    rewritten_query: str,
    gate_type: str | None,
    route_decision: str | None,
    route_latency_ms: int | None,
    h_scope_mode: str | None,
    p_items: list[dict],
    h_items: list[dict],
    hyde_hypothesis: str | None,
    injected_id_set: set[str],
    injected_block: str,
    sufficiency_check: dict[str, object],
    fallback_reason: str = "",
    error: str | None = None,
) -> RagTrace:
    from core.observe.events import RagItemTrace, RagTrace

    def _item_to_trace(item: dict, path: str) -> RagItemTrace:
        raw_extra = item.get("extra_json")
        extra_str = json.dumps(raw_extra, ensure_ascii=False) if raw_extra else None
        return RagItemTrace(
            item_id=str(item.get("id", "")),
            memory_type=str(item.get("memory_type", "")),
            score=float(item.get("score", 0.0)),
            summary=str(item.get("summary", "")),
            happened_at=item.get("happened_at"),
            extra_json=extra_str,
            retrieval_path=path,
            injected=str(item.get("id", "")) in injected_id_set,
        )

    trace_items: list[RagItemTrace] = []
    for item in p_items:
        trace_items.append(_item_to_trace(item, "procedure"))
    for item in h_items:
        path = str(item.get("_retrieval_path", "history_raw") or "history_raw")
        trace_items.append(_item_to_trace(item, path))

    return RagTrace(
        source="agent",
        session_key=session_key,
        original_query=user_msg,
        query=rewritten_query,
        gate_type=gate_type,
        route_decision=route_decision,
        route_latency_ms=route_latency_ms,
        hyde_hypothesis=hyde_hypothesis,
        history_scope_mode=h_scope_mode,
        history_gate_reason=None,
        items=trace_items,
        injected_block=injected_block,
        sufficiency_check_json=json.dumps(sufficiency_check, ensure_ascii=False),
        fallback_reason=fallback_reason,
        error=error,
    )
