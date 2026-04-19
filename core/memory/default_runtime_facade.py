from __future__ import annotations

import inspect
import logging
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, TypedDict, cast

from agent.looping.memory_gate import (
    _decide_history_route,
    _format_gate_history,
    _trace_memory_retrieve,
    _trace_route_reason,
)
from core.memory.engine import (
    MemoryHit,
    MemoryEngineRetrieveResult,
    MemoryEngineRetrieveRequest,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryScope,
    RememberRequest,
    RememberResult,
)
from memory2.query_rewriter import GateDecision
from core.memory.runtime_facade import (
    ConsolidationRunner,
    ContextRetrievalRequest,
    ContextRetrievalResult,
    ContextRetriever,
    InterestRetrievalRequest,
    InterestRetrievalResult,
)

if TYPE_CHECKING:
    from agent.postturn.protocol import PostTurnEvent
    from agent.core.types import HistoryMessage
    from agent.looping.ports import LLMServices, MemoryConfig, MemoryServices
    from core.observe.events import RagTrace
    from agent.core.types import ToolCallGroup
    from core.memory.engine import MemoryEngine
    from core.memory.port import MemoryPort
    from core.memory.profile import ProfileMaintenanceStore


InterestRetriever = Callable[[InterestRetrievalRequest], Awaitable[InterestRetrievalResult]]
logger = logging.getLogger("memory.runtime_facade")


class _GateResult(TypedDict):
    gate_type: str | None
    episodic_query: str
    route_decision: str
    route_latency_ms: int
    fallback_reason: str
    history_memory_types: list[str]


class _RawItemsPayload(TypedDict, total=False):
    items: list[dict]


class _HitLike(Protocol):
    id: str
    score: float
    injected: bool
    metadata: dict[str, object]


class DefaultMemoryRuntimeFacade:
    def __init__(
        self,
        *,
        port: "MemoryPort",
        engine: "MemoryEngine | None" = None,
        profile_maint: "ProfileMaintenanceStore | None" = None,
        context_retriever: ContextRetriever | None = None,
        retrieval_semantics: ContextRetriever | None = None,
        consolidation_runner: ConsolidationRunner | None = None,
        interest_retriever: InterestRetriever | None = None,
    ) -> None:
        self._port = port
        self._engine = engine
        self._profile_maint = profile_maint or port
        self._context_retriever = context_retriever
        self._retrieval_semantics = retrieval_semantics
        self._consolidation_runner = consolidation_runner
        self._interest_retriever = interest_retriever

    def bind_context_retriever(self, retriever: ContextRetriever) -> None:
        self._context_retriever = retriever

    def bind_retrieval_semantics(self, retriever: ContextRetriever) -> None:
        self._retrieval_semantics = retriever

    def bind_consolidation_runner(self, runner: ConsolidationRunner) -> None:
        self._consolidation_runner = runner

    async def _reinforce_context_result(
        self,
        result: ContextRetrievalResult,
    ) -> ContextRetrievalResult:
        item_ids = [
            str(item_id)
            for item_id in result.injected_item_ids
            if str(item_id or "").strip()
        ]
        if not item_ids:
            return result
        try:
            await asyncio.to_thread(self._port.reinforce_items_batch, item_ids)
        except Exception as e:
            logger.warning("[memory_runtime_facade] reinforce_items_batch failed: %s", e)
        return result

    async def ingest_post_turn(self, event: PostTurnEvent) -> MemoryIngestResult:
        # 1. 先保证 post-turn 入口和旧 pipeline 一样只依赖 engine。
        if self._engine is None:
            return MemoryIngestResult(
                accepted=False,
                summary="memory engine unavailable",
                raw={"reason": "engine_unavailable"},
            )

        # 2. 按旧 post-turn 语义组装标准 ingest request。
        # 2.1 tool_chain 保持旧结构，避免 worker 侧归一化漂移。
        # 2.2 source_ref 继续固定到 @post_response。
        source_ref = f"{event.session_key}@post_response"
        return await self._engine.ingest(
            MemoryIngestRequest(
                content={
                    "user_message": event.user_message,
                    "assistant_response": event.assistant_response,
                    "tool_chain": [_tool_group_to_dict(group) for group in event.tool_chain],
                    "source_ref": source_ref,
                },
                source_kind="conversation_turn",
                scope=MemoryScope(
                    session_key=event.session_key,
                    channel=event.channel,
                    chat_id=event.chat_id,
                ),
                metadata={"source_ref": source_ref},
            )
        )

    async def retrieve_context(
        self, request: ContextRetrievalRequest
    ) -> ContextRetrievalResult:
        # 1. 先允许外部用 callback 强行覆盖 retrieval 行为。
        if self._context_retriever is not None:
            result = await self._context_retriever(request)
            return await self._reinforce_context_result(result)

        # 2. 默认 retrieval semantics owner 已挂到 facade 后，优先走它。
        if self._retrieval_semantics is not None:
            result = await self._retrieval_semantics(request)
            return await self._reinforce_context_result(result)

        # 3. 没切主链前，提供一个极薄 fallback，保证 facade 可单独 contract test。
        if self._engine is None:
            result = ContextRetrievalResult(
                trace={"source": "default_runtime_facade", "mode": "disabled"},
                scope_mode="disabled",
            )
            return await self._reinforce_context_result(result)

        # 4. fallback 只做单次 engine.retrieve，不假装拥有旧 pipeline 全部语义。
        engine_result = await self._engine.retrieve(
            MemoryEngineRetrieveRequest(
                query=request.message,
                scope=MemoryScope(
                    session_key=request.session_key,
                    channel=request.channel,
                    chat_id=request.chat_id,
                ),
                context={
                    "history": request.history,
                    "session_metadata": request.session_metadata,
                },
                hints=dict(request.extra),
            )
        )
        injected_ids = [hit.id for hit in engine_result.hits if hit.injected]
        result = ContextRetrievalResult(
            episodic_hits=[_memory_hit_to_item(hit) for hit in engine_result.hits],
            injected_item_ids=injected_ids,
            text_block=engine_result.text_block,
            trace={
                "source": "default_runtime_facade",
                "mode": "engine_fallback",
                **dict(engine_result.trace),
            },
            scope_mode="engine_fallback",
            raw=dict(engine_result.raw),
        )
        return await self._reinforce_context_result(result)

    async def run_consolidation(
        self,
        session: object,
        *,
        archive_all: bool = False,
    ) -> None:
        # 1. consolidation 在 phase 1 还不迁 owner，只提供统一入口。
        if self._consolidation_runner is None:
            raise RuntimeError("consolidation_runner unavailable")
        await self._consolidation_runner(session, archive_all)

    async def retrieve_interest_block(
        self, request: InterestRetrievalRequest
    ) -> InterestRetrievalResult:
        # 1. proactive 未来会切到 facade，这里先保留旧的 preference/profile recall 语义。
        if self._interest_retriever is not None:
            return await self._interest_retriever(request)

        # 2. 默认 fallback 直接转调 port.retrieve_related。
        hits = self._port.retrieve_related(
            request.query,
            memory_types=["preference", "profile"],
            top_k=request.top_k,
            scope_channel=request.scope.channel or None,
            scope_chat_id=request.scope.chat_id or None,
            require_scope_match=bool(request.scope.channel and request.scope.chat_id),
        )
        hits = await _await_if_needed(hits) or []
        texts = [str(hit.get("text", "") or "") for hit in hits if hit.get("text")]
        return InterestRetrievalResult(
            text_block="\n---\n".join(texts),
            hits=list(hits),
            trace={"source": "default_runtime_facade", "mode": "port_fallback"},
            raw={"hits": list(hits)},
        )

    async def remember_explicit(self, request: RememberRequest) -> RememberResult:
        # 1. 显式记忆仍然以 engine 为 owner，facade 只做统一入口。
        if self._engine is None:
            raise RuntimeError("memory engine unavailable")
        return await self._engine.remember(request)

    def read_long_term_context(self) -> str:
        # 1. 文件侧长期上下文读取先继续走 profile_maint/port。
        return str(self._profile_maint.read_long_term() or "")

    def read_self(self) -> str:
        # 1. proactive prompt 仍需要自我认知块，这里保持 file-side 读取入口。
        return str(self._profile_maint.read_self() or "")

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        # 1. consolidation 侧仍依赖 history 文件，这里只是收一个稳定入口。
        return str(self._profile_maint.read_history(max_chars=max_chars) or "")

    def read_recent_context(self) -> str:
        if hasattr(self._profile_maint, "read_recent_context"):
            return str(self._profile_maint.read_recent_context() or "")
        return ""


class DefaultRetrievalSemantics:
    def __init__(
        self,
        *,
        memory: "MemoryServices",
        config: "MemoryConfig",
        llm: "LLMServices",
        workspace: Path,
        light_model: str,
    ) -> None:
        self._gate_resolver = _GateResolver(
            memory=memory,
            config=config,
            llm=llm,
            light_model=light_model,
        )
        self._episodic_retriever = _EpisodicRetriever(memory=memory, config=config)
        self._finalizer = _MemoryRetrievalFinalizer(
            memory=memory,
            config=config,
            workspace=workspace,
        )

    async def retrieve_context(
        self,
        request: ContextRetrievalRequest,
    ) -> ContextRetrievalResult:
        retrieved_block = ""
        rag_trace: RagTrace | None = None
        gate_type = "history_route"
        sufficiency_trace: dict[str, object] = _empty_sufficiency_state()
        p_items: list[dict] = []
        h_items: list[dict] = []
        h_scope_mode = "disabled"
        hyde_hypothesis: str | None = None
        injected_item_ids: list[str] = []
        route_decision = "NO_RETRIEVE"
        rewritten_query = request.message
        try:
            # 1. 先把近窗对话转成 gate / HyDE 需要的轻量上下文。
            main_history = _to_history_dicts(
                request.history[-self._gate_resolver.memory_window :]
            )
            recent_turns = _format_gate_history(main_history, max_turns=3)
            hyde_context = _build_hyde_context(main_history)

            # 2. 再跑默认 retrieval semantics：
            #    gate resolve + procedure/preference recall。
            gate_result, p_items, p_result = await self._gate_resolver.resolve(
                message=request.message,
                session_metadata=request.session_metadata,
                recent_turns=recent_turns,
            )
            resolved_gate = cast(_GateResult, gate_result)
            gate_type = str(resolved_gate["gate_type"])
            rewritten_query = str(resolved_gate["episodic_query"])
            route_decision = str(resolved_gate["route_decision"])
            route_ms = resolved_gate["route_latency_ms"]
            fallback_reason = str(resolved_gate["fallback_reason"])
            history_memory_types = resolved_gate["history_memory_types"]
            gate_latency_ms = {"route": route_ms}

            # 3. gate 放行后，再补 event/profile + HyDE + sufficiency。
            h_items, h_scope_mode, hyde_hypothesis, selected_items, retrieved_block, injected_item_ids = (
                await self._episodic_retriever.retrieve(
                    message=request.message,
                    session_key=request.session_key,
                    channel=request.channel,
                    chat_id=request.chat_id,
                    recent_turns=recent_turns,
                    route_decision=route_decision,
                    rewritten_query=rewritten_query,
                    history_memory_types=history_memory_types,
                    procedure_items=p_items,
                    procedure_result=p_result,
                    hyde_context=hyde_context,
                    sufficiency_trace=sufficiency_trace,
                )
            )

            # 4. 最后统一组 trace / injected block，交给上层继续编排。
            rag_trace = self._finalizer.finalize(
                session_key=request.session_key,
                message=request.message,
                channel=request.channel,
                chat_id=request.chat_id,
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
            rag_trace = self._finalizer.trace_exception(
                session_key=request.session_key,
                message=request.message,
                channel=request.channel,
                chat_id=request.chat_id,
                gate_type=gate_type,
                sufficiency_trace=sufficiency_trace,
                error=e,
            )
        return ContextRetrievalResult(
            normative_hits=list(p_items),
            episodic_hits=list(h_items),
            injected_item_ids=list(injected_item_ids),
            text_block=retrieved_block,
            trace={
                "source": "default_runtime_facade",
                "mode": "default_semantics",
                "gate_type": gate_type,
                "route_decision": route_decision,
            },
            hyde_hypothesis=hyde_hypothesis,
            scope_mode=h_scope_mode,
            sufficiency_trace=dict(sufficiency_trace),
            raw={"rag_trace": rag_trace, "rewritten_query": rewritten_query},
        )


class _GateResolver:
    def __init__(
        self,
        *,
        memory: "MemoryServices",
        config: "MemoryConfig",
        llm: "LLMServices",
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
    ) -> tuple[_GateResult, list[dict], MemoryEngineRetrieveResult | None]:
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
        memory: "MemoryServices",
        config: "MemoryConfig",
    ) -> None:
        self._memory = memory
        self._config = config

    async def retrieve(
        self,
        *,
        message: str,
        session_key: str,
        channel: str,
        chat_id: str,
        recent_turns: str,
        route_decision: str,
        rewritten_query: str,
        history_memory_types: list[str],
        procedure_items: list[dict],
        procedure_result: MemoryEngineRetrieveResult | None,
        hyde_context: str,
        sufficiency_trace: dict[str, object],
    ) -> tuple[list[dict], str, str | None, list[dict], str, list[str]]:
        history_items, history_scope_mode, hyde_hypothesis, history_result = await _retrieve_episodic_items(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            route_decision=route_decision,
            rewritten_query=rewritten_query,
            history_memory_types=history_memory_types,
            hyde_context=hyde_context,
            memory=self._memory,
            config=self._config,
        )
        selected_items, retrieved_block, injected_item_ids = _build_injection_payload(
            procedure_items=procedure_items,
            procedure_result=procedure_result,
            history_items=history_items,
            history_result=history_result,
        )
        history_items, history_scope_mode, selected_items, retrieved_block, injected_item_ids = (
            await _retry_empty_episodic_block(
                message=message,
                recent_turns=recent_turns,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                history_memory_types=history_memory_types,
                procedure_items=procedure_items,
                procedure_result=procedure_result,
                history_items=history_items,
                history_scope_mode=history_scope_mode,
                selected_items=selected_items,
                retrieved_block=retrieved_block,
                injected_item_ids=injected_item_ids,
                sufficiency_trace=sufficiency_trace,
                memory=self._memory,
                config=self._config,
                history_result=history_result,
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
        memory: "MemoryServices",
        config: "MemoryConfig",
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
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    now = datetime.now()
    date_str = now.strftime(f"%Y-%m-%d {weekday_cn[now.weekday()]} %H:%M")
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
    memory: "MemoryServices",
    config: "MemoryConfig",
    llm: "LLMServices",
    light_model: str,
) -> tuple[_GateResult, list[dict], MemoryEngineRetrieveResult | None]:
    if memory.query_rewriter is not None:
        decision_task = asyncio.create_task(
            memory.query_rewriter.decide(
                user_msg=message,
                recent_history=recent_turns,
            )
        )
        procedure_task = asyncio.create_task(
            _retrieve_engine_items(
                memory=memory,
                query=message,
                scope=MemoryScope(),
                mode="procedure",
                memory_types=["procedure", "preference"],
                top_k=config.top_k_procedure,
            )
        )
        decision, p_result = await asyncio.gather(decision_task, procedure_task)
        return {
            "gate_type": "query_rewriter",
            "episodic_query": decision.episodic_query,
            "route_decision": "RETRIEVE" if decision.needs_episodic else "NO_RETRIEVE",
            "route_latency_ms": decision.latency_ms,
            "fallback_reason": "",
            "history_memory_types": ["event", "profile"],
        }, _map_engine_result_to_history_items(p_result), p_result
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
    memory: "MemoryServices",
    config: "MemoryConfig",
    llm: "LLMServices",
    light_model: str,
) -> tuple[_GateResult, list[dict], MemoryEngineRetrieveResult | None]:
    p_task = asyncio.create_task(
        _retrieve_engine_items(
            memory=memory,
            query=message,
            scope=MemoryScope(),
            mode="procedure",
            memory_types=["procedure", "preference"],
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
    p_result, route_decision_obj = await asyncio.gather(p_task, route_task)
    route_reason = _trace_route_reason(route_decision_obj)
    return {
        "gate_type": "history_route",
        "episodic_query": route_decision_obj.rewritten_query,
        "route_decision": "RETRIEVE" if route_decision_obj.needs_history else "NO_RETRIEVE",
        "route_latency_ms": route_decision_obj.latency_ms,
        "fallback_reason": "" if route_reason == "ok" else route_reason,
        "history_memory_types": ["event", "profile"],
    }, _map_engine_result_to_history_items(p_result), p_result


async def _retrieve_episodic_items(
    *,
    session_key: str,
    channel: str,
    chat_id: str,
    route_decision: str,
    rewritten_query: str,
    history_memory_types: list[str],
    hyde_context: str,
    memory: "MemoryServices",
    config: "MemoryConfig",
) -> tuple[list[dict], str, str | None, MemoryEngineRetrieveResult | None]:
    if route_decision != "RETRIEVE" or not history_memory_types:
        return [], "disabled", None, None
    if memory.engine is None:
        return [], "disabled", None, None
    if memory.hyde_enhancer is None:
        engine_result = await _retrieve_engine_items(
            memory=memory,
            query=rewritten_query,
            scope=MemoryScope(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
            ),
            mode="episodic",
            memory_types=history_memory_types,
            top_k=config.top_k_history,
            recent_turns=hyde_context,
            require_scope_match=True,
        )
        return (
            _map_engine_result_to_history_items(engine_result),
            "global",
            None,
            engine_result,
        )

    engine_result, hyde_hypothesis = await _retrieve_episodic_with_hyde(
        memory=memory,
        query=rewritten_query,
        scope=MemoryScope(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
        ),
        memory_types=history_memory_types,
        top_k=config.top_k_history,
        hyde_context=hyde_context,
    )
    scope_mode = "global+hyde" if hyde_hypothesis else "global"
    return _map_engine_result_to_history_items(engine_result), scope_mode, hyde_hypothesis, engine_result


def _map_engine_result_to_history_items(
    engine_result: MemoryEngineRetrieveResult,
) -> list[dict]:
    if engine_result.hits:
        history_items: list[dict] = []
        for hit in engine_result.hits:
            metadata = dict(hit.metadata) if isinstance(hit.metadata, dict) else {}
            memory_type = str(metadata.pop("memory_type", "") or "")
            retrieval_path = str(metadata.pop("_retrieval_path", "history_raw") or "history_raw")
            history_items.append(
                {
                    "id": hit.id,
                    "memory_type": memory_type,
                    "summary": hit.summary,
                    "score": hit.score,
                    "source_ref": hit.source_ref,
                    "extra_json": metadata,
                    "_retrieval_path": retrieval_path,
                }
            )
        return history_items
    raw = cast(_RawItemsPayload, engine_result.raw)
    return [
        dict(
            item,
            _retrieval_path=str(item.get("_retrieval_path", "history_raw") or "history_raw"),
        )
        for item in raw.get("items", [])
        if isinstance(item, dict)
    ]


def _build_injection_payload(
    *,
    procedure_items: list[dict],
    procedure_result: MemoryEngineRetrieveResult | None,
    history_items: list[dict],
    history_result: MemoryEngineRetrieveResult | None,
) -> tuple[list[dict], str, list[str]]:
    procedure_ids = _engine_injected_ids(procedure_result)
    history_ids = _engine_injected_ids(history_result)
    selected_procedure = _filter_injected_items(procedure_items, procedure_ids)
    selected_history = _filter_injected_items(history_items, history_ids)
    selected_items = _merge_memory_items(selected_procedure + selected_history)
    block = "\n\n".join(
        block
        for block in [
            procedure_result.text_block if procedure_result is not None else "",
            history_result.text_block if history_result is not None else "",
        ]
        if block
    )
    return selected_items, block, _dedupe_ids(procedure_ids + history_ids)


async def _retry_empty_episodic_block(
    *,
    message: str,
    recent_turns: str,
    session_key: str,
    channel: str,
    chat_id: str,
    route_decision: str,
    rewritten_query: str,
    history_memory_types: list[str],
    procedure_items: list[dict],
    procedure_result: MemoryEngineRetrieveResult | None,
    history_items: list[dict],
    history_scope_mode: str,
    selected_items: list[dict],
    retrieved_block: str,
    injected_item_ids: list[str],
    sufficiency_trace: dict[str, object],
    memory: "MemoryServices",
    config: "MemoryConfig",
    history_result: MemoryEngineRetrieveResult | None,
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
        return (
            history_items,
            history_scope_mode,
            selected_items,
            retrieved_block,
            injected_item_ids,
        )

    extra_h_items, extra_scope_mode, _retry_hypothesis, extra_history_result = await _retrieve_episodic_items(
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
        route_decision="RETRIEVE",
        rewritten_query=result.refined_query,
        history_memory_types=history_memory_types,
        hyde_context=recent_turns,
        memory=memory,
        config=config,
    )
    sufficiency_trace["retry_count"] = 1
    history_items = history_items + extra_h_items
    history_scope_mode = extra_scope_mode or history_scope_mode
    history_result = cast(
        MemoryEngineRetrieveResult | None,
        extra_history_result if extra_history_result is not None else history_result,
    )
    selected_items, retrieved_block, injected_item_ids = _build_injection_payload(
        procedure_items=procedure_items,
        procedure_result=procedure_result,
        history_items=history_items,
        history_result=history_result,
    )
    return history_items, history_scope_mode, selected_items, retrieved_block, injected_item_ids


async def _retrieve_engine_items(
    *,
    memory: "MemoryServices",
    query: str,
    scope: MemoryScope,
    mode: str,
    memory_types: list[str],
    top_k: int,
    recent_turns: str = "",
    require_scope_match: bool = False,
) -> MemoryEngineRetrieveResult:
    if memory.engine is None:
        return MemoryEngineRetrieveResult(text_block="", hits=[], trace={}, raw={"items": []})
    return await memory.engine.retrieve(
        MemoryEngineRetrieveRequest(
            query=query,
            context={"recent_turns": recent_turns},
            scope=scope,
            mode=mode,
            hints={
                "memory_types": memory_types,
                "require_scope_match": require_scope_match,
            },
            top_k=top_k,
        )
    )


async def _retrieve_episodic_with_hyde(
    *,
    memory: "MemoryServices",
    query: str,
    scope: MemoryScope,
    memory_types: list[str],
    top_k: int,
    hyde_context: str,
) -> tuple[MemoryEngineRetrieveResult, str | None]:
    if memory.hyde_enhancer is None:
        return await _retrieve_engine_items(
            memory=memory,
            query=query,
            scope=scope,
            mode="episodic",
            memory_types=memory_types,
            top_k=top_k,
            recent_turns=hyde_context,
            require_scope_match=True,
        ), None

    raw_task = asyncio.create_task(
        _retrieve_engine_items(
            memory=memory,
            query=query,
            scope=scope,
            mode="episodic",
            memory_types=memory_types,
            top_k=top_k,
            recent_turns=hyde_context,
            require_scope_match=True,
        )
    )
    hyp_task = asyncio.create_task(
        memory.hyde_enhancer.generate_hypothesis(query, hyde_context)
    )
    raw_result, hypothesis = await asyncio.gather(raw_task, hyp_task)
    if not hypothesis:
        return raw_result, None

    raw_result = _annotate_engine_result_path(raw_result, "history_raw")
    hyde_result = await _retrieve_engine_items(
        memory=memory,
        query=hypothesis,
        scope=scope,
        mode="episodic",
        memory_types=memory_types,
        top_k=top_k,
        recent_turns=hyde_context,
        require_scope_match=True,
    )
    hyde_result = _annotate_engine_result_path(hyde_result, "history_hyde")
    merged_items = _max_pool_history_items(
        _engine_raw_items(raw_result) + _engine_raw_items(hyde_result)
    )
    merged_hits = _merge_engine_hits(raw_result.hits + hyde_result.hits)
    blocks = [raw_result.text_block]
    if hyde_result.text_block and not set(_engine_injected_ids(hyde_result)).issubset(
        set(_engine_injected_ids(raw_result))
    ):
        blocks.append(hyde_result.text_block)
    return (
        MemoryEngineRetrieveResult(
            text_block="\n\n".join(block for block in blocks if block),
            hits=merged_hits,
            trace=dict(raw_result.trace),
            raw={"items": merged_items},
        ),
        hypothesis,
    )


def _engine_raw_items(result: MemoryEngineRetrieveResult | None) -> list[dict]:
    if result is None:
        return []
    raw = cast(_RawItemsPayload, result.raw)
    return [
        dict(item)
        for item in raw.get("items", [])
        if isinstance(item, dict)
    ]


def _annotate_engine_result_path(
    result: MemoryEngineRetrieveResult,
    path: str,
) -> MemoryEngineRetrieveResult:
    for hit in result.hits:
        metadata = dict(hit.metadata) if isinstance(hit.metadata, dict) else {}
        metadata["_retrieval_path"] = path
        hit.metadata = metadata
    raw = cast(_RawItemsPayload, result.raw)
    for item in raw.get("items", []):
        if isinstance(item, dict):
            item["_retrieval_path"] = path
    return result


def _engine_injected_ids(result: MemoryEngineRetrieveResult | None) -> list[str]:
    if result is None:
        return []
    return [hit.id for hit in result.hits if hit.injected and hit.id]


def _filter_injected_items(items: list[dict], injected_ids: list[str]) -> list[dict]:
    injected_id_set = set(injected_ids)
    return [
        item
        for item in items
        if str(item.get("id", "")) in injected_id_set
    ]


def _merge_engine_hits(hits: list[MemoryHit]) -> list[MemoryHit]:
    by_id: dict[str, MemoryHit] = {}
    extras: list[MemoryHit] = []
    for hit in hits:
        item_id = str(getattr(hit, "id", "") or "")
        if not item_id:
            extras.append(hit)
            continue
        existing = by_id.get(item_id)
        if existing is None:
            by_id[item_id] = hit
            continue
        injected = bool(getattr(existing, "injected", False)) or bool(
            getattr(hit, "injected", False)
        )
        existing_score = float(getattr(existing, "score", 0.0) or 0.0)
        hit_score = float(getattr(hit, "score", 0.0) or 0.0)
        if hit_score > existing_score:
            hit.injected = injected
            by_id[item_id] = hit
            continue
        existing.injected = injected
    return list(by_id.values()) + extras


def _max_pool_history_items(items: list[dict]) -> list[dict]:
    pooled: dict[str, dict] = {}
    extras: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "") or "")
        if not item_id:
            extras.append(item)
            continue
        current = pooled.get(item_id)
        if current is None or float(item.get("score", 0.0) or 0.0) > float(
            current.get("score", 0.0) or 0.0
        ):
            pooled[item_id] = item
    merged = list(pooled.values()) + extras
    merged.sort(
        key=lambda item: (float(item.get("score", 0.0) or 0.0), str(item.get("id", ""))),
        reverse=True,
    )
    return merged


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


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item_id in ids:
        value = str(item_id or "")
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


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
    config: "MemoryConfig",
) -> "RagTrace":
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
    procedure_guard_applied = _has_procedure_guard_hit(
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
        procedure_guard_applied=procedure_guard_applied,
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


def _has_procedure_guard_hit(
    *,
    procedure_items: list[dict],
    injected_item_ids: list[str],
    config: "MemoryConfig",
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
        config.procedure_guard_enabled
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
) -> "RagTrace":
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


def _tool_group_to_dict(group: "ToolCallGroup") -> dict[str, Any]:
    return {
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


def _memory_hit_to_item(hit) -> dict[str, Any]:
    metadata = dict(getattr(hit, "metadata", {}) or {})
    memory_type = str(metadata.get("memory_type", "") or "")
    item = {
        "id": str(getattr(hit, "id", "") or ""),
        "summary": str(getattr(hit, "summary", "") or ""),
        "text": str(getattr(hit, "content", "") or ""),
        "score": float(getattr(hit, "score", 0.0) or 0.0),
        "source_ref": str(getattr(hit, "source_ref", "") or ""),
        "memory_type": memory_type,
        "extra_json": metadata,
        "injected": bool(getattr(hit, "injected", False)),
    }
    return item


async def _await_if_needed(value):
    if inspect.isawaitable(value):
        return await value
    return value
