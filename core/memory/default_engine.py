from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast
from typing import Any

from agent.config_models import Config
from agent.memory import MemoryStore
from agent.provider import LLMProvider, LLMResponse
from agent.skills import SkillsLoader
from core.memory.engine import (
    ConsolidateRequest,
    ConsolidateResult,
    EngineProfile,
    ExplicitRetrievalRequest,
    ExplicitRetrievalResult,
    ForgetRequest,
    ForgetResult,
    InterestRetrievalRequest,
    InterestRetrievalResult,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryHit,
    MemoryIngestRequest,
    MemoryIngestResult,
    RefreshRecentTurnsRequest,
    RememberRequest,
    RememberResult,
)
from core.net.http import SharedHttpResources
from memory2.embedder import Embedder
from memory2.memorizer import Memorizer
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.procedure_tagger import ProcedureTagger
from memory2.query_builder import build_procedure_queries
from memory2.retriever import Retriever
from memory2.rule_schema import build_procedure_rule_schema
from memory2.store import MemoryStore2

if TYPE_CHECKING:
    from bus.publisher import EventPublisher

logger = logging.getLogger("memory.default_engine")

_HYPOTHESIS_MAX_TOKENS = 80
_HYPOTHESIS_TIMEOUT_S = 3.0
_VECTOR_SCORE_THRESHOLD = 0.35
_VECTOR_TOP_K = 15
_ChatCall = Callable[..., Awaitable[LLMResponse]]


class DefaultMemoryEngine:
    DESCRIPTOR = MemoryEngineDescriptor(
        name="default",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_MESSAGES,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.MANAGE_HISTORY,
                MemoryCapability.MANAGE_UPDATE,
                MemoryCapability.MANAGE_DELETE,
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={"owner": "core.memory.default_engine"},
    )

    def __init__(
        self,
        *,
        config: Config | None = None,
        workspace: Path | None = None,
        provider: LLMProvider | None = None,
        light_provider: LLMProvider | None = None,
        http_resources: SharedHttpResources | None = None,
        event_publisher: "EventPublisher | None" = None,
        retriever: Any | None = None,
        memorizer: Any | None = None,
        tagger: Any | None = None,
        post_response_worker: Any | None = None,
    ) -> None:
        if retriever is not None or config is None:
            # TODO(memory-engine-cleanup): 旧单元测试和外部构造器迁移到 config 构造后删除注入兼容壳。
            self._config = config
            self._workspace = workspace or Path(".")
            self._provider = provider
            self._light_provider = light_provider or provider
            self._light_model = ""
            self._v1_store = cast(MemoryStore, None)
            self._v2_store = None
            self._embedder = None
            self._memorizer = cast(Memorizer | None, memorizer)
            self._retriever = cast(Retriever | None, retriever)
            self._tagger = cast(ProcedureTagger | None, tagger)
            self._post_response_worker = cast(
                PostResponseMemoryWorker | None,
                post_response_worker,
            )
            self._consolidation = None
            self.closeables = []
            return

        assert workspace is not None
        assert provider is not None
        assert http_resources is not None
        self._config = config
        self._workspace = workspace
        self._provider = provider
        self._light_provider = light_provider or provider
        self._light_model = config.light_model or config.model
        self._v1_store = MemoryStore(workspace)
        self._v2_store: MemoryStore2 | None = None
        self._embedder: Embedder | None = None
        self._memorizer: Memorizer | None = None
        self._retriever: Retriever | None = None
        self._tagger: ProcedureTagger | None = None
        self._post_response_worker: PostResponseMemoryWorker | None = None
        self._consolidation = None
        self.closeables: list[object] = []

        if not config.memory_v2.enabled:
            return

        db_path = (
            Path(config.memory_v2.db_path)
            if config.memory_v2.db_path
            else workspace / "memory" / "memory2.db"
        )
        self._v2_store = MemoryStore2(db_path)
        self._embedder = Embedder(
            base_url=config.memory_v2.base_url
            or config.light_base_url
            or config.base_url
            or "",
            api_key=config.memory_v2.api_key
            or config.light_api_key
            or config.api_key,
            model=config.memory_v2.embed_model,
            requester=http_resources.external_default,
        )
        self._memorizer = Memorizer(self._v2_store, self._embedder)
        self._retriever = Retriever(
            self._v2_store,
            self._embedder,
            top_k=config.memory_v2.retrieve_top_k,
            score_threshold=config.memory_v2.score_threshold,
            score_thresholds={
                "procedure": config.memory_v2.score_threshold_procedure,
                "preference": config.memory_v2.score_threshold_preference,
                "event": config.memory_v2.score_threshold_event,
                "profile": config.memory_v2.score_threshold_profile,
            },
            relative_delta=config.memory_v2.relative_delta,
            inject_max_chars=config.memory_v2.inject_max_chars,
            inject_max_forced=config.memory_v2.inject_max_forced,
            inject_max_procedure_preference=config.memory_v2.inject_max_procedure_preference,
            inject_max_event_profile=config.memory_v2.inject_max_event_profile,
            inject_line_max=config.memory_v2.inject_line_max,
            procedure_guard_enabled=config.memory_v2.procedure_guard_enabled,
            hotness_alpha=0.20,
        )
        skills_loader = SkillsLoader(workspace)
        self._tagger = ProcedureTagger(
            provider=self._light_provider,
            model=self._light_model,
            skills_fn=lambda: [
                s["name"] for s in skills_loader.list_skills(filter_unavailable=False)
            ],
        )
        self._post_response_worker = PostResponseMemoryWorker(
            memorizer=self._memorizer,
            retriever=self._retriever,
            light_provider=self._light_provider,
            light_model=self._light_model,
            event_publisher=event_publisher,
        )
        self.closeables = [self._v2_store, self._embedder]

        from agent.looping.consolidation import ConsolidationService
        from memory2.profile_extractor import ProfileFactExtractor

        # TODO(memory-engine-cleanup): 旧归档服务仍是兼容壳，后续把实现搬入 engine 后删除。
        self._consolidation = ConsolidationService(
            memory=self,
            profile_maint=self,
            provider=provider,
            model=config.model,
            keep_count=_keep_count(config.memory_window),
            profile_extractor=ProfileFactExtractor(
                llm_client=self._light_provider,
                model=self._light_model,
            ),
            recent_context_provider=self._light_provider,
            recent_context_model=self._light_model,
        )

    @classmethod
    def ensure_workspace_storage(cls, *, config: Config, workspace: Path) -> None:
        if not config.memory_v2.enabled:
            return
        db_path = (
            Path(config.memory_v2.db_path)
            if config.memory_v2.db_path
            else workspace / "memory" / "memory2.db"
        )
        store = MemoryStore2(db_path)
        store.close()

    @classmethod
    def for_dashboard_workspace(cls, workspace: Path) -> "DefaultMemoryEngine":
        engine = cls.__new__(cls)
        engine._config = None
        engine._workspace = workspace
        engine._provider = None
        engine._light_provider = None
        engine._light_model = ""
        engine._v1_store = MemoryStore(workspace)
        engine._v2_store = MemoryStore2(workspace / "memory" / "memory2.db")
        engine._embedder = None
        engine._memorizer = None
        engine._retriever = None
        engine._tagger = None
        engine._post_response_worker = None
        engine._consolidation = None
        engine.closeables = [engine._v2_store]
        return engine

    async def retrieve(
        self,
        request: MemoryEngineRetrieveRequest,
    ) -> MemoryEngineRetrieveResult:
        if self._retriever is None:
            return MemoryEngineRetrieveResult(text_block="", hits=[], raw={"items": []})

        scope = self._resolve_scope(request.scope)
        queries = self._resolve_queries(request)
        memory_types = self._resolve_memory_types(request)
        items = await self._retrieve_related(
            request.query,
            memory_types=memory_types,
            top_k=request.top_k,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=bool(request.hints.get("require_scope_match", False)),
            aux_queries=queries[1:],
        )
        text_block, injected_ids = self._retriever.build_injection_block(items)
        hits = [
            self._build_hit(item, injected_ids=injected_ids)
            for item in items
            if isinstance(item, dict)
        ]
        return MemoryEngineRetrieveResult(
            text_block=text_block,
            hits=hits,
            trace={
                "engine": self.DESCRIPTOR.name,
                "profile": self.DESCRIPTOR.profile.value,
                "mode": request.mode,
            },
            raw={"items": items},
        )

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        scope = self._resolve_scope(request.scope)
        if self._post_response_worker is None:
            return MemoryIngestResult(
                accepted=False,
                summary="post_response_worker unavailable",
                raw={"reason": "worker_unavailable"},
            )
        if request.source_kind not in {"conversation_turn", "conversation_batch"}:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported source_kind",
                raw={"reason": "unsupported_source_kind"},
            )
        normalized = self._normalize_ingest_content(request.content)
        if normalized is None:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported content for conversation ingest",
                raw={"reason": "invalid_content"},
            )

        await self._post_response_worker.run(
            user_msg=normalized["user_message"],
            agent_response=normalized["assistant_response"],
            tool_chain=normalized["tool_chain"],
            source_ref=str(
                request.metadata.get("source_ref")
                or normalized["source_ref"]
                or f"{scope.session_key}@post_response"
            ),
            session_key=scope.session_key,
            channel=scope.channel,
            chat_id=scope.chat_id,
        )
        return MemoryIngestResult(
            accepted=True,
            summary="delegated to post_response_worker",
            raw={"engine": self.DESCRIPTOR.name},
        )

    async def remember(self, request: RememberRequest) -> RememberResult:
        if self._memorizer is None:
            raise RuntimeError("memorizer unavailable")

        raw_steps = request.raw_extra.get("steps")
        steps = [str(step) for step in raw_steps] if isinstance(raw_steps, list) else None
        memory_type = _coerce_memory_type(
            request.memory_type,
            str(request.raw_extra.get("tool_requirement") or ""),
            steps,
        )
        extra = {
            "tool_requirement": request.raw_extra.get("tool_requirement"),
            "steps": list(steps or []),
        }
        if memory_type == "procedure":
            extra["rule_schema"] = build_procedure_rule_schema(
                summary=request.summary,
                tool_requirement=str(request.raw_extra.get("tool_requirement") or "") or None,
                steps=list(steps or []),
            )
            await self._attach_trigger_tags(extra=extra, summary=request.summary)

        result = await self._memorizer.save_item_with_supersede(
            summary=request.summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=request.source_ref,
        )
        write_status, actual_id = _split_write_result(result)
        return RememberResult(
            item_id=actual_id,
            actual_type=memory_type,
            write_status=write_status,
            superseded_ids=[],
        )

    async def forget(self, request: ForgetRequest) -> ForgetResult:
        store = self._require_v2_store()
        clean_ids = _dedupe_ids(request.ids)
        items = store.get_items_by_ids(clean_ids)
        found_ids = [str(item.get("id") or "") for item in items if item.get("id")]
        if found_ids:
            store.mark_superseded_batch(found_ids)
        return ForgetResult(
            superseded_ids=found_ids,
            missing_ids=[item_id for item_id in clean_ids if item_id not in set(found_ids)],
            items=[
                {
                    "id": item.get("id"),
                    "memory_type": item.get("memory_type"),
                    "summary": item.get("summary"),
                }
                for item in items
            ],
        )

    async def consolidate(self, request: ConsolidateRequest) -> ConsolidateResult:
        if self._consolidation is None:
            return ConsolidateResult(trace={"mode": "disabled"})
        await self._consolidation.consolidate(
            request.session,
            archive_all=request.archive_all,
            force=request.force,
        )
        return ConsolidateResult(trace={"mode": "legacy_service"})

    async def refresh_recent_turns(
        self,
        request: RefreshRecentTurnsRequest,
    ) -> None:
        if self._consolidation is None:
            return
        await self._consolidation.refresh_recent_turns(session=request.session)

    async def retrieve_explicit(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        if request.search_mode == "grep":
            return self._retrieve_explicit_grep(request)
        return await self._retrieve_explicit_semantic(request)

    async def retrieve_interest_block(
        self,
        request: InterestRetrievalRequest,
    ) -> InterestRetrievalResult:
        scope = self._resolve_scope(request.scope)
        hits = await self._retrieve_related(
            request.query,
            memory_types=["preference", "profile"],
            top_k=request.top_k,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=bool(scope.channel and scope.chat_id),
        )
        texts = [str(hit.get("text", "") or "") for hit in hits if hit.get("text")]
        return InterestRetrievalResult(
            text_block="\n---\n".join(texts),
            hits=list(hits),
            trace={"source": self.DESCRIPTOR.name, "mode": "interest"},
            raw={"hits": list(hits)},
        )

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def read_long_term(self) -> str:
        return self._v1_store.read_long_term()

    def write_long_term(self, content: str) -> None:
        self._v1_store.write_long_term(content)

    def read_self(self) -> str:
        return self._v1_store.read_self()

    def write_self(self, content: str) -> None:
        self._v1_store.write_self(content)

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        return self._v1_store.read_history(max_chars=max_chars)

    def read_history(self, max_chars: int = 0) -> str:
        return self._v1_store.read_history(max_chars=max_chars)

    def read_recent_context(self) -> str:
        return self._v1_store.read_recent_context()

    def write_recent_context(self, content: str) -> None:
        self._v1_store.write_recent_context(content)

    def backup_long_term(self, backup_name: str = "MEMORY.bak.md") -> None:
        if self._v1_store.memory_file.exists():
            shutil.copyfile(
                self._v1_store.memory_file,
                self._v1_store.memory_file.with_name(backup_name),
            )

    def get_memory_context(self) -> str:
        return self._v1_store.get_memory_context()

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())

    def read_pending(self) -> str:
        return self._v1_store.read_pending()

    def append_pending(self, facts: str) -> None:
        self._v1_store.append_pending(facts)

    def append_pending_once(
        self,
        facts: str,
        source_ref: str,
        kind: str = "pending",
    ) -> bool:
        return self._v1_store.append_pending_once(
            facts,
            source_ref=source_ref,
            kind=kind,
        )

    def snapshot_pending(self) -> str:
        return self._v1_store.snapshot_pending()

    def commit_pending_snapshot(self) -> None:
        self._v1_store.commit_pending_snapshot()

    def rollback_pending_snapshot(self) -> None:
        self._v1_store.rollback_pending_snapshot()

    def append_history(self, entry: str) -> None:
        self._v1_store.append_history(entry)

    def append_history_once(
        self,
        entry: str,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        return self._v1_store.append_history_once(
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        return self._v1_store.append_journal(
            date_str,
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        if self._memorizer is not None:
            self._memorizer.reinforce_items_batch(ids)

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        store = self._v2_store
        return store.keyword_match_procedures(action_tokens) if store is not None else []

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        store = self._v2_store
        if store is None:
            return []
        return store.list_events_by_time_range(time_start, time_end, limit=limit)

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        store = self._require_v2_store()
        return store.list_items_for_dashboard(
            q=q,
            memory_type=memory_type,
            status=status,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            has_embedding=has_embedding,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        return self._require_v2_store().get_item_for_dashboard(
            item_id,
            include_embedding=include_embedding,
        )

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        return self._require_v2_store().update_item_for_dashboard(
            item_id,
            status=status,
            extra_json=extra_json,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    def delete_item(self, item_id: str) -> bool:
        return self._require_v2_store().delete_item(item_id)

    def delete_items_batch(self, ids: list[str]) -> int:
        return self._require_v2_store().delete_items_batch(ids)

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        return self._require_v2_store().find_similar_items_for_dashboard(
            item_id,
            top_k=top_k,
            memory_type=memory_type,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
        )

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> None:
        if self._memorizer is None:
            return
        await self._memorizer.save_from_consolidation(
            history_entry=history_entry,
            behavior_updates=behavior_updates,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            emotional_weight=emotional_weight,
        )

    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        if self._memorizer is None:
            return ""
        return await self._memorizer.save_item(
            summary=summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def save_item_with_supersede(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        if self._memorizer is None:
            return ""
        return await self._memorizer.save_item_with_supersede(
            summary=summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def _retrieve_explicit_semantic(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        hyp1_task = asyncio.create_task(self._gen_hypothesis(request.query, style="event"))
        hyp2_task = asyncio.create_task(self._gen_hypothesis(request.query, style="general"))
        hyp1, hyp2 = await asyncio.gather(hyp1_task, hyp2_task)
        aux_queries = [text for text in (hyp1, hyp2) if text]
        types = [request.memory_type] if request.memory_type else None
        hits = await self._retrieve_related(
            request.query,
            memory_types=types,
            top_k=max(request.limit, _VECTOR_TOP_K),
            scope_channel=request.scope.channel or None,
            scope_chat_id=request.scope.chat_id or None,
            require_scope_match=bool(request.scope.channel and request.scope.chat_id),
            aux_queries=aux_queries,
            score_threshold=_VECTOR_SCORE_THRESHOLD,
            time_start=request.time_start,
            time_end=request.time_end,
            keyword_enabled=True,
        )
        sliced = list(hits)[: request.limit]
        return ExplicitRetrievalResult(
            hits=sliced,
            trace={
                "source": self.DESCRIPTOR.name,
                "mode": "semantic",
                "hit_count": len(sliced),
                "hyde_hypotheses": aux_queries,
            },
            raw={"hits": sliced},
        )

    def _retrieve_explicit_grep(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        if request.time_start is None or request.time_end is None:
            return ExplicitRetrievalResult(
                trace={"source": self.DESCRIPTOR.name, "mode": "grep_missing_time"}
            )
        hits = self.list_events_by_time_range(
            request.time_start,
            request.time_end,
            limit=request.limit,
        )
        return ExplicitRetrievalResult(
            hits=list(hits),
            trace={"source": self.DESCRIPTOR.name, "mode": "grep", "hit_count": len(hits)},
            raw={"hits": list(hits)},
        )

    async def _retrieve_related(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
        aux_queries: list[str] | None = None,
        score_threshold: float | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        keyword_enabled: bool = True,
    ) -> list[dict]:
        if self._retriever is None:
            return []
        return await self._retriever.retrieve(
            query,
            memory_types=memory_types,
            top_k=top_k,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
            aux_queries=aux_queries,
            score_threshold=score_threshold,
            time_start=time_start,
            time_end=time_end,
            keyword_enabled=keyword_enabled,
        )

    async def _gen_hypothesis(self, query: str, style: str) -> str | None:
        prompt = _explicit_hypothesis_prompt(query, style)
        try:
            chat = cast(_ChatCall, getattr(self._light_provider, "chat"))
            resp = await asyncio.wait_for(
                chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._light_model,
                    max_tokens=_HYPOTHESIS_MAX_TOKENS,
                ),
                timeout=_HYPOTHESIS_TIMEOUT_S,
            )
            text = (resp.content or "").strip()
            return text if text else None
        except Exception as e:
            logger.debug("explicit retrieval hypothesis failed: %s", e)
            return None

    async def _attach_trigger_tags(self, *, extra: dict, summary: str) -> None:
        if self._tagger is None:
            return
        try:
            trigger_tags = await self._tagger.tag(summary)
        except Exception:
            return
        if trigger_tags is not None:
            extra["trigger_tags"] = trigger_tags

    def _require_v2_store(self) -> MemoryStore2:
        if self._v2_store is None:
            raise RuntimeError("memory v2 store unavailable")
        return self._v2_store

    @classmethod
    def _build_hit(
        cls,
        item: dict,
        *,
        injected_ids: list[str] | None = None,
    ) -> MemoryHit:
        extra = item.get("extra_json")
        metadata = dict(extra) if isinstance(extra, dict) else {}
        metadata["memory_type"] = item.get("memory_type", "")
        item_id = str(item.get("id", "") or "")
        return MemoryHit(
            id=item_id,
            summary=str(item.get("summary", "") or ""),
            content=str(item.get("summary", "") or ""),
            score=float(item.get("score", 0.0) or 0.0),
            source_ref=str(item.get("source_ref", "") or ""),
            engine_kind=cls.DESCRIPTOR.name,
            metadata=metadata,
            injected=item_id in set(injected_ids or []),
        )

    @staticmethod
    def _resolve_scope(scope):
        if scope.channel and scope.chat_id:
            return scope
        if not scope.session_key or ":" not in scope.session_key:
            return scope
        channel, chat_id = scope.session_key.split(":", 1)
        return type(scope)(
            session_key=scope.session_key,
            channel=scope.channel or channel,
            chat_id=scope.chat_id or chat_id,
        )

    @staticmethod
    def _normalize_ingest_content(
        content: object,
    ) -> "_NormalizedIngestContent | None":
        if isinstance(content, dict):
            raw_tool_chain = content.get("tool_chain")
            normalized_tool_chain = (
                [item for item in raw_tool_chain if isinstance(item, dict)]
                if isinstance(raw_tool_chain, list)
                else []
            )
            return cast(
                _NormalizedIngestContent,
                {
                    "user_message": str(content.get("user_message", "") or ""),
                    "assistant_response": str(
                        content.get("assistant_response", "") or ""
                    ),
                    "tool_chain": normalized_tool_chain,
                    "source_ref": str(content.get("source_ref", "") or ""),
                },
            )
        if not isinstance(content, list):
            return None

        user_message = ""
        assistant_response = ""
        tool_chain: list[dict] = []
        for message in content:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "")
            body = str(message.get("content", "") or "")
            if role == "user" and body:
                user_message = body
            elif role == "assistant" and body:
                assistant_response = body
                maybe_tool_chain = message.get("tool_chain")
                if isinstance(maybe_tool_chain, list):
                    tool_chain = maybe_tool_chain
        if not user_message and not assistant_response:
            return None
        return cast(
            _NormalizedIngestContent,
            {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "tool_chain": tool_chain,
                "source_ref": "",
            },
        )

    @staticmethod
    def _resolve_memory_types(
        request: MemoryEngineRetrieveRequest,
    ) -> list[str] | None:
        memory_types = request.hints.get("memory_types")
        if isinstance(memory_types, list):
            return [str(item) for item in memory_types if str(item).strip()]
        if request.mode == "procedure":
            return ["procedure", "preference"]
        if request.mode == "episodic":
            return ["event", "profile"]
        return None

    @staticmethod
    def _resolve_queries(request: MemoryEngineRetrieveRequest) -> list[str]:
        raw_queries = request.hints.get("queries")
        if isinstance(raw_queries, list):
            queries = [str(item).strip() for item in raw_queries if str(item).strip()]
            if queries:
                return queries
        if request.mode == "procedure":
            return build_procedure_queries(request.query)
        return [request.query]


class _NormalizedIngestContent(TypedDict):
    user_message: str
    assistant_response: str
    tool_chain: list[dict]
    source_ref: str


def _coerce_memory_type(
    memory_type: str,
    tool_requirement: str | None,
    steps: list[str] | None,
) -> str:
    if memory_type != "procedure":
        return memory_type
    if tool_requirement and tool_requirement.strip():
        return memory_type
    if steps and any(str(step).strip() for step in steps):
        return memory_type
    return "preference"


def _split_write_result(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if ":" not in raw:
        return "new", raw
    status, item_id = raw.split(":", 1)
    return status or "new", item_id


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        item_id = str(raw or "").strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


def _keep_count(window: int) -> int:
    aligned_window = max(6, ((max(1, window) + 5) // 6) * 6)
    return aligned_window // 2


def _explicit_hypothesis_prompt(query: str, style: str) -> str:
    if style == "event":
        return (
            "你是个人助手的记忆系统。根据用户提问，生成一条带具体时间的假想记忆条目，"
            "格式如 '[2026-03-08] 用户...'\n"
            "规则：第三人称、简洁事实陈述、只输出那一条文本\n\n"
            f"用户提问：{query}\n假想记忆条目："
        )
    return (
        "你是个人助手的记忆系统。根据用户提问，生成一条假想记忆条目。\n"
        "规则：始终生成肯定式、第三人称（'用户…'）、简洁事实陈述、只输出那一条文本\n\n"
        f"用户提问：{query}\n假想记忆条目："
    )
