from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypedDict, cast

from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryHit,
    MemoryIngestRequest,
    MemoryIngestResult,
    RememberRequest,
    RememberResult,
)
from memory2.rule_schema import build_procedure_rule_schema

if TYPE_CHECKING:
    from memory2.memorizer import Memorizer
    from memory2.post_response_worker import PostResponseMemoryWorker
    from memory2.procedure_tagger import ProcedureTagger
    from memory2.retriever import Retriever


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
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={"owner": "memory2"},
    )

    def __init__(
        self,
        retriever: "Retriever",
        memorizer: "Memorizer | None" = None,
        tagger: "ProcedureTagger | None" = None,
        post_response_worker: "PostResponseMemoryWorker | None" = None,
    ) -> None:
        self._retriever = retriever
        self._memorizer = memorizer
        self._tagger = tagger
        self._post_response_worker = post_response_worker

    # ┌──────────────────────────────────────────────┐
    # │ DefaultMemoryEngine.retrieve                 │
    # ├──────────────────────────────────────────────┤
    # │ MemoryEngineRetrieveRequest                  │
    # │ -> memory2 Retriever                        │
    # │ -> MemoryHit + text_block                   │
    # └──────────────────────────────────────────────┘
    async def retrieve(
        self, request: MemoryEngineRetrieveRequest
    ) -> MemoryEngineRetrieveResult:
        scope = self._resolve_scope(request.scope)
        items = await self._retrieve_items(request=request, scope=scope)
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

    # ┌──────────────────────────────────────────────┐
    # │ DefaultMemoryEngine.ingest                   │
    # ├──────────────────────────────────────────────┤
    # │ MemoryIngestRequest                          │
    # │ -> PostResponseMemoryWorker.run              │
    # │ -> MemoryIngestResult                        │
    # └──────────────────────────────────────────────┘
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

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    @classmethod
    def _build_hit(cls, item: dict, *, injected_ids: list[str] | None = None) -> MemoryHit:
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
            normalized_tool_chain = [
                item for item in raw_tool_chain if isinstance(item, dict)
            ] if isinstance(raw_tool_chain, list) else []
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

    async def _retrieve_items(self, *, request, scope) -> list[dict]:
        memory_types = request.hints.get("memory_types")
        queries = request.hints.get("queries")
        active_queries = (
            [str(item).strip() for item in queries if str(item).strip()]
            if isinstance(queries, list)
            else []
        )
        if not active_queries:
            active_queries = [request.query]
        all_results = await self._gather_queries(
            queries=active_queries,
            memory_types=list(memory_types) if isinstance(memory_types, list) else None,
            top_k=request.top_k,
            scope=scope,
            require_scope_match=bool(request.hints.get("require_scope_match", False)),
        )
        return _max_pool_memory_items(all_results, top_k=request.top_k)

    async def _gather_queries(
        self,
        *,
        queries: list[str],
        memory_types: list[str] | None,
        top_k: int | None,
        scope,
        require_scope_match: bool,
    ) -> list[list[dict]]:
        tasks = [
            self._retriever.retrieve(
                item_query,
                memory_types=memory_types,
                top_k=top_k,
                scope_channel=scope.channel or None,
                scope_chat_id=scope.chat_id or None,
                require_scope_match=require_scope_match,
            )
            for item_query in queries
        ]
        return await asyncio.gather(*tasks)

    async def _attach_trigger_tags(self, *, extra: dict, summary: str) -> None:
        if self._tagger is None:
            return
        try:
            trigger_tags = await self._tagger.tag(summary)
        except Exception:
            return
        if trigger_tags is not None:
            extra["trigger_tags"] = trigger_tags


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


def _max_pool_memory_items(raw_results: list[list[dict]], top_k: int | None) -> list[dict]:
    pooled: dict[str, dict] = {}
    ordered: list[dict] = []
    for bucket in raw_results:
        for item in bucket:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "") or "")
            if item_id:
                current = pooled.get(item_id)
                if current is None or float(item.get("score", 0.0) or 0.0) > float(
                    current.get("score", 0.0) or 0.0
                ):
                    pooled[item_id] = item
                continue
            ordered.append(item)
    merged = list(pooled.values()) + ordered
    merged.sort(
        key=lambda item: (float(item.get("score", 0.0) or 0.0), str(item.get("id", ""))),
        reverse=True,
    )
    if top_k is None:
        return merged
    return merged[: max(1, int(top_k))]


def _split_write_result(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if ":" not in raw:
        return "new", raw
    status, item_id = raw.split(":", 1)
    return status or "new", item_id
