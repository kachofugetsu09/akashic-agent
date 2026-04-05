from __future__ import annotations

from typing import TYPE_CHECKING

from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryHit,
    MemoryIngestRequest,
    MemoryIngestResult,
)

if TYPE_CHECKING:
    from memory2.post_response_worker import PostResponseMemoryWorker
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
        post_response_worker: "PostResponseMemoryWorker | None" = None,
    ) -> None:
        self._retriever = retriever
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
        # 1. 读取检索请求，映射到当前 memory2 retriever 的最小参数集合。
        scope = self._resolve_scope(request.scope)
        memory_types = request.hints.get("memory_types")
        items = await self._retriever.retrieve(
            request.query,
            memory_types=list(memory_types) if isinstance(memory_types, list) else None,
            top_k=request.top_k,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=bool(request.hints.get("require_scope_match", False)),
        )

        # 2. 调用默认 rich retrieval 组装 block，并把命中项收敛成统一结构。
        text_block, injected_ids = self._retriever.build_injection_block(items)
        hits = [
            self._build_hit(item, injected_ids=injected_ids)
            for item in items
            if isinstance(item, dict)
        ]

        # 3. 汇总 trace/raw，返回给上层内部引擎调用方。
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
        # 1. 读取输入与上下文，完成当前兼容壳支持范围判断。
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

        # 2. 构造当前默认实现需要的参数，并进入现有 post-turn 主流程。
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

        # 3. 返回兼容壳结果；当前 memory2 worker 不直接暴露 created ids。
        return MemoryIngestResult(
            accepted=True,
            summary="delegated to post_response_worker",
            raw={"engine": self.DESCRIPTOR.name},
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
    def _normalize_ingest_content(content: object) -> dict[str, object] | None:
        if isinstance(content, dict):
            return {
                "user_message": str(content.get("user_message", "") or ""),
                "assistant_response": str(content.get("assistant_response", "") or ""),
                "tool_chain": (
                    content.get("tool_chain") if isinstance(content.get("tool_chain"), list) else []
                ),
                "source_ref": str(content.get("source_ref", "") or ""),
            }
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
        return {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "tool_chain": tool_chain,
            "source_ref": "",
        }
