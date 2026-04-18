from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

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


class MemURetrieveService(Protocol):
    async def retrieve(
        self,
        queries: list[dict[str, object]],
        where: dict[str, object] | None = None,
    ) -> dict[str, object]: ...

    async def memorize(
        self,
        *,
        resource_url: str,
        modality: str,
        user: dict[str, object] | None = None,
    ) -> dict[str, object]: ...


class MemUScopeModel(BaseModel):
    session_key: str | None = None
    channel: str | None = None
    chat_id: str | None = None


class MemUMemoryEngine:
    DESCRIPTOR = MemoryEngineDescriptor(
        name="memu",
        profile=EngineProfile.WORKFLOW_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_TEXT,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
            }
        ),
        notes={"owner": "memu", "ingest": "text_only"},
    )

    def __init__(self, service: MemURetrieveService, input_dir: Path) -> None:
        self._service = service
        self._input_dir = input_dir

    async def retrieve(
        self, request: MemoryEngineRetrieveRequest
    ) -> MemoryEngineRetrieveResult:
        response = await self._service.retrieve(
            queries=_build_memu_queries(request),
            where=_build_memu_where(request),
        )
        hits = _map_memu_hits(response)
        text_block, injected_ids = _build_memu_injection_block(hits, top_k=request.top_k)
        hits = _mark_injected_hits(hits, injected_ids)
        return MemoryEngineRetrieveResult(
            text_block=text_block,
            hits=hits,
            trace={
                "engine": self.DESCRIPTOR.name,
                "profile": self.DESCRIPTOR.profile.value,
                "mode": request.mode,
                "needs_retrieval": bool(response.get("needs_retrieval", True)),
                "rewritten_query": response.get("rewritten_query", request.query),
            },
            raw=response if isinstance(response, dict) else {},
        )

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        if request.source_kind not in {"conversation_turn", "conversation_batch"}:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported source_kind",
                raw={"reason": "unsupported_source_kind"},
            )

        text = _build_memu_text_payload(request.content)
        if not text:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported content for memu ingest",
                raw={"reason": "invalid_content"},
            )

        scope = request.scope
        safe_session = (scope.session_key or "dev").replace("/", "_").replace(":", "_")
        out_dir = self._input_dir / safe_session
        out_dir.mkdir(parents=True, exist_ok=True)
        source_path = out_dir / f"{int(time.time() * 1000)}.txt"
        source_path.write_text(text, encoding="utf-8")

        result = await self._service.memorize(
            resource_url=str(source_path),
            modality="text",
            user=_build_memu_user(scope),
        )
        items = result.get("items")
        created_ids = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    created_ids.append(str(item["id"]))

        return MemoryIngestResult(
            accepted=True,
            created_ids=created_ids,
            summary="memorized by memu",
            raw={
                "resource_path": str(source_path),
                "engine": self.DESCRIPTOR.name,
                "response": result if isinstance(result, dict) else {},
            },
        )

    async def remember(self, request: RememberRequest) -> RememberResult:
        scope = request.scope
        safe_session = (scope.session_key or "dev").replace("/", "_").replace(":", "_")
        out_dir = self._input_dir / safe_session
        out_dir.mkdir(parents=True, exist_ok=True)
        source_path = out_dir / f"{int(time.time() * 1000)}_remember.txt"
        source_path.write_text(request.summary.strip(), encoding="utf-8")

        result = await self._service.memorize(
            resource_url=str(source_path),
            modality="text",
            user=_build_memu_user(scope),
        )
        item_id = ""
        items = result.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    item_id = str(item["id"])
                    break
        if not item_id:
            item_id = source_path.stem
        return RememberResult(
            item_id=item_id,
            actual_type=request.memory_type,
            write_status="new",
            superseded_ids=[],
        )

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR


def _build_memu_queries(request: MemoryEngineRetrieveRequest) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    recent_turns = request.context.get("recent_turns")
    if isinstance(recent_turns, str) and recent_turns.strip():
        queries.append({"role": "system", "content": recent_turns.strip()})
    queries.append({"role": "user", "content": request.query})
    return queries


def _build_memu_where(request: MemoryEngineRetrieveRequest) -> dict[str, object]:
    return _build_memu_user(request.scope)


def _build_memu_user(scope) -> dict[str, object]:
    where: dict[str, object] = {}
    if scope.channel:
        where["channel"] = scope.channel
    if scope.chat_id:
        where["chat_id"] = scope.chat_id
    if scope.session_key:
        where["session_key"] = scope.session_key
    return where


def _map_memu_hits(response: dict[str, object]) -> list[MemoryHit]:
    hits: list[MemoryHit] = []
    for kind in ("items", "resources", "categories"):
        bucket = response.get(kind)
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            metadata = dict(item)
            item_id = str(metadata.pop("id", "") or "")
            summary = str(
                metadata.get("summary")
                or metadata.get("name")
                or metadata.get("content")
                or ""
            )
            content = str(metadata.get("content") or summary)
            score = float(metadata.get("score", 0.0) or 0.0)
            source_ref = str(metadata.get("resource_id") or metadata.get("id") or "")
            hits.append(
                MemoryHit(
                    id=item_id,
                    summary=summary,
                    content=content,
                    score=score,
                    source_ref=source_ref,
                    engine_kind=kind.rstrip("s"),
                    metadata=metadata,
                )
            )
    return hits


def _build_memu_injection_block(
    hits: list[MemoryHit],
    *,
    top_k: int | None,
) -> tuple[str, list[str]]:
    sorted_hits = sorted(
        [hit for hit in hits if hit.summary.strip()],
        key=lambda hit: (hit.score, hit.id),
        reverse=True,
    )
    limit = max(1, int(top_k or len(sorted_hits) or 1))
    selected = sorted_hits[:limit]
    if not selected:
        return "", []
    lines = [f"- {hit.summary.strip()}" for hit in selected]
    return "## 【相关记忆】\n" + "\n".join(lines), [hit.id for hit in selected if hit.id]


def _mark_injected_hits(hits: list[MemoryHit], injected_ids: list[str]) -> list[MemoryHit]:
    injected = set(injected_ids)
    for hit in hits:
        hit.injected = hit.id in injected
    return hits


def _build_memu_text_payload(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        user_message = str(content.get("user_message", "") or "").strip()
        assistant_response = str(content.get("assistant_response", "") or "").strip()
        tool_chain = content.get("tool_chain")
        parts: list[str] = []
        if user_message:
            parts.append(f"user: {user_message}")
        if assistant_response:
            parts.append(f"assistant: {assistant_response}")
        if isinstance(tool_chain, list) and tool_chain:
            parts.append("tool_chain:")
            parts.append(json.dumps(tool_chain, ensure_ascii=False))
        return "\n".join(parts).strip()
    if not isinstance(content, list):
        return ""

    lines: list[str] = []
    for message in content:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "") or "").strip()
        body = str(message.get("content", "") or "").strip()
        if role and body:
            lines.append(f"{role}: {body}")
    return "\n".join(lines).strip()
