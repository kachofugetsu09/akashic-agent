"""Akasic Agent MemoryEngine adapter for Akasha V2."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import numpy as np

from agent.config_models import Config
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import (
    EngineProfile,
    EvidenceRef,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryRecord,
    MemoryToolProfile,
    MemoryToolSpec,
)
from memory2.embedder import Embedder
from session.embedding_store import MessageEmbeddingStore

from .application.cycle import RetrievalTicket
from .application.runtime import OnlineMemoryRuntime
from .config import AkashaConfig, resolve_workspace_path
from .domain.model import Turn

if TYPE_CHECKING:
    from bus.event_bus import EventBus
    from core.net.http import SharedHttpResources


class UnsupportedOperationError(RuntimeError):
    """Report a write operation outside Akasha's unsupervised contract."""


@dataclass(frozen=True)
class PendingRetrieval:
    """Bind one stateful query to the later committed source turn."""

    ticket: RetrievalTicket
    query_timestamp: datetime
    query_text: str
    query_dense: np.ndarray


@dataclass(frozen=True)
class RetrievalRecords:
    """Keep direct dense recall separate from explicit pattern completion."""

    dense: tuple[MemoryRecord, ...]
    completion: tuple[MemoryRecord, ...]

    @property
    def combined(self) -> list[MemoryRecord]:
        return [*self.dense, *self.completion]


class AkashaMemoryEngine:
    """Adapt the standalone explicit memory runtime to Akasic Agent."""

    DESCRIPTOR = MemoryEngineDescriptor(
        name="akasha",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_MESSAGES,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.ENRICH_GRAPH_RELATIONS,
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={
            "truth": "sessions.db/messages",
            "learning": "unsupervised_memory_cycle",
            "external_reinforcement": "ignored_by_design",
        },
    )

    def __init__(
        self,
        *,
        config: Config,
        akasha_config: AkashaConfig,
        workspace: Path,
        http_resources: SharedHttpResources,
        event_publisher: EventBus | None,
    ) -> None:
        """Load persisted memory, construct embedding, and wire commits."""

        # 1. Construct host infrastructure and restore the sidecar state.
        embedding = config.memory.embedding
        self._embedding_model = embedding.model
        self._embedder = Embedder(
            base_url=(
                embedding.base_url
                or config.light_base_url
                or config.base_url
                or ""
            ),
            api_key=(
                embedding.api_key
                or config.light_api_key
                or config.api_key
            ),
            model=embedding.model,
            output_dimensionality=embedding.output_dimensionality,
            requester=http_resources.external_default,
        )
        self._config = akasha_config
        self._workspace = workspace
        self._sessions_path = workspace / "sessions.db"
        self._embedding_store = MessageEmbeddingStore(
            self._sessions_path
        )
        self._runtime = OnlineMemoryRuntime(
            sessions_path=self._sessions_path,
            index_path=resolve_workspace_path(
                workspace,
                akasha_config.index_path,
            ),
            memory_path=resolve_workspace_path(
                workspace,
                akasha_config.db_path,
            ),
            embedding_model=embedding.model,
            embedding_dimension=embedding.output_dimensionality,
            config=akasha_config.memory_config(),
        )

        # 2. Keep one global graph writer and one pending query per session.
        self._lock = threading.RLock()
        self._pending: dict[str, PendingRetrieval] = {}
        self.closeables: list[object] = [
            self._runtime,
            self._embedding_store,
            self._embedder,
        ]
        if event_publisher is not None:
            self.closeables.append(
                event_publisher.on(
                    TurnCommitted,
                    self._on_turn_committed,
                )
            )

    @property
    def embedding_api(self) -> Embedder:
        return self._embedder

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        """Retrieve explicit completion and optionally retain its ticket."""

        # 1. Validate the host query boundary and unsupported intent.
        if request.intent == "timeline":
            return MemoryQueryResult(
                trace={
                    "engine": "akasha",
                    "intent": "timeline_unsupported",
                }
            )
        text = request.text.strip()
        if not text:
            return MemoryQueryResult(
                trace={"engine": "akasha", "hit_count": 0}
            )
        if request.timestamp is None:
            raise ValueError("Akasha query requires timestamp")
        if request.limit <= 0:
            raise ValueError("Akasha query limit must be positive")
        for name, value in (
            ("time_start", request.filters.time_start),
            ("time_end", request.filters.time_end),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"Akasha {name} must be timezone-aware")

        # 2. Embed the cue and run a non-mutating shared MemoryCycle read.
        dense = np.asarray(
            await self._embedder.embed(text),
            dtype=np.float32,
        )
        with self._lock:
            cue, ticket = self._runtime.query_turn(
                text=text,
                dense=dense,
                session_key=request.scope.session_key,
                timestamp=request.timestamp,
            )
            lanes = self._records(ticket, cue, request)
            retains_ticket = (
                request.intent == "context"
                and request.effect == "stateful"
                and bool(request.scope.session_key)
            )
            if retains_ticket:
                session_key = request.scope.session_key
                if not session_key:
                    raise RuntimeError(
                        "stateful context query lost its session key"
                    )
                self._pending[session_key] = PendingRetrieval(
                    ticket,
                    request.timestamp,
                    text,
                    dense.copy(),
                )
        # 3. Render context only for the runtime context-injection intent.
        text_block = (
            self._context_block(lanes, request.timestamp)
            if request.intent == "context"
            else ""
        )
        return MemoryQueryResult(
            text_block=text_block,
            records=lanes.combined,
            trace={
                "engine": "akasha",
                "requested_effect": request.effect,
                "effect": (
                    "stateful" if retains_ticket else "read_only"
                ),
                "state_version": ticket.state_version,
                "seed_count": len(ticket.evidence.seed),
                "dense_count": len(lanes.dense),
                "active_basin_count": (
                    ticket.completion.active_basin_count
                ),
                "completion_count": len(lanes.completion),
                "pushes": ticket.completion.pushes,
                "residual_l1": ticket.completion.residual_l1,
            },
        )

    async def ingest(
        self,
        request: MemoryIngestRequest,
    ) -> MemoryIngestResult:
        """Reject non-canonical writes; TurnCommitted owns ingestion."""

        return MemoryIngestResult(
            accepted=False,
            summary="Akasha ingestion requires TurnCommitted message IDs",
            raw={
                "reason": "requires_persisted_messages",
                "source_kind": request.source_kind,
            },
        )

    async def mutate(
        self,
        request: MemoryMutation,
    ) -> MemoryMutationResult:
        """Reject manual remember and forget outside the fact source."""

        if request.kind == "forget":
            return MemoryMutationResult(
                accepted=False,
                status="unsupported",
                missing_ids=list(request.ids),
            )
        return MemoryMutationResult(
            accepted=False,
            status="unsupported",
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        """Ignore external reinforcement by the unsupervised design."""

        _ = ids

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile(
            recall=MemoryToolSpec(
                description=(
                    "从 Akasha V2 显式记忆图召回历史对话和关联情景。"
                    "结果包含原始消息证据和模式补全来源。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要回忆的历史主题",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 40,
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
                search_hint="历史对话 情景回忆 关联记忆",
            )
        )

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        _ = action_tokens
        return []

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        selected = [
            turn
            for turn in self._runtime.cycle.turns
            if time_start
            <= datetime.fromisoformat(turn.started_at)
            <= time_end
        ]
        return [
            self._turn_row(turn)
            for turn in selected[-limit:]
        ]

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
        del memory_type, status, scope_channel, scope_chat_id
        del has_embedding, sort_by
        rows = [
            self._turn_row(turn)
            for turn in self._runtime.cycle.turns
            if (
                not q
                or q in turn.user_text
                or q in turn.assistant_text
            )
            and (
                not source_ref
                or source_ref in {
                    turn.user_message_id,
                    turn.assistant_message_id,
                }
            )
        ]
        rows.sort(
            key=lambda row: cast(int, row["node_id"]),
            reverse=sort_order == "desc",
        )
        total = len(rows)
        start = (page - 1) * page_size
        return rows[start : start + page_size], total

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        _ = include_embedding
        for turn in self._runtime.cycle.turns:
            if turn.turn_id == item_id:
                return self._turn_row(turn)
        return None

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
        del status, extra_json, source_ref, happened_at
        del emotional_weight
        raise UnsupportedOperationError(
            f"Akasha dashboard state is read-only: {item_id}"
        )

    def delete_item(self, item_id: str) -> bool:
        raise UnsupportedOperationError(
            f"Akasha does not delete source turn: {item_id}"
        )

    def delete_items_batch(self, ids: list[str]) -> int:
        raise UnsupportedOperationError(
            f"Akasha does not delete source turns: {ids}"
        )

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        del memory_type, include_superseded
        turns = self._runtime.cycle.turns
        query = next(
            (turn for turn in turns if turn.turn_id == item_id),
            None,
        )
        if query is None or query.user_dense is None:
            return []
        scored = []
        for turn in turns:
            if turn.turn_id == item_id or turn.user_dense is None:
                continue
            score = float(np.dot(query.user_dense, turn.user_dense))
            if score >= score_threshold:
                row = self._turn_row(turn)
                row["score"] = score
                scored.append(row)
        return sorted(
            scored,
            key=lambda row: (-float(row["score"]), int(row["node_id"])),
        )[:top_k]

    async def _on_turn_committed(self, event: TurnCommitted) -> None:
        """Persist embeddings and apply one canonical MemoryCycle commit."""

        # 1. Respect the host's explicit exclusion and validate stable IDs.
        if (
            event.session_key.split(":", 1)[0] == "scheduler"
            or bool((event.extra or {}).get("skip_post_memory"))
        ):
            with self._lock:
                self._pending.pop(event.session_key, None)
            return
        user_id = event.persisted_user_message_id
        assistant_id = event.assistant_message_id
        if not user_id or not assistant_id:
            raise ValueError(
                "TurnCommitted requires persisted user and assistant IDs"
            )

        # 2. Embed exact persisted text and update the canonical embedding cache.
        messages = _load_messages(
            self._sessions_path,
            event.session_key,
            user_id,
            assistant_id,
        )
        with self._lock:
            pending = self._pending.get(event.session_key)
        if (
            pending is not None
            and pending.query_text == cast(str, messages[0]["content"])
        ):
            assistant_vector = await self._embedder.embed(
                cast(str, messages[1]["content"])
            )
            vectors = [
                pending.query_dense.tolist(),
                assistant_vector,
            ]
        else:
            vectors = await self._embedder.embed_batch(
                [cast(str, message["content"]) for message in messages]
            )
        _upsert_embeddings(
            self._embedding_store,
            self._embedding_model,
            messages,
            vectors,
        )

        # 3. Commit the matching ticket or recompute on the latest state.
        with self._lock:
            if self._pending.get(event.session_key) is pending:
                self._pending.pop(event.session_key, None)
            self._runtime.commit_from_source(
                user_message_id=user_id,
                assistant_message_id=assistant_id,
                ticket=None if pending is None else pending.ticket,
            )

    def _records(
        self,
        ticket: RetrievalTicket,
        cue: Turn,
        request: MemoryQuery,
    ) -> RetrievalRecords:
        """Select direct dense and explicit completion as independent lanes."""

        # 1. Select dense evidence before chronological presentation.
        turns = self._runtime.cycle.turns
        completion_limit = (
            self._config.context_recall_limit
            if request.intent == "context"
            else request.limit
        )
        dense_limit = min(5, request.limit)
        completion_nodes = {
            item.node_id: item for item in ticket.completion.items
        }
        dense_candidates = []
        for turn in turns:
            score = _dense_score(cue.user_dense, turn)
            if score is None or not _matches_filters(
                turn,
                ("direct_dense",),
                request,
            ):
                continue
            dense_candidates.append((turn.node_id, score))
        dense_candidates.sort(key=lambda item: (-item[1], item[0]))
        dense_nodes = {
            node_id for node_id, _ in dense_candidates[:dense_limit]
        }
        dense_records = [
            _memory_record(
                turns[node_id],
                score=score,
                lane="dense",
                sources=("direct_dense",),
                basin_ids=(),
                injected=request.intent == "context",
                also_completed=node_id in completion_nodes,
            )
            for node_id, score in dense_candidates[:dense_limit]
        ]

        # 2. Fill completion after stable-ID dedupe against the dense lane.
        completion_records = []
        for item in ticket.completion.items:
            if item.node_id in dense_nodes:
                continue
            turn = turns[item.node_id]
            if not _matches_filters(turn, item.sources, request):
                continue
            completion_records.append(
                _memory_record(
                    turn,
                    score=item.score,
                    lane="completion",
                    sources=item.sources,
                    basin_ids=item.basin_ids,
                    injected=request.intent == "context",
                    also_completed=False,
                )
            )
            if len(completion_records) == completion_limit:
                break

        # 3. Ranking chooses membership; chronology chooses presentation.
        return RetrievalRecords(
            dense=tuple(_sort_records_by_time(dense_records)),
            completion=tuple(
                _sort_records_by_time(completion_records)
            ),
        )

    def _context_block(
        self,
        lanes: RetrievalRecords,
        timestamp: datetime,
    ) -> str:
        """Render the legacy left/right memory layout within one budget."""

        # 1. Render each lane without adding per-item semantic labels.
        if not lanes.dense and not lanes.completion:
            return ""
        parts = [
            (
                "# Akasha memory now="
                f"{timestamp.astimezone(ZoneInfo('Asia/Shanghai')):%m-%d}"
            )
        ]
        if lanes.dense:
            parts.append(
                _format_records(
                    "## 左脑记忆：精确回忆",
                    lanes.dense,
                )
            )
        if lanes.completion:
            parts.append(
                _format_records(
                    "## 右脑联想：潜意识第一反应",
                    lanes.completion,
                )
            )

        # 2. Apply the host injection budget after complete formatting.
        text = "\n\n".join(parts)
        if len(text) <= self._config.inject_max_chars:
            return text
        omitted = len(text) - self._config.inject_max_chars
        return (
            text[: self._config.inject_max_chars].rstrip()
            + f"\n...[Akasha 已截断 {omitted} 字]"
        )

    @staticmethod
    def _turn_row(turn) -> dict[str, object]:
        return {
            "id": turn.turn_id,
            "node_id": turn.node_id,
            "memory_type": "episodic_turn",
            "summary": _turn_summary(
                turn.user_text,
                turn.assistant_text,
            ),
            "source_ref": turn.session_key,
            "created_at": turn.started_at,
            "status": "active",
        }


def _load_messages(
    sessions_path: Path,
    session_key: str,
    user_id: str,
    assistant_id: str,
) -> list[dict[str, object]]:
    connection = sqlite3.connect(
        f"file:{sessions_path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, session_key, seq, role, content, ts
            FROM messages
            WHERE id IN (?, ?)
            ORDER BY seq
            """,
            (user_id, assistant_id),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 2:
        raise ValueError("TurnCommitted messages do not exist")
    if (
        rows[0]["id"] != user_id
        or rows[0]["role"] != "user"
        or rows[1]["id"] != assistant_id
        or rows[1]["role"] != "assistant"
        or any(row["session_key"] != session_key for row in rows)
    ):
        raise ValueError("TurnCommitted message identity mismatch")
    return [dict(row) for row in rows]


def _upsert_embeddings(
    store: MessageEmbeddingStore,
    model: str,
    messages: list[dict[str, object]],
    vectors: list[list[float]],
) -> None:
    if len(vectors) != len(messages):
        raise ValueError("embedding count differs from committed messages")
    for message, raw_vector in zip(messages, vectors, strict=True):
        vector = np.asarray(raw_vector, dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError("committed embedding must be one finite vector")
        store.upsert(
            message_id=str(message["id"]),
            content=str(message["content"]),
            model=model,
            embedding=vector.tolist(),
        )


def _memory_record(
    turn: Turn,
    *,
    score: float,
    lane: str,
    sources: tuple[str, ...],
    basin_ids: tuple[str, ...],
    injected: bool,
    also_completed: bool,
) -> MemoryRecord:
    """Expose one historical turn with stable source identity."""

    preview = _assistant_preview(turn.assistant_text)
    return MemoryRecord(
        id=turn.turn_id,
        kind="episodic_turn",
        summary=_turn_summary(turn.user_text, preview),
        score=score,
        engine_kind="akasha",
        evidence=[
            EvidenceRef(
                kind="message_range",
                refs=[
                    turn.user_message_id,
                    turn.assistant_message_id,
                ],
                source_ref=turn.session_key,
            )
        ],
        signals={
            "lane": lane,
            "sources": list(sources),
            "basin_ids": list(basin_ids),
            "completion": lane == "completion",
            "also_completed": also_completed,
            "started_at": turn.started_at,
            "user_text": turn.user_text,
            "assistant_preview": preview,
        },
        injected=injected,
    )


def _dense_score(
    query: np.ndarray | None,
    turn: Turn,
) -> float | None:
    if query is None:
        return None
    scores = [
        float(np.dot(query, vector))
        for vector in (turn.user_dense, turn.assistant_dense)
        if vector is not None
    ]
    return max(scores) if scores else None


def _assistant_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= 50:
        return normalized
    return normalized[:50] + "..."


def _turn_summary(user: str, assistant: str) -> str:
    return f"用户：{user}\n助手：{assistant}"


def _sort_records_by_time(
    records: list[MemoryRecord],
) -> list[MemoryRecord]:
    return sorted(
        records,
        key=lambda record: (
            _parse_turn_time(str(record.signals["started_at"])),
            record.id.encode("utf-8"),
        ),
        reverse=True,
    )


def _format_records(
    title: str,
    records: tuple[MemoryRecord, ...],
) -> str:
    lines = [title]
    for record in records:
        user = str(record.signals["user_text"])
        assistant = str(record.signals["assistant_preview"])
        timestamp = _parse_turn_time(
            str(record.signals["started_at"])
        ).astimezone(ZoneInfo("Asia/Shanghai"))
        refs = record.evidence[0].refs
        lines.append(
            f"- user={_json_string(user)} "
            f"assistant={_json_string(assistant)} "
            f"t={timestamp:%m-%d} "
            f"source_ref={json.dumps(refs, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_turn_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return parsed


def _matches_filters(
    turn,
    sources: tuple[str, ...],
    request: MemoryQuery,
) -> bool:
    """Apply caller-owned output filters without changing graph dynamics."""

    # 1. Match the only supported record kind and optional time interval.
    filters = request.filters
    if filters.kinds and "episodic_turn" not in filters.kinds:
        return False
    started = datetime.fromisoformat(turn.started_at)
    if started.tzinfo is None:
        started = started.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    if filters.time_start is not None and started < filters.time_start:
        return False
    if filters.time_end is not None and started > filters.time_end:
        return False

    # 2. Strong relevance excludes weak relative-tail-only associations.
    return (
        filters.relevance_floor != "strong"
        or any(source != "relative_tail" for source in sources)
    )
