"""Akasic Agent MemoryEngine adapter for Akasha V2."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast
from zoneinfo import ZoneInfo

import numpy as np

from agent.config_models import Config
from agent.control.context import running_turn_id
from agent.tools.base import Tool
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
from session.store import InteractionDeletion

from .application.cycle import RetrievalTicket
from .application.runtime import OnlineMemoryRuntime, StagedOnlineCommit
from .config import AkashaConfig, resolve_workspace_path
from .domain.model import Turn

if TYPE_CHECKING:
    from bus.event_bus import EventBus
    from core.net.http import SharedHttpResources


class UnsupportedOperationError(RuntimeError):
    """Report a write operation outside Akasha's unsupervised contract."""


@dataclass(frozen=True)
class RetrievalRecords:
    """Keep direct dense recall separate from explicit pattern completion."""

    dense: tuple[MemoryRecord, ...]
    completion: tuple[MemoryRecord, ...]

    @property
    def combined(self) -> list[MemoryRecord]:
        return [*self.dense, *self.completion]


@dataclass(frozen=True)
class PendingRetrieval:
    """Bind one stateful query and its prompt lanes to the active host turn."""

    ticket: RetrievalTicket
    query_timestamp: datetime
    query_text: str
    query_dense: np.ndarray
    turn_id: str
    records: RetrievalRecords


@dataclass(frozen=True)
class ActiveRecallSnapshot:
    """Expose the exact pending prompt lanes without mutable graph state."""

    query_id: str
    records: RetrievalRecords


@dataclass(frozen=True)
class AkashaFeedbackMarker:
    """Persist one agent-selected Message-level feedback event."""

    action: Literal["remember", "forget"]
    target_message_ids: tuple[str, ...]
    target_turn_ids: tuple[str, ...]
    reason: str

    @property
    def extra_key(self) -> str:
        return (
            "akasha_reinforce"
            if self.action == "remember"
            else "akasha_forget"
        )

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "action": self.action,
            "target_message_ids": list(self.target_message_ids),
            "target_turn_ids": list(self.target_turn_ids),
            "source": f"agent.{self.action}_memory",
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.action == "remember":
            payload["boost"] = 3.0
        return payload


class _AkashaFeedbackTool(Tool):
    """Stage one of the two Message-level feedback primitives."""

    name = "_akasha_feedback"
    action: Literal["remember", "forget"]
    description = "由 Akasha tool_profile 注入工具描述。"
    parameters = {
        "type": "object",
        "properties": {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "reason": {"type": "string"},
        },
        "required": ["message_ids"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        memory: AkashaMemoryEngine,
        spec: MemoryToolSpec,
    ) -> None:
        self._memory = memory
        self.description = spec.description
        self.parameters = spec.parameters

    async def execute(
        self,
        message_ids: list[str],
        reason: str = "",
        **_: Any,
    ) -> str:
        return json.dumps(
            self._memory.stage_feedback(
                turn_id=running_turn_id.get(),
                action=self.action,
                message_ids=message_ids,
                reason=reason,
            ),
            ensure_ascii=False,
        )


class AkashaRememberTool(_AkashaFeedbackTool):
    """Strengthen selected history or the user's current correction."""

    name = "remember_memory"
    action = "remember"


class AkashaForgetTool(_AkashaFeedbackTool):
    """Suppress selected history without inventing a replacement."""

    name = "forget_memory"
    action = "forget"


class AkashaFeedbackPersistModule:
    """Export staged feedback markers into the current user message."""

    slot = "akasha.feedback.persist"
    requires = ("after_reasoning.emit", "reasoning:ctx")

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    async def run(self, frame: Any) -> Any:
        engine = self._plugin.context.memory_engine
        if engine is None or engine.describe().name != "akasha":
            return frame
        markers = cast(
            AkashaMemoryEngine,
            engine,
        ).take_staged_feedback(running_turn_id.get())
        for marker in markers:
            frame.slots[f"persist:user:{marker.extra_key}"] = marker.payload()
        return frame


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
            "message_feedback": "remember_and_forget_markers",
            "legacy_item_reinforcement": "ignored_by_design",
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
        self._pending_changed = threading.Condition(self._lock)
        self._source_event_gate = asyncio.Lock()
        self._commit_gate = asyncio.Lock()
        self._publish_task: asyncio.Task[None] | None = None
        self._pending: dict[str, PendingRetrieval] = {}
        self._source_generation = 0
        self._source_invalidated_error: RuntimeError | None = None
        self._staged_feedback: dict[
            str,
            dict[Literal["remember", "forget"], AkashaFeedbackMarker],
        ] = {}
        self.closeables: list[object] = [
            self._runtime,
            self._embedding_store,
            self._embedder,
            self,
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

        # 2. Embed without blocking commits, then fence the shared graph read.
        dense = np.asarray(
            await self._embedder.embed(text),
            dtype=np.float32,
        )
        async with self._commit_gate:
            await self._wait_for_publication()
            with self._lock:
                self._require_valid_source()
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
                    turn_id = request.context.get("turn_id", "")
                    if not isinstance(turn_id, str):
                        raise ValueError(
                            "Akasha context turn_id must be a string"
                        )
                    self._pending[session_key] = PendingRetrieval(
                        ticket,
                        request.timestamp,
                        text,
                        dense.copy(),
                        turn_id,
                        lanes,
                    )
                    self._pending_changed.notify_all()
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

    def wait_for_active_recall(
        self,
        session_key: str,
        turn_id: str,
        *,
        timeout: float = 15.0,
    ) -> ActiveRecallSnapshot | None:
        """Wait for and return only the retrieval bound to one active turn."""

        deadline = monotonic() + max(0.0, timeout)
        with self._pending_changed:
            while True:
                pending = self._pending.get(session_key)
                if pending is not None and pending.turn_id == turn_id:
                    return ActiveRecallSnapshot(
                        query_id=pending.ticket.turn_id,
                        records=pending.records,
                    )
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return None
                self._pending_changed.wait(remaining)

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
            ),
            tools=(
                MemoryToolSpec(
                    name="remember_memory",
                    description=(
                        "记住或强化用户明确确认正确、有用、以后应优先的内容。"
                        "参数接受 fetch_messages 返回的 Message ID；如果正确内容就在"
                        "用户当前消息里，传保留值 current_user_message。"
                        "不要传 recall_memory 的记忆条目 id、会话 source_ref 或 "
                        "Akasha 图节点 id。"
                        "调用前必须先用 recall_memory 或 search_messages 找候选，再用 "
                        "fetch_messages 读取原文并取得 messages[].id。"
                        "recall_memory 返回的 evidence[].refs 是可回源的 Message ID；"
                        "不要把仅表示会话的 evidence[].source_ref 当作 Message ID。"
                        "用户纠正旧内容时，用 forget_memory 忘记旧 Message，再用本工具"
                        "记住正确 Message 或 current_user_message；没有第三种 correct 动作。"
                        "仅仅回答了一个问题、召回了一段历史或模型自行觉得重要，不得调用。"
                    ),
                    parameters=AkashaRememberTool.parameters,
                    risk="write",
                    search_hint="记住 强化 偏好 正确纠正",
                    tool_class=AkashaRememberTool,
                ),
                MemoryToolSpec(
                    name="forget_memory",
                    description=(
                        "忘记或抑制用户明确要求撤回、判错、过时或不再使用的历史内容。"
                        "参数只接受 fetch_messages 返回的旧 Message ID；"
                        "不要传 recall_memory 记忆条目 id、会话 source_ref、"
                        "Akasha 图节点 id 或 current_user_message。"
                        "调用前必须先 recall_memory 或 search_messages，再用 "
                        "fetch_messages 读原文并取得 messages[].id。"
                        "如果用户同时给出正确替代，先用本工具忘记旧 Message，"
                        "再调用 remember_memory 记住正确 Message 或 "
                        "current_user_message；没有第三种 correct 动作。"
                        "如果只是纯忘记，只调用本工具，并在成功后中性确认；"
                        "不得把被忘内容的反命题当成新事实或新偏好。"
                        "上一段对话已经结束也不受影响：定位旧 Message ID 后在本轮调用，"
                        "系统会把它转换成 Akasha turn。"
                    ),
                    parameters=AkashaForgetTool.parameters,
                    risk="write",
                    search_hint="忘记 撤回 错误 过时 不要再提",
                    tool_class=AkashaForgetTool,
                ),
            ),
        )

    def stage_feedback(
        self,
        *,
        turn_id: str,
        action: str,
        message_ids: list[str],
        reason: str,
    ) -> dict[str, object]:
        """Resolve Message IDs and stage one marker for this active turn."""

        # 1. Validate the agent tool boundary before resolving graph identity.
        clean_ids = _clean_feedback_message_ids(message_ids)
        clean_reason = reason.strip()
        error = _feedback_request_error(
            turn_id=turn_id,
            action=action,
            message_ids=clean_ids,
            reason=clean_reason,
        )
        if error is not None:
            return error

        # 2. Convert public Message identities to canonical Akasha turns.
        resolved = self._resolve_and_stage_feedback(
            turn_id=turn_id,
            action=cast(
                Literal["remember", "forget"],
                action,
            ),
            message_ids=clean_ids,
            reason=clean_reason,
        )
        if isinstance(resolved, dict):
            return resolved
        marker = resolved

        # 3. Report staging honestly; persistence happens after reasoning.
        return {
            "status": "staged",
            "action": marker.action,
            "target_message_ids": list(marker.target_message_ids),
            "target_turn_ids": list(marker.target_turn_ids),
            "applies_after": "current_turn_commit",
        }

    def _resolve_and_stage_feedback(
        self,
        *,
        turn_id: str,
        action: Literal["remember", "forget"],
        message_ids: list[str],
        reason: str,
    ) -> AkashaFeedbackMarker | dict[str, object]:
        """Resolve Message IDs and atomically merge one active-turn marker."""

        with self._lock:
            turns_by_message = {
                message_id: turn
                for turn in self._runtime.cycle.turns
                for message_id in (
                    turn.user_message_id,
                    turn.assistant_message_id,
                )
            }
            missing = [
                message_id
                for message_id in message_ids
                if (
                    message_id not in turns_by_message
                    and not (
                        action == "remember"
                        and message_id == "current_user_message"
                    )
                )
            ]
            if missing:
                return _missing_feedback_messages(missing)
            marker = AkashaFeedbackMarker(
                action=action,
                target_message_ids=tuple(message_ids),
                target_turn_ids=tuple(
                    dict.fromkeys(
                        (
                            "current_turn"
                            if message_id == "current_user_message"
                            else turns_by_message[message_id].turn_id
                        )
                        for message_id in message_ids
                    )
                ),
                reason=reason,
            )
            staged = self._staged_feedback.setdefault(turn_id, {})
            opposite = staged.get(
                "forget" if action == "remember" else "remember"
            )
            overlap = (
                set(marker.target_turn_ids)
                & set(opposite.target_turn_ids)
                if opposite is not None
                else set()
            )
            if overlap:
                return {
                    "status": "not_staged",
                    "error": "conflicting_feedback_actions",
                    "target_turn_ids": sorted(overlap),
                }
            previous = staged.get(action)
            if previous is not None:
                marker = _merge_feedback_markers(previous, marker)
            staged[action] = marker
            return marker

    def take_staged_feedback(
        self,
        turn_id: str,
    ) -> tuple[AkashaFeedbackMarker, ...]:
        """Consume both marker primitives owned by one active host turn."""

        if not turn_id:
            return ()
        with self._lock:
            staged = self._staged_feedback.pop(turn_id, {})
        return tuple(
            staged[action]
            for action in ("forget", "remember")
            if action in staged
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
        self._require_valid_source()
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
        self._require_valid_source()
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
        self._require_valid_source()
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
        self._require_valid_source()
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
        """Serialize source-event embedding and staging with source deletion."""

        with self._lock:
            source_generation = self._source_generation
            source_was_invalid = self._source_invalidated_error is not None
        async with self._source_event_gate:
            with self._lock:
                if (
                    source_was_invalid
                    or source_generation != self._source_generation
                ):
                    return
            await self._commit_source_event(event)

    async def _commit_source_event(self, event: TurnCommitted) -> None:
        """Stage one committed turn and publish its graph asynchronously."""

        # 1. Respect the host's explicit exclusion and validate stable IDs.
        if (
            event.session_key.split(":", 1)[0] == "scheduler"
            or bool((event.extra or {}).get("skip_post_memory"))
        ):
            with self._lock:
                self._pending.pop(event.session_key, None)
            return
        user_ids = event.persisted_user_message_ids or (
            (event.persisted_user_message_id,)
            if event.persisted_user_message_id
            else ()
        )
        user_id = user_ids[0] if user_ids else None
        assistant_id = event.assistant_message_id
        if not user_id or not assistant_id:
            raise ValueError(
                "TurnCommitted requires persisted user and assistant IDs"
            )

        # 2. Embed exact persisted text without blocking other provider calls.
        messages = _load_messages(
            self._sessions_path,
            event.session_key,
            user_ids,
            assistant_id,
        )
        with self._lock:
            pending = self._pending.get(event.session_key)
        if (
            len(user_ids) == 1
            and pending is not None
            and pending.query_text == cast(str, messages[0]["content"])
        ):
            assistant_vector = await self._embedder.embed(
                cast(str, messages[-1]["content"])
            )
            vectors = [
                pending.query_dense.tolist(),
                assistant_vector,
            ]
        else:
            vectors = await self._embedder.embed_batch(
                [cast(str, message["content"]) for message in messages]
            )
        # 3. Serialize durable staging behind the prior graph publication.
        async with self._commit_gate:
            await self._wait_for_publication()
            with self._lock:
                self._require_valid_source()
                _upsert_embeddings(
                    self._embedding_store,
                    self._embedding_model,
                    messages,
                    vectors,
                )
                selected_ticket = None
                if self._pending.get(event.session_key) is pending:
                    self._pending.pop(event.session_key, None)
                    selected_ticket = None if pending is None else pending.ticket
                staged = self._runtime.stage_from_source(
                    user_message_id=user_id,
                    assistant_message_id=assistant_id,
                    ticket=selected_ticket,
                )
            self._publish_task = asyncio.create_task(
                asyncio.to_thread(self._publish_staged, staged),
                name="akasha-publish-staged",
            )

    async def delete_interaction_source(
        self,
        control_turn_id: str,
        delete_source: Callable[[], InteractionDeletion | None],
    ) -> InteractionDeletion | None:
        """封住在线读写，执行窄 source 删除并重建派生状态。"""

        # 1. 与在线 commit 串行，整个 source mutation 窗口不开放旧图。
        async with self._source_event_gate:
            async with self._commit_gate:
                await self._wait_for_publication()
                return await asyncio.to_thread(
                    self._delete_source_and_rebuild,
                    control_turn_id,
                    delete_source,
                )

    def _delete_source_and_rebuild(
        self,
        control_turn_id: str,
        delete_source: Callable[[], InteractionDeletion | None],
    ) -> InteractionDeletion | None:
        """在工作线程内完成 source transaction 与确定性派生发布。"""

        # 1. 先使所有非 gate 管理读也 fail-loud，再调用唯一授权的删除动作。
        with self._lock:
            self._source_invalidated_error = RuntimeError(
                "Akasha source mutation is in progress"
            )
        try:
            deletion = delete_source()
        except Exception:
            with self._lock:
                self._source_invalidated_error = None
            raise
        if deletion is None:
            with self._lock:
                self._source_invalidated_error = None
            return None
        if deletion.control_turn_id != control_turn_id:
            with self._lock:
                self._source_invalidated_error = RuntimeError(
                    "Akasha source deletion returned a different interaction"
                )
            raise RuntimeError("interaction deletion identity mismatch")

        # 2. 递增 source 代际，并清除所有基于旧图节点生成的 pending ticket。
        with self._lock:
            self._source_generation += 1
            if self._pending:
                self._pending.clear()
                self._pending_changed.notify_all()

        # 3. 以 canonical source 全量替换 turn 派生状态。
        try:
            self._runtime.rebuild_from_source()
        except Exception as exc:
            with self._lock:
                self._source_invalidated_error = RuntimeError(
                    "Akasha derived state is stale after interaction deletion"
                )
            raise RuntimeError(
                "Akasha failed to reconcile interaction deletion: "
                f"{deletion.control_turn_id}"
            ) from exc
        with self._lock:
            self._source_invalidated_error = None
        return deletion

    async def aclose(self) -> None:
        """Drain a staged graph publication before closing owned resources."""

        await self._wait_for_publication()

    async def _wait_for_publication(self) -> None:
        """Wait for the current graph snapshot and preserve its failure."""

        task = self._publish_task
        if task is None:
            return
        await asyncio.shield(task)
        if self._publish_task is task:
            self._publish_task = None

    def _publish_staged(self, staged: StagedOnlineCommit) -> None:
        """Publish one staged suffix under the shared runtime lock."""

        with self._lock:
            self._runtime.publish_staged(staged)

    def _require_valid_source(self) -> None:
        if self._source_invalidated_error is not None:
            raise self._source_invalidated_error

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
            if turn.node_id in self._runtime.cycle.inhibited_nodes:
                continue
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
    user_ids: tuple[str, ...],
    assistant_id: str,
) -> list[dict[str, object]]:
    connection = sqlite3.connect(
        f"file:{sessions_path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        message_ids = (*user_ids, assistant_id)
        placeholders = ",".join("?" for _ in message_ids)
        rows = connection.execute(
            f"""
            SELECT id, session_key, seq, role, content, ts
            FROM messages
            WHERE id IN ({placeholders})
            ORDER BY seq
            """,
            message_ids,
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != len(user_ids) + 1:
        raise ValueError("TurnCommitted messages do not exist")
    if (
        tuple(str(row["id"]) for row in rows[:-1]) != user_ids
        or any(row["role"] != "user" for row in rows[:-1])
        or rows[-1]["id"] != assistant_id
        or rows[-1]["role"] != "assistant"
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


def _clean_feedback_message_ids(message_ids: list[str]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for raw in message_ids:
        message_id = str(raw).strip()
        if message_id and message_id not in seen:
            seen.add(message_id)
            clean.append(message_id)
    return clean


def _feedback_request_error(
    *,
    turn_id: str,
    action: str,
    message_ids: list[str],
    reason: str,
) -> dict[str, object] | None:
    if not turn_id:
        return {"status": "not_staged", "error": "no_active_turn"}
    if action not in {"remember", "forget"}:
        return {"status": "not_staged", "error": "invalid_action"}
    if not message_ids:
        return {"status": "not_staged", "error": "message_ids_required"}
    if len(message_ids) > 20:
        return {
            "status": "not_staged",
            "error": "too_many_message_ids",
            "maximum": 20,
        }
    if action == "forget" and "current_user_message" in message_ids:
        return {
            "status": "not_staged",
            "error": "cannot_forget_current_user_message",
        }
    if len(reason) > 500:
        return {
            "status": "not_staged",
            "error": "reason_too_long",
            "maximum": 500,
        }
    return None


def _missing_feedback_messages(
    message_ids: list[str],
) -> dict[str, object]:
    return {
        "status": "not_staged",
        "error": "messages_not_in_akasha",
        "missing_message_ids": message_ids,
        "hint": (
            "请重新 fetch_messages，并只选择完整 "
            "user/assistant 回合中的消息 ID。"
        ),
    }


def _merge_feedback_markers(
    previous: AkashaFeedbackMarker,
    current: AkashaFeedbackMarker,
) -> AkashaFeedbackMarker:
    if previous.action != current.action:
        raise ValueError("feedback marker actions must match")
    return AkashaFeedbackMarker(
        action=current.action,
        target_message_ids=tuple(
            dict.fromkeys(
                [
                    *previous.target_message_ids,
                    *current.target_message_ids,
                ]
            )
        ),
        target_turn_ids=tuple(
            dict.fromkeys(
                [
                    *previous.target_turn_ids,
                    *current.target_turn_ids,
                ]
            )
        ),
        reason=current.reason or previous.reason,
    )


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
