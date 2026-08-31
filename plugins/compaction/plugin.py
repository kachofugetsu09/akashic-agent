from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any, cast
from pydantic import BaseModel, ConfigDict, field_validator

from agent.control.replay_format import split_replay_batches
from plugins.compaction.engine import (
    CommittedContextUnit,
    ContextCompactionError,
    ContextCompactor,
    ContextPayloadSegments,
    compaction_scope_id,
    window_initial_context_units,
)
from agent.plugin_composition import (
    CONTEXT_PROJECTION_COMMITTED,
    CONTEXT_PROJECTION_FACTS,
    PROVIDER_REQUEST_PROJECTION,
    SESSION_COMPACTION_STORAGE,
    BoundChatModel,
    Context,
    ContextProjectionCommitted,
    ContextProjectionFact,
    PreparedProviderRequest,
    ProviderRequestBinding,
    ProviderTurnInput,
    ModelRequest,
)
from core.error_context import current_provider_attempt, current_provider_operation
from agent.prompting import is_context_frame
from plugins.compaction.receipts import SqliteCompactionReceipts
from plugins.compaction.runtime import (
    CompactionProjection,
    SessionCompactionRuntime,
    validate_committed_receipt,
)

api_version = 3
name = "compaction"
version = "3.0.0"
desc = "Project complete provider requests into an immutable Session ledger"
author = "Akashic Core"
inject = (SESSION_COMPACTION_STORAGE,)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ("memory/consolidation_writes.db",)

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Compaction-owned request history policy."""

    model_config = ConfigDict(extra="forbid")
    keep_recent_tokens: int = 20_000

    @field_validator("keep_recent_tokens", mode="before")
    @classmethod
    def validate_keep_recent_tokens(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("keep_recent_tokens 必须是正整数")
        return value


ConfigModel = Config


@dataclass(slots=True)
class _DetachedSession:
    """Detached Session view that cannot mutate the SessionManager cache."""

    key: str
    created_at: str
    units: tuple[CommittedContextUnit, ...]
    last_consolidated: int = 0

    def history_units(self, *, after_seq: int) -> list[CommittedContextUnit]:
        return [
            unit for unit in self.units if unit.consolidated_through_seq > after_seq
        ]


class _CompactionTurn:
    """Keep projection state private until Core binds the rendered request."""

    def __init__(
        self,
        ctx: Context,
        runtime: SessionCompactionRuntime,
        session: _DetachedSession,
        projection: CompactionProjection,
        keep_recent_tokens: int | None,
        facts: _DurableFacts | None = None,
        access_grant: object | None = None,
    ) -> None:
        self._ctx = ctx
        self._runtime = runtime
        self._session = session
        self._projection = projection
        self._keep_recent_tokens = keep_recent_tokens
        self._facts = facts
        self._access_grant = access_grant

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            [*self._projection.segments.prefix]
            + [
                dict(message)
                for unit in self._projection.segments.committed_units
                for message in unit.messages
            ]
        )

    def bind(self, binding: ProviderRequestBinding) -> _CompactionGate:
        initial_messages = binding.initial_messages
        projection = self._projection
        full_expected_history = list(self.history)
        if (
            not initial_messages
            or initial_messages[1 : 1 + binding.history_count] != full_expected_history
        ):
            raise ContextCompactionError(
                "session projection 与 prompt render history 不一致"
            )
        committed_units = projection.segments.committed_units
        history_count = binding.history_count
        agent_model = cast(BoundChatModel, binding.agent_model)
        if projection.active is None and projection.head.next_generation == 1:
            original_units = committed_units
            committed_units = window_initial_context_units(agent_model, committed_units)
            if len(committed_units) != len(original_units):
                logger.info(
                    "[上下文窗口] generation-0 首次截断 session=%s units=%d→%d",
                    self._session.key,
                    len(original_units),
                    len(committed_units),
                )
            windowed_history = [
                *projection.segments.prefix,
                *[message for unit in committed_units for message in unit.messages],
            ]
            initial_messages[1 : 1 + history_count] = windowed_history
            history_count = len(windowed_history)
        replay_start = 1 + history_count
        replay_end = replay_start + len(binding.attempt_replay)
        replay_batches: list[list[dict[str, Any]]] = []
        replay_tail: list[dict[str, Any]] = []
        if binding.attempt_replay:
            replay_slice = initial_messages[replay_start:replay_end]
            if replay_slice != binding.attempt_replay:
                raise RuntimeError("control attempt replay 未出现在完整 prompt history")
            replay_batches, replay_tail = split_replay_batches(binding.attempt_replay)
        if len(replay_batches) != binding.prior_tool_groups:
            raise RuntimeError(
                "control attempt replay 与 prior tool chain 数量不一致: "
                f"replay={len(replay_batches)} tool_chain={binding.prior_tool_groups}"
            )
        current_pending = [*replay_tail, *initial_messages[replay_end:]]
        interaction_inputs = [
            message.get("content")
            for message in [*binding.attempt_replay, *initial_messages[replay_end:]]
            if message.get("role") == "user"
            and not (
                isinstance(message.get("content"), str)
                and is_context_frame(cast(str, message["content"]))
            )
        ]
        segments = ContextPayloadSegments(
            prefix=(initial_messages[0], *projection.segments.prefix),
            committed_units=committed_units,
            current_anchor=(),
            active_batches=tuple(tuple(batch) for batch in replay_batches),
            pending=tuple(current_pending),
        )
        compactor = ContextCompactor(
            provider=agent_model,
            scope_id=compaction_scope_id(
                self._session.key,
                self._session.created_at,
            ),
            active_compaction=projection.active,
            current_query=json.dumps(
                {"logical_interaction_inputs": interaction_inputs},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            payload_segments=segments,
            max_output_tokens=binding.max_output_tokens,
            keep_recent_tokens=self._keep_recent_tokens or 20_000,
            ledger_parent_generation=projection.head.parent_generation,
            next_generation=projection.head.next_generation,
            fallback_provider=binding.fallback_model,
            chat_call=_call_compaction_summary,
        )
        return _CompactionGate(
            self._ctx,
            self._runtime,
            self._session,
            projection.head,
            compactor,
            channel=binding.channel,
            chat_id=binding.chat_id,
            facts=self._facts,
            access_grant=self._access_grant,
        )


class _CompactionGate:
    """Own compaction policy, checkpoint commits, and durable fact publication."""

    def __init__(
        self,
        ctx: Context,
        runtime: SessionCompactionRuntime,
        session: _DetachedSession,
        head: Any,
        compactor: ContextCompactor,
        *,
        channel: str,
        chat_id: str,
        facts: _DurableFacts | None,
        access_grant: object | None,
    ) -> None:
        self._ctx = ctx
        self._runtime = runtime
        self._session = session
        self._head = head
        self._compactor = cast(Any, compactor)
        self._channel = channel
        self._chat_id = chat_id
        self._facts = facts
        self._access_grant = access_grant
        self._pending_facts: list[ContextProjectionFact] = []

    @property
    def pending_start(self) -> int:
        return self._compactor.pending_start

    async def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int | None,
        trigger: str,
        force: bool,
    ) -> PreparedProviderRequest:
        self._compactor.set_pending(messages)
        prepared = await self._compactor.prepare(
            messages,
            pending_start=self._compactor.pending_start,
            tools=tools,
            trigger=cast(Any, trigger),
            force=force,
            max_output_tokens=max_output_tokens,
        )
        checkpoint = prepared.checkpoint
        if prepared.compacted and checkpoint is not None and checkpoint.committable:
            suppressed = self._runtime.checkpoint_suppresses_post_commit(
                self._session.key,
                checkpoint,
            )
            row = await self._runtime.commit_checkpoint(
                self._session,
                checkpoint,
                head=self._head,
                scope_channel=self._channel,
                scope_chat_id=self._chat_id,
            )
            self._head = replace(
                self._head,
                parent_generation=row.generation,
                next_generation=row.generation + 1,
            )
            self._compactor.acknowledge_committed_checkpoint(row.generation)
            if not suppressed:
                if self._facts is not None and self._access_grant is not None:
                    fact = self._facts.get_committed(
                        self._access_grant,
                        session_key=self._session.key,
                        source_ref=checkpoint.source_ref,
                    )
                    if fact is None:
                        raise RuntimeError("committed projection 缺少 durable fact")
                    self._pending_facts.append(fact)
        usages = (prepared.summary_usage,) if prepared.summary_usage is not None else ()
        return PreparedProviderRequest(
            pending_start=prepared.pending_start,
            estimated_tokens=prepared.estimated_tokens,
            token_quality=prepared.estimate_quality,
            changed=prepared.compacted,
            auxiliary_usages=usages,
        )

    def can_retry_context_error(self, *, context_window: int) -> bool:
        return context_window > 0

    def record_completed_batch(
        self,
        messages: list[dict[str, Any]],
        *,
        batch_start: int,
    ) -> None:
        self._compactor.record_completed_batch(messages, batch_start=batch_start)

    async def record_response(self, **kwargs: Any) -> None:
        self._compactor.record_response(**kwargs)
        if self._access_grant is None:
            return
        pending, self._pending_facts = self._pending_facts, []
        for fact in pending:
            await self._ctx.observe(
                CONTEXT_PROJECTION_COMMITTED,
                _notice(fact, self._access_grant),
            )


class _PublishedProjection:
    """Open request-local projections from immutable Session inputs."""

    def __init__(
        self,
        ctx: Context,
        storage: Any,
        receipts: SqliteCompactionReceipts,
        *,
        keep_recent_tokens: int | None = None,
    ) -> None:
        self._ctx = ctx
        self._storage = storage
        self._receipts = receipts
        self._keep_recent_tokens = keep_recent_tokens

    async def open_turn(self, input: ProviderTurnInput) -> _CompactionTurn:
        scoped_storage = self._storage.scope(input.access_grant)
        runtime = SessionCompactionRuntime(
            storage=scoped_storage,
            receipts=self._receipts,
        )
        facts = _DurableFacts(self._storage, self._receipts)
        units = tuple(
            CommittedContextUnit(
                source_from_seq=unit.source_from_seq,
                consolidated_through_seq=unit.consolidated_through_seq,
                source_message_ids=unit.source_message_ids,
                messages=unit.messages(),
                message_refs=unit.message_refs,
            )
            for unit in input.history_units
        )
        session = _DetachedSession(
            key=input.session_key,
            created_at=input.session_created_at,
            units=units,
        )
        projection = await runtime.projection(
            session,
            prefix=[],
            current_anchor=[],
            pending=[],
        )
        for fact in facts.list_committed(
            input.access_grant,
            session_key=input.session_key,
        ):
            await self._ctx.observe(
                CONTEXT_PROJECTION_COMMITTED,
                _notice(fact, input.access_grant),
            )
        return _CompactionTurn(
            self._ctx,
            runtime,
            session,
            projection,
            self._keep_recent_tokens,
            facts,
            input.access_grant,
        )


class _DurableFacts:
    """Rebuild committed notifications from receipts and the Session ledger."""

    def __init__(self, storage: Any, receipts: SqliteCompactionReceipts) -> None:
        self._storage = storage
        self._receipts = receipts

    def list_committed(
        self,
        access_grant: object,
        *,
        session_key: str,
    ) -> tuple[ContextProjectionFact, ...]:
        storage = self._storage.scope(access_grant)
        facts: list[ContextProjectionFact] = []
        for receipt in self._receipts.list_all():
            receipt_session_key = receipt.get("session_key")
            source_ref = receipt.get("source_ref")
            if not isinstance(receipt_session_key, str) or not isinstance(source_ref, str):
                raise ValueError("compaction receipt 缺少 durable fact identity")
            if receipt_session_key != session_key:
                continue
            row = storage.get_committed(session_key, source_ref)
            if row is None or row.invalidated_at is not None:
                continue
            _ = validate_committed_receipt(receipt, row)
            # v3 belonged to the retired PENDING/optimizer pipeline. Publishing it
            # here would repeat historical Markdown work after an upgrade.
            if receipt.get("version") == 3:
                continue
            facts.append(
                ContextProjectionFact(
                    session_key=session_key,
                    generation=row.generation,
                    source_ref=source_ref,
                    checkpoint_json=json.dumps(
                        receipt,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    scope_channel=str(receipt.get("scope_channel", "")),
                    scope_chat_id=str(receipt.get("scope_chat_id", "")),
                )
            )
        return tuple(sorted(facts, key=lambda fact: fact.generation))

    def get_committed(
        self,
        access_grant: object,
        *,
        session_key: str,
        source_ref: str,
    ) -> ContextProjectionFact | None:
        return next(
            (
                fact
                for fact in self.list_committed(
                    access_grant,
                    session_key=session_key,
                )
                if fact.session_key == session_key and fact.source_ref == source_ref
            ),
            None,
        )


def _notice(
    fact: ContextProjectionFact,
    access_grant: object,
) -> ContextProjectionCommitted:
    return ContextProjectionCommitted(
        session_key=fact.session_key,
        generation=fact.generation,
        source_ref=fact.source_ref,
        scope_channel=fact.scope_channel,
        scope_chat_id=fact.scope_chat_id,
        suppress_post_commit=False,
        access_grant=access_grant,
    )


async def apply(ctx: Context, config: object) -> None:
    """Provide optional request projection and recoverable committed facts."""

    if not isinstance(config, Config):
        raise TypeError("Compaction config 必须通过 ConfigModel 校验")
    storage = ctx.require(SESSION_COMPACTION_STORAGE)
    receipts = SqliteCompactionReceipts(
        ctx.workspace_file("memory/consolidation_writes.db")
    )
    facts = _DurableFacts(storage, receipts)
    _ = await ctx.provide(
        PROVIDER_REQUEST_PROJECTION,
        _PublishedProjection(
            ctx,
            storage,
            receipts,
            keep_recent_tokens=config.keep_recent_tokens,
        ),
    )
    _ = await ctx.provide(
        CONTEXT_PROJECTION_FACTS,
        facts,
    )


async def _call_compaction_summary(
    *,
    provider: BoundChatModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    disable_thinking: bool = True,
):
    """Call the plugin's auxiliary model operation outside the business gate."""

    operation_token = current_provider_operation.set("compaction_summary")
    attempt_token = current_provider_attempt.set(0)
    try:
        return await provider.complete(
            ModelRequest(
                messages=messages,
                tools=tools,
                max_output_tokens=max_tokens,
                disable_reasoning=disable_thinking,
            )
        )
    finally:
        current_provider_attempt.reset(attempt_token)
        current_provider_operation.reset(operation_token)
