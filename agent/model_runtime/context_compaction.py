from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

from agent.model_runtime.execution_history import active_shell_execution_origins
from agent.model_runtime.types import ModelUsage
from agent.prompting import is_context_frame

if TYPE_CHECKING:
    from agent.provider import LLMProvider


SUMMARY_FORMAT_VERSION = 1
SUMMARY_HEADINGS = (
    "## Goal",
    "## Constraints & Preferences",
    "## Progress",
    "### Done",
    "### In Progress",
    "### Blocked",
    "## Key Decisions",
    "## Next Steps",
    "## Critical Context",
)
SUMMARY_MAX_TOKENS = 8192
KEEP_RECENT_TOKENS = 20_000
_SUMMARY_PROMPT = """更新当前长任务的上下文压缩摘要。

摘要只替代已经完成的旧 session 历史，完整 messages 和 tool results 仍由 SessionDB 保留。只记录输入中已经出现的事实，不补充猜测，不把计划写成已完成。

必须严格使用以下标题，不得增加标题：
## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
## Critical Context

保留文件路径、符号、命令、错误、数值、外部效果和验证结果。若输入中存在仍在运行的 shell execution，必须保留 execution_id、命令和当前状态。省略重复探索、无用日志、tool_call_id 和其他协议细节。只输出摘要正文。"""

CompactionTrigger = Literal["soft_limit", "context_overflow"]


class ContextCompactionError(RuntimeError):
    """当前完整 payload 无法在模型输入边界内继续执行。"""


@dataclass(frozen=True)
class CommittedContextUnit:
    """One immutable, complete logical interaction from SessionDB."""

    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    messages: tuple[dict[str, Any], ...]
    message_refs: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.source_from_seq < 0 or self.consolidated_through_seq < self.source_from_seq:
            raise ValueError("context unit seq boundary 无效")
        if not self.source_message_ids:
            raise ValueError("context unit 必须包含 source message ids")
        if not self.messages:
            raise ValueError("context unit 必须包含 model messages")
        if self.message_refs and len(self.message_refs) != len(self.messages):
            raise ValueError("context unit message_refs 必须与 messages 等长")


@dataclass(frozen=True)
class ContextPayloadSegments:
    """Assembler-owned payload order; no message-equality inference is allowed."""

    prefix: tuple[dict[str, Any], ...]
    committed_units: tuple[CommittedContextUnit, ...]
    current_anchor: tuple[dict[str, Any], ...]
    temporary_summary: tuple[dict[str, Any], ...] = ()
    active_batches: tuple[tuple[dict[str, Any], ...], ...] = ()
    pending: tuple[dict[str, Any], ...] = ()

    def flatten(self) -> list[dict[str, Any]]:
        return [
            *[_copy_message(message) for message in self.prefix],
            *[
                _copy_message(message)
                for unit in self.committed_units
                for message in unit.messages
            ],
            *[_copy_message(message) for message in self.current_anchor],
            *[_copy_message(message) for message in self.temporary_summary],
            *[
                _copy_message(message)
                for batch in self.active_batches
                for message in batch
            ],
            *[_copy_message(message) for message in self.pending],
        ]


@dataclass(frozen=True)
class ActiveCompaction:
    """The active session ledger row projected into the next model payload."""

    generation: int
    summary: str
    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    retained_tail: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ContextCompaction:
    """One temporary projection and its committed source provenance."""

    summary: str
    generation: int
    parent_generation: int
    trigger: CompactionTrigger
    context_window: int
    soft_limit_tokens: int
    hard_input_tokens: int
    keep_recent_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    retained_tail: tuple[dict[str, Any], ...]
    summary_usage: ModelUsage | None
    source_ref: str = ""

    @property
    def committable(self) -> bool:
        return bool(self.source_message_ids)

    def to_payload(self) -> dict[str, object]:
        """Return a bounded diagnostic payload; SessionStore remains persistence owner."""

        return {
            "summary_format_version": SUMMARY_FORMAT_VERSION,
            "summary": self.summary,
            "generation": self.generation,
            "parent_generation": self.parent_generation,
            "trigger": self.trigger,
            "context_window": self.context_window,
            "threshold_tokens": self.soft_limit_tokens,
            "hard_input_tokens": self.hard_input_tokens,
            "keep_recent_tokens": self.keep_recent_tokens,
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "source_from_seq": self.source_from_seq,
            "consolidated_through_seq": self.consolidated_through_seq,
            "source_message_ids": list(self.source_message_ids),
            "retained_tail": [dict(message) for message in self.retained_tail],
            "summary_usage": _usage_payload(self.summary_usage),
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class PreparedQueryContext:
    pending_start: int
    estimated_tokens: int
    estimate_quality: Literal["approximate", "exact_plus_delta"]
    compacted: bool
    summary_usage: ModelUsage | None
    checkpoint: ContextCompaction | None = None


class _ContextTokenMeter:
    def __init__(self) -> None:
        self._exact_input_tokens: int | None = None
        self._message_count = 0
        self._tool_digest = ""

    def estimate(
        self,
        provider: "LLMProvider",
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[int, Literal["approximate", "exact_plus_delta"]]:
        digest = _tool_schema_digest(tools)
        if (
            self._exact_input_tokens is not None
            and digest == self._tool_digest
            and len(messages) >= self._message_count
        ):
            delta = provider.estimate_appended_message_tokens(
                messages[self._message_count :]
            )
            return self._exact_input_tokens + delta, "exact_plus_delta"
        return provider.estimate_context_tokens(messages, tools), "approximate"

    def record_response(
        self,
        *,
        message_count: int,
        tools: list[dict],
        usage: ModelUsage | None,
    ) -> None:
        if usage is None or usage.input_tokens is None:
            self.invalidate()
            return
        self._exact_input_tokens = usage.input_tokens
        self._message_count = message_count
        self._tool_digest = _tool_schema_digest(tools)

    def invalidate(self) -> None:
        self._exact_input_tokens = None
        self._message_count = 0
        self._tool_digest = ""


class ContextCompactor:
    """Compact committed session units into a temporary payload and checkpoint."""

    def __init__(
        self,
        *,
        provider: "LLMProvider",
        model: str,
        scope_id: str,
        active_compaction: ActiveCompaction | None = None,
        current_query: object | None = None,
        payload_segments: ContextPayloadSegments,
        max_output_tokens: int = 0,
        trigger_percent: float = 0.74,
        keep_recent_tokens: int = KEEP_RECENT_TOKENS,
        ledger_parent_generation: int | None = None,
        next_generation: int | None = None,
        fallback_provider: "LLMProvider | None" = None,
        fallback_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._scope_id = str(scope_id).strip()
        if not self._scope_id:
            raise ValueError("scope_id 不能为空")
        self._max_output_tokens = _validate_output_budget(provider, max_output_tokens)
        self._trigger_percent = _validate_trigger_percent(trigger_percent)
        self._keep_recent_tokens = _validate_keep_recent_tokens(keep_recent_tokens)
        if ledger_parent_generation is not None and (
            not isinstance(ledger_parent_generation, int)
            or isinstance(ledger_parent_generation, bool)
            or ledger_parent_generation < 0
        ):
            raise ValueError("ledger_parent_generation 必须是非负整数")
        if next_generation is not None and (
            not isinstance(next_generation, int)
            or isinstance(next_generation, bool)
            or next_generation <= 0
        ):
            raise ValueError("next_generation 必须是正整数")
        if (
            ledger_parent_generation is not None
            and next_generation is not None
            and next_generation <= ledger_parent_generation
        ):
            raise ValueError("next_generation 必须大于 ledger_parent_generation")
        self._ledger_parent_generation = ledger_parent_generation
        self._next_generation = next_generation
        self._fallback_provider = fallback_provider
        self._fallback_model = str(fallback_model or "").strip()
        self._segments = _copy_segments(payload_segments)
        self._committed_units = list(self._segments.committed_units)
        self._active_compaction = active_compaction
        self._completed_batches = [
            tuple(_copy_message(message) for message in batch)
            for batch in self._segments.active_batches
        ]
        self._current_query = (
            _bounded_text(current_query, 8000)
            if current_query is not None
            else _find_current_query(self._segments.current_anchor)
        )
        self._persistent_summary = active_compaction.summary if active_compaction else ""
        self._temporary_summary = ""
        self._committed_checkpoint: ContextCompaction | None = None
        self._compaction: ContextCompaction | None = None
        self._meter = _ContextTokenMeter()

    @property
    def pending_start(self) -> int:
        return _pending_start(self._segments)

    @property
    def compaction(self) -> ContextCompaction | None:
        return self._compaction

    @property
    def has_compactable_prefix(self) -> bool:
        return bool(self._candidate_units())

    def checkpoint_payload(self) -> dict[str, object] | None:
        return self._compaction.to_payload() if self._compaction is not None else None

    @property
    def committed_checkpoint(self) -> ContextCompaction | None:
        """Return only the checkpoint whose source IDs are committed SessionDB facts."""

        return self._committed_checkpoint

    def record_response(
        self,
        *,
        message_count: int,
        tools: list[dict],
        usage: ModelUsage | None,
    ) -> None:
        self._meter.record_response(
            message_count=message_count,
            tools=tools,
            usage=usage,
        )

    def record_completed_batch(self, messages: list[dict], *, batch_start: int) -> None:
        """Retain a closed active-turn batch for the next temporary compaction."""

        if batch_start < 0 or batch_start >= len(messages):
            raise ValueError("完整工具批次缺少可记录消息")
        batch = tuple(_copy_message(message) for message in messages[batch_start:])
        if not _is_closed_tool_batch(batch):
            raise ValueError("工具批次必须在 assistant tool call 与全部 result 闭合后记录")
        self._completed_batches.append(batch)
        self._segments = ContextPayloadSegments(
            prefix=self._segments.prefix,
            committed_units=self._segments.committed_units,
            current_anchor=self._segments.current_anchor,
            temporary_summary=self._segments.temporary_summary,
            active_batches=tuple(self._completed_batches),
            pending=self._segments.pending,
        )

    async def prepare(
        self,
        messages: list[dict],
        *,
        pending_start: int,
        tools: list[dict],
        trigger: CompactionTrigger = "soft_limit",
        force: bool = False,
    ) -> PreparedQueryContext:
        """Run the one payload gate before the next provider request."""

        # 1. Estimate the assembled payload and trigger on either boundary.
        if messages != self._segments.flatten():
            raise ContextCompactionError("context_compaction_payload_segments_mismatch")
        expected_pending_start = _pending_start(self._segments)
        if pending_start != expected_pending_start:
            raise ValueError(
                "pending_start 与 assembler payload segments 不一致: "
                f"expected={expected_pending_start} actual={pending_start}"
            )
        estimated, quality = self._meter.estimate(self._provider, messages, tools)
        soft_limit = math.floor(self._provider.context_window * self._trigger_percent)
        hard_limit = hard_input_limit(self._provider, self._max_output_tokens)
        boundary_hit = estimated >= soft_limit or estimated >= hard_limit
        if not force and not boundary_hit:
            return PreparedQueryContext(pending_start, estimated, quality, False, None)

        # 2. First compact committed units; never mix active-turn evidence into that row.
        candidates = self._candidate_units()
        if not candidates:
            raise ContextCompactionError("context_compaction_no_closed_prefix")
        committed_candidates = [
            unit for unit in candidates if _is_persistable_unit(unit)
        ]
        active_candidates = [
            unit for unit in candidates if not _is_persistable_unit(unit)
        ]
        prefix, current_anchor, previous_temp, pending = self._split_segments()
        retained_committed = committed_candidates
        retained_active = active_candidates
        current_prefix = [_copy_message(message) for message in prefix]
        temporary_summary = [
            _copy_message(message) for message in previous_temp
        ]
        committed_checkpoint: ContextCompaction | None = None
        summary_usage: ModelUsage | None = None
        if committed_candidates:
            try:
                selected_committed, retained_committed = self._select_units(
                    committed_candidates
                )
            except ContextCompactionError:
                selected_committed = []
            if selected_committed:
                summary, summary_usage = await self._summarize(
                    selected_committed,
                    include_temporary=False,
                )
                current_prefix = [
                    *_without_previous_compaction(current_prefix),
                    *build_compaction_messages(
                        summary,
                        generation=self._ledger_generation(),
                        source_ref=_compaction_source_ref(
                            self._scope_id,
                            self._ledger_generation(),
                        ),
                    ),
                ]
                self._persistent_summary = summary
                committed_checkpoint = _build_checkpoint(
                    summary=summary,
                    generation=self._ledger_generation(),
                    parent_generation=self._ledger_parent(),
                    trigger=trigger,
                    context_window=self._provider.context_window,
                    soft_limit_tokens=soft_limit,
                    hard_input_tokens=hard_limit,
                    keep_recent_tokens=self._keep_recent_tokens,
                    estimated_tokens_before=estimated,
                    estimated_tokens_after=0,
                    selected=selected_committed,
                    retained=retained_committed,
                    summary_usage=summary_usage,
                    source_ref=_compaction_source_ref(
                        self._scope_id,
                        self._ledger_generation(),
                    ),
                )
        rebuilt = _flatten_projection(
            prefix=current_prefix,
            committed=retained_committed,
            current_anchor=current_anchor,
            temporary_summary=temporary_summary,
            active=retained_active,
            pending=pending,
        )
        after = self._provider.estimate_context_tokens(rebuilt, tools)

        # 3. If the committed checkpoint is not enough, compact only closed active batches.
        temporary_checkpoint: ContextCompaction | None = None
        if (
            after >= soft_limit
            or after >= hard_limit
            or (force and committed_checkpoint is None and bool(active_candidates))
        ):
            if not active_candidates:
                raise ContextCompactionError(
                    "context_compaction_insufficient "
                    f"estimated={after} soft_limit={soft_limit} hard_limit={hard_limit}"
                )
            selected_active, retained_active = self._select_units(active_candidates)
            active_summary, active_usage = await self._summarize(
                selected_active,
                include_temporary=True,
            )
            temporary_summary = build_compaction_messages(
                active_summary,
                generation=0,
                source_ref=_temporary_source_ref(self._scope_id),
            )
            self._temporary_summary = active_summary
            summary_usage = active_usage
            rebuilt = _flatten_projection(
                prefix=current_prefix,
                committed=retained_committed,
                current_anchor=current_anchor,
                temporary_summary=temporary_summary,
                active=retained_active,
                pending=pending,
            )
            after = self._provider.estimate_context_tokens(rebuilt, tools)
            temporary_checkpoint = _build_checkpoint(
                summary=active_summary,
                generation=0,
                parent_generation=self._ledger_parent(),
                trigger=trigger,
                context_window=self._provider.context_window,
                soft_limit_tokens=soft_limit,
                hard_input_tokens=hard_limit,
                keep_recent_tokens=self._keep_recent_tokens,
                estimated_tokens_before=estimated,
                estimated_tokens_after=after,
                selected=selected_active,
                retained=retained_active,
                summary_usage=active_usage,
                source_ref=_temporary_source_ref(self._scope_id),
            )
        if after >= soft_limit or after >= hard_limit:
            raise ContextCompactionError(
                "context_compaction_insufficient "
                f"estimated={after} soft_limit={soft_limit} hard_limit={hard_limit}"
            )

        # 4. Commit only the source-backed row; the active projection remains ephemeral.
        if committed_checkpoint is not None:
            committed_checkpoint = _replace_checkpoint_after_projection(
                committed_checkpoint,
                estimated_tokens_after=after,
            )
            self._committed_checkpoint = committed_checkpoint
            self._active_compaction = ActiveCompaction(
                generation=committed_checkpoint.generation,
                summary=committed_checkpoint.summary,
                source_from_seq=committed_checkpoint.source_from_seq,
                consolidated_through_seq=committed_checkpoint.consolidated_through_seq,
                source_message_ids=committed_checkpoint.source_message_ids,
                retained_tail=committed_checkpoint.retained_tail,
            )
        checkpoint = committed_checkpoint or temporary_checkpoint
        if checkpoint is None:
            raise ContextCompactionError("context_compaction_no_closed_prefix")
        retained_active_batches = [tuple(unit.messages) for unit in retained_active]
        self._segments = ContextPayloadSegments(
            prefix=tuple(current_prefix),
            committed_units=tuple(retained_committed),
            current_anchor=tuple(current_anchor),
            temporary_summary=tuple(temporary_summary),
            active_batches=tuple(retained_active_batches),
            pending=tuple(pending),
        )
        self._committed_units = list(retained_committed)
        self._completed_batches = retained_active_batches
        self._compaction = checkpoint
        messages[:] = rebuilt
        self._meter.invalidate()
        return PreparedQueryContext(
            _pending_start(self._segments),
            after,
            "approximate",
            True,
            summary_usage,
            checkpoint,
        )

    def _ledger_parent(self) -> int:
        if self._ledger_parent_generation is not None:
            return self._ledger_parent_generation
        return self._active_compaction.generation if self._active_compaction else 0

    def _ledger_generation(self) -> int:
        if self._next_generation is None:
            raise ContextCompactionError("context_compaction_ledger_head_required")
        return self._next_generation

    def _candidate_units(self) -> list[CommittedContextUnit]:
        units: list[CommittedContextUnit] = []
        for unit in self._committed_units:
            units.extend(_split_closed_batches(unit))
        for index, batch in enumerate(self._completed_batches):
            units.append(
                CommittedContextUnit(
                    source_from_seq=0,
                    consolidated_through_seq=0,
                    source_message_ids=(f"active:{index}",),
                    messages=tuple(batch),
                )
            )
        return units

    def _select_units(
        self,
        candidates: list[CommittedContextUnit],
    ) -> tuple[list[CommittedContextUnit], list[CommittedContextUnit]]:
        if len(candidates) < 2:
            raise ContextCompactionError("context_compaction_no_closed_prefix")
        kept: list[CommittedContextUnit] = []
        kept_tokens = 0
        for unit in reversed(candidates):
            tokens = self._provider.estimate_appended_message_tokens(list(unit.messages))
            kept.insert(0, unit)
            kept_tokens += tokens
            if kept_tokens >= self._keep_recent_tokens:
                break
        cut = max(1, len(candidates) - len(kept))
        cut = min(cut, len(candidates) - 1)
        selected = candidates[:cut]
        retained = candidates[cut:]
        active_index = _active_execution_unit_index(candidates)
        if active_index is not None and cut > active_index:
            cut = active_index
            selected = candidates[:cut]
            retained = candidates[cut:]
        if not selected:
            raise ContextCompactionError("context_compaction_no_closed_prefix")
        return selected, retained

    def _split_segments(
        self,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """Return prefix, current anchor, pending, and active sections in owner order."""

        return (
            [_copy_message(message) for message in self._segments.prefix],
            [_copy_message(message) for message in self._segments.current_anchor],
            [_copy_message(message) for message in self._segments.temporary_summary],
            [_copy_message(message) for message in self._segments.pending],
        )

    async def _summarize(
        self,
        selected: Sequence[CommittedContextUnit],
        *,
        include_temporary: bool,
    ) -> tuple[str, ModelUsage | None]:
        sections = [_SUMMARY_PROMPT, "\n[Current user query]\n", self._current_query]
        previous_summary = self._persistent_summary
        if include_temporary and self._temporary_summary:
            previous_summary = (
                f"{previous_summary}\n{self._temporary_summary}"
                if previous_summary
                else self._temporary_summary
            )
        if previous_summary:
            sections.extend(
                ["\n[Previous compaction summary]\n", previous_summary]
            )
        sections.append("\n[Closed history to consolidate]\n")
        sections.extend(
            _serialize_message(message)
            for unit in selected
            for message in unit.messages
        )
        providers = [(self._provider, self._model)]
        if self._fallback_provider is not None:
            fallback = (self._fallback_provider, self._fallback_model or self._model)
            if fallback[0] is not self._provider or fallback[1] != self._model:
                providers.append(fallback)
        failures: list[str] = []
        for provider, model in providers:
            try:
                summary_input = [{"role": "user", "content": "".join(sections)}]
                response = await provider.chat(
                    messages=summary_input,
                    tools=[],
                    model=model,
                    max_tokens=_summary_output_limit(provider, summary_input),
                    disable_thinking=True,
                )
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            summary = (response.content or "").strip()
            if response.tool_calls or not _valid_summary(summary):
                failures.append("summary response failed Pi heading validation")
                continue
            return summary, response.usage
        raise ContextCompactionError(
            "context_compaction_summary_failed: " + "; ".join(failures)
        )


def build_compaction_messages(
    summary: str,
    *,
    generation: int,
    source_ref: str,
) -> list[dict[str, Any]]:
    """Build a non-executable temporary context block."""

    return [
        {
            "role": "system",
            "content": (
                "<session-context-compaction>\n"
                f"generation={generation}; source_ref={source_ref}\n"
                f"{summary}\n"
                "</session-context-compaction>"
            ),
        }
    ]


def hard_input_limit(provider: "LLMProvider", max_output_tokens: int) -> int:
    """Return the exact input boundary for this provider request."""

    context_window = int(provider.context_window)
    if context_window <= 0:
        raise ValueError("context_window 必须是正整数")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool):
        raise ValueError("max_output_tokens 必须是整数")
    if max_output_tokens < 0 or max_output_tokens >= context_window:
        raise ValueError("max_output_tokens 必须在 [0, context_window) 内")
    return context_window - max_output_tokens


def _validate_output_budget(provider: "LLMProvider", value: int) -> int:
    hard_input_limit(provider, value)
    return value


def _validate_trigger_percent(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("trigger_percent 必须是数字")
    percent = float(value)
    if not 0 < percent < 1:
        raise ValueError("trigger_percent 必须在 (0, 1) 内")
    return percent


def _validate_keep_recent_tokens(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("keep_recent_tokens 必须是正整数")
    return value


def _summary_output_limit(
    provider: "LLMProvider",
    summary_input: list[dict[str, Any]],
) -> int:
    estimated_input = provider.estimate_context_tokens(summary_input, [])
    available = int(provider.context_window) - estimated_input
    if available <= 0:
        raise ContextCompactionError(
            "context_compaction_summary_input_exceeds_window "
            f"estimated={estimated_input} window={provider.context_window}"
        )
    return max(1, min(SUMMARY_MAX_TOKENS, available))


def _copy_segments(segments: ContextPayloadSegments) -> ContextPayloadSegments:
    return ContextPayloadSegments(
        prefix=tuple(_copy_message(message) for message in segments.prefix),
        committed_units=tuple(
            CommittedContextUnit(
                source_from_seq=unit.source_from_seq,
                consolidated_through_seq=unit.consolidated_through_seq,
                source_message_ids=tuple(unit.source_message_ids),
                messages=tuple(_copy_message(message) for message in unit.messages),
                message_refs=tuple(unit.message_refs),
            )
            for unit in segments.committed_units
        ),
        current_anchor=tuple(_copy_message(message) for message in segments.current_anchor),
        temporary_summary=tuple(
            _copy_message(message) for message in segments.temporary_summary
        ),
        active_batches=tuple(
            tuple(_copy_message(message) for message in batch)
            for batch in segments.active_batches
        ),
        pending=tuple(_copy_message(message) for message in segments.pending),
    )


def _pending_start(segments: ContextPayloadSegments) -> int:
    return len(segments.prefix) + sum(
        len(unit.messages) for unit in segments.committed_units
    ) + len(segments.current_anchor) + len(segments.temporary_summary) + sum(
        len(batch) for batch in segments.active_batches
    )


def _flatten_projection(
    *,
    prefix: Sequence[dict[str, Any]],
    committed: Sequence[CommittedContextUnit],
    current_anchor: Sequence[dict[str, Any]],
    temporary_summary: Sequence[dict[str, Any]],
    active: Sequence[CommittedContextUnit],
    pending: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten the owner-provided order without interleaving active data into ledger source."""

    return [
        *[_copy_message(message) for message in prefix],
        *[
            _strip_opaque_state(message)
            for unit in committed
            for message in unit.messages
        ],
        *[_copy_message(message) for message in current_anchor],
        *[_copy_message(message) for message in temporary_summary],
        *[
            _strip_opaque_state(message)
            for unit in active
            for message in unit.messages
        ],
        *[_copy_message(message) for message in pending],
    ]


def _replace_checkpoint_after_projection(
    checkpoint: ContextCompaction,
    *,
    estimated_tokens_after: int,
) -> ContextCompaction:
    return replace(checkpoint, estimated_tokens_after=estimated_tokens_after)


def _build_checkpoint(
    *,
    summary: str,
    generation: int,
    parent_generation: int,
    trigger: CompactionTrigger,
    context_window: int,
    soft_limit_tokens: int,
    hard_input_tokens: int,
    keep_recent_tokens: int,
    estimated_tokens_before: int,
    estimated_tokens_after: int,
    selected: Sequence[CommittedContextUnit],
    retained: Sequence[CommittedContextUnit],
    summary_usage: ModelUsage | None,
    source_ref: str,
) -> ContextCompaction:
    source_ids = tuple(
        message_id
        for unit in selected
        for message_id in unit.source_message_ids
        if not message_id.startswith("active")
    )
    committed = [unit for unit in selected if _is_persistable_unit(unit)]
    if committed:
        source_from_seq = committed[0].source_from_seq
        through_seq = committed[-1].consolidated_through_seq
    else:
        source_from_seq = 0
        through_seq = 0
    retained_tail = tuple(
        item
        for unit in retained
        for item in _retained_tail_payload(unit)
    )
    return ContextCompaction(
        summary=summary,
        generation=generation,
        parent_generation=parent_generation,
        trigger=trigger,
        context_window=context_window,
        soft_limit_tokens=soft_limit_tokens,
        hard_input_tokens=hard_input_tokens,
        keep_recent_tokens=keep_recent_tokens,
        estimated_tokens_before=estimated_tokens_before,
        estimated_tokens_after=estimated_tokens_after,
        source_from_seq=source_from_seq,
        consolidated_through_seq=through_seq,
        source_message_ids=source_ids,
        retained_tail=retained_tail,
        summary_usage=summary_usage,
        source_ref=source_ref,
    )


def _active_execution_unit_index(units: Sequence[CommittedContextUnit]) -> int | None:
    call_units: dict[str, int] = {}
    messages: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        messages.extend(unit.messages)
        for message in unit.messages:
            if message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    call_units[str(call["id"])] = index
    origins = active_shell_execution_origins(messages)
    active = [call_units[call_id] for call_id in origins.values() if call_id in call_units]
    return min(active) if active else None


def _split_closed_batches(unit: CommittedContextUnit) -> list[CommittedContextUnit]:
    """Split a committed interaction only immediately after closed tool batches."""

    boundaries = _closed_batch_boundaries(unit.messages)
    if not boundaries or boundaries[-1] >= len(unit.messages):
        return [unit]
    refs = list(unit.message_refs)
    if not refs:
        refs = [
            (str(message.get("id")), int(message.get("seq", 0)))
            for message in unit.messages
            if isinstance(message.get("id"), str)
            and isinstance(message.get("seq"), int)
        ]
    parts: list[CommittedContextUnit] = []
    start = 0
    for index, end in enumerate([*boundaries, len(unit.messages)]):
        if end <= start:
            continue
        part_messages = tuple(_copy_message(message) for message in unit.messages[start:end])
        part_refs = tuple(refs[start:end]) if len(refs) == len(unit.messages) else ()
        part_ids = tuple(ref[0] for ref in part_refs if ref[0])
        if not part_ids:
            # A partial cut without per-message provenance is temporary only.
            part_ids = (f"active-split:{unit.source_from_seq}:{index}",)
        seqs = [ref[1] for ref in part_refs]
        parts.append(
            CommittedContextUnit(
                source_from_seq=min(seqs) if seqs else unit.source_from_seq,
                consolidated_through_seq=max(seqs) if seqs else unit.consolidated_through_seq,
                source_message_ids=part_ids,
                messages=part_messages,
                message_refs=part_refs,
            )
        )
        start = end
    return parts


def _is_persistable_unit(unit: CommittedContextUnit) -> bool:
    return bool(unit.source_message_ids) and not any(
        item.startswith("active") for item in unit.source_message_ids
    )


def _closed_batch_boundaries(messages: Sequence[dict[str, Any]]) -> list[int]:
    boundaries: list[int] = []
    cursor = 0
    while cursor < len(messages):
        message = messages[cursor]
        if message.get("role") != "assistant" or not isinstance(
            message.get("tool_calls"), list
        ):
            cursor += 1
            continue
        calls = {
            str(call.get("id"))
            for call in cast(list[object], message["tool_calls"])
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        }
        if not calls:
            cursor += 1
            continue
        results: set[str] = set()
        end = cursor + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            result_id = messages[end].get("tool_call_id")
            if isinstance(result_id, str):
                results.add(result_id)
            end += 1
        if results == calls:
            boundaries.append(end)
            cursor = end
        else:
            cursor += 1
    return boundaries


def _retained_tail_payload(unit: CommittedContextUnit) -> list[dict[str, Any]]:
    refs = list(unit.message_refs)
    payload: list[dict[str, Any]] = []
    for index, message in enumerate(unit.messages):
        item: dict[str, Any] = {"message": _strip_opaque_state(message)}
        if len(refs) == len(unit.messages):
            item["id"], item["seq"] = refs[index]
        elif isinstance(message.get("id"), str):
            item["id"] = message["id"]
            if isinstance(message.get("seq"), int):
                item["seq"] = message["seq"]
        payload.append(item)
    return payload


def _is_closed_tool_batch(messages: Sequence[dict[str, Any]]) -> bool:
    calls: set[str] = set()
    results: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list):
                calls.update(
                    str(call.get("id"))
                    for call in raw_calls
                    if isinstance(call, dict) and isinstance(call.get("id"), str)
                )
        elif message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            results.add(str(message["tool_call_id"]))
    return bool(calls) and calls == results


def _tool_schema_digest(tools: list[dict]) -> str:
    encoded = json.dumps(
        tools,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compaction_source_ref(scope_id: str, generation: int) -> str:
    digest = hashlib.sha256(f"{scope_id}\0{generation}".encode("utf-8")).hexdigest()
    return f"context-compaction:{scope_id}:{generation}:{digest[:16]}"


def _temporary_source_ref(scope_id: str) -> str:
    digest = hashlib.sha256(scope_id.encode("utf-8")).hexdigest()
    return f"context-compaction:temporary:{digest[:16]}"


def _copy_message(message: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(message))


def _strip_opaque_state(message: Mapping[str, Any]) -> dict[str, Any]:
    clean = _copy_message(message)
    clean.pop("model_state", None)
    return clean


def _without_previous_compaction(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if not (
            message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and "<session-context-compaction>" in message["content"]
        )
    ]


def _find_current_query(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and is_context_frame(content):
            continue
        return _bounded_text(content, 8000)
    return "(current user query is retained in the active prompt)"


def _serialize_message(message: Mapping[str, Any]) -> str:
    clean = {
        key: value
        for key, value in message.items()
        if key not in {"model_state", "reasoning_content"}
    }
    clean["content"] = _bounded_text(clean.get("content"), 8000)
    return "\n" + json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def _bounded_text(value: object, limit: int) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value or "")
    if len(text) <= limit:
        return text
    marker = f"\n…{len(text) - limit} chars omitted from compaction input…\n"
    keep = max(0, limit - len(marker))
    head = keep // 2
    return text[:head] + marker + text[-(keep - head) :]


def _valid_summary(summary: str) -> bool:
    headings = [
        line.strip()
        for line in summary.splitlines()
        if line.lstrip().startswith("#")
    ]
    return headings == list(SUMMARY_HEADINGS)


def _usage_payload(usage: ModelUsage | None) -> dict[str, object]:
    if usage is None:
        return {}
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "request_count": usage.request_count,
        "covered_request_count": usage.covered_request_count,
        "coverage": usage.coverage.value,
    }
