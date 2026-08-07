from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

from agent.model_runtime.execution_history import active_shell_execution_origins
from agent.model_runtime.types import LLMResponse, ModelUsage
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

    def __post_init__(self) -> None:
        if self.source_from_seq < 0 or self.consolidated_through_seq < self.source_from_seq:
            raise ValueError("context unit seq boundary 无效")
        if not self.source_message_ids:
            raise ValueError("context unit 必须包含 source message ids")
        if not self.messages:
            raise ValueError("context unit 必须包含 model messages")


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
        base_messages: list[dict],
        scope_id: str,
        committed_units: Sequence[CommittedContextUnit] = (),
        active_compaction: ActiveCompaction | None = None,
        completed_batches: Sequence[Sequence[dict]] = (),
        current_query: object | None = None,
        max_output_tokens: int = 0,
        trigger_percent: float = 0.74,
        keep_recent_tokens: int = KEEP_RECENT_TOKENS,
        fallback_provider: "LLMProvider | None" = None,
        fallback_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_messages = [_copy_message(message) for message in base_messages]
        self._scope_id = str(scope_id).strip()
        if not self._scope_id:
            raise ValueError("scope_id 不能为空")
        self._max_output_tokens = _validate_output_budget(provider, max_output_tokens)
        self._trigger_percent = _validate_trigger_percent(trigger_percent)
        self._keep_recent_tokens = _validate_keep_recent_tokens(keep_recent_tokens)
        self._fallback_provider = fallback_provider
        self._fallback_model = str(fallback_model or "").strip()
        self._committed_units = list(committed_units)
        self._active_compaction = active_compaction
        self._completed_batches = [
            tuple(_copy_message(message) for message in batch)
            for batch in completed_batches
        ]
        self._current_query = (
            _bounded_text(current_query, 8000)
            if current_query is not None
            else _find_current_query(self._base_messages)
        )
        self._compaction: ContextCompaction | None = None
        self._meter = _ContextTokenMeter()

    @property
    def pending_start(self) -> int:
        return len(self._base_messages)

    @property
    def compaction(self) -> ContextCompaction | None:
        return self._compaction

    @property
    def has_compactable_prefix(self) -> bool:
        return bool(self._candidate_units())

    def checkpoint_payload(self) -> dict[str, object] | None:
        return self._compaction.to_payload() if self._compaction is not None else None

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
        estimated, quality = self._meter.estimate(self._provider, messages, tools)
        soft_limit = math.floor(self._provider.context_window * self._trigger_percent)
        hard_limit = hard_input_limit(self._provider, self._max_output_tokens)
        boundary_hit = estimated >= soft_limit or estimated >= hard_limit
        if not force and not boundary_hit:
            return PreparedQueryContext(pending_start, estimated, quality, False, None)

        # 2. Select only complete committed units or closed active batches.
        candidates = self._candidate_units()
        if not candidates:
            raise ContextCompactionError("context_compaction_no_closed_prefix")
        selected, retained = self._select_units(candidates)
        summary, summary_usage = await self._summarize(selected)
        generation = (
            1
            if self._active_compaction is None
            else self._active_compaction.generation + 1
        )
        parent_generation = (
            0
            if self._active_compaction is None
            else self._active_compaction.generation
        )

        # 3. Replace old history with one opaque summary and complete raw tail.
        prefix, pending = self._split_base(messages, pending_start, candidates)
        summary_message = build_compaction_messages(
            summary,
            generation=generation,
            source_ref=_compaction_source_ref(self._scope_id, generation),
        )
        retained_messages = [
            _strip_opaque_state(message)
            for unit in retained
            for message in unit.messages
        ]
        rebuilt = [
            *prefix,
            *summary_message,
            *retained_messages,
            *[_strip_opaque_state(message) for message in pending],
        ]
        after = self._provider.estimate_context_tokens(rebuilt, tools)
        if after >= soft_limit or after >= hard_limit:
            raise ContextCompactionError(
                "context_compaction_insufficient "
                f"estimated={after} soft_limit={soft_limit} hard_limit={hard_limit}"
            )

        checkpoint = _build_checkpoint(
            summary=summary,
            generation=generation,
            parent_generation=parent_generation,
            trigger=trigger,
            context_window=self._provider.context_window,
            soft_limit_tokens=soft_limit,
            hard_input_tokens=hard_limit,
            keep_recent_tokens=self._keep_recent_tokens,
            estimated_tokens_before=estimated,
            estimated_tokens_after=after,
            selected=selected,
            retained=retained,
            summary_usage=summary_usage,
            source_ref=_compaction_source_ref(self._scope_id, generation),
        )
        self._compaction = checkpoint
        self._active_compaction = ActiveCompaction(
            generation=checkpoint.generation,
            summary=checkpoint.summary,
            source_from_seq=checkpoint.source_from_seq,
            consolidated_through_seq=checkpoint.consolidated_through_seq,
            source_message_ids=checkpoint.source_message_ids,
            retained_tail=checkpoint.retained_tail,
        )
        self._committed_units = [unit for unit in retained if unit.source_message_ids]
        self._completed_batches = [
            tuple(unit.messages) for unit in retained if not unit.source_message_ids
        ]
        messages[:] = rebuilt
        self._meter.invalidate()
        return PreparedQueryContext(
            len(prefix) + len(summary_message) + len(retained_messages),
            after,
            "approximate",
            True,
            summary_usage,
            checkpoint,
        )

    def _candidate_units(self) -> list[CommittedContextUnit]:
        units = list(self._committed_units)
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
            if kept and kept_tokens + tokens > self._keep_recent_tokens:
                break
            kept.insert(0, unit)
            kept_tokens += tokens
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

    def _split_base(
        self,
        messages: list[dict],
        pending_start: int,
        candidates: list[CommittedContextUnit],
    ) -> tuple[list[dict], list[dict]]:
        """Find the committed projection in the assembled payload and preserve active input."""

        if pending_start < 0 or pending_start > len(messages):
            raise ValueError("pending_start 超出当前 payload")
        # The caller may provide only current-turn batches. In that case old history
        # is represented by the committed unit messages at the front of the payload.
        first = candidates[0].messages
        start = _find_subsequence(messages, first)
        if start is None:
            raise ContextCompactionError("context_compaction_source_not_in_payload")
        return (
            _without_previous_compaction(
                [_copy_message(message) for message in messages[:start]]
            ),
            [_copy_message(message) for message in messages[pending_start:]],
        )

    async def _summarize(
        self,
        selected: Sequence[CommittedContextUnit],
    ) -> tuple[str, ModelUsage | None]:
        sections = [_SUMMARY_PROMPT, "\n[Current user query]\n", self._current_query]
        if self._active_compaction is not None:
            sections.extend(
                ["\n[Previous compaction summary]\n", self._active_compaction.summary]
            )
        sections.append("\n[Committed history to consolidate]\n")
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
                response = await provider.chat(
                    messages=[{"role": "user", "content": "".join(sections)}],
                    tools=[],
                    model=model,
                    max_tokens=_summary_output_limit(provider, self._max_output_tokens),
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


def _summary_output_limit(provider: "LLMProvider", max_output_tokens: int) -> int:
    hard_limit = hard_input_limit(provider, max_output_tokens)
    return max(1, min(SUMMARY_MAX_TOKENS, hard_limit // 8 or hard_limit))


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
        if not message_id.startswith("active:")
    )
    committed = [unit for unit in selected if not any(
        item.startswith("active:") for item in unit.source_message_ids
    )]
    if committed:
        source_from_seq = committed[0].source_from_seq
        through_seq = committed[-1].consolidated_through_seq
    else:
        source_from_seq = 0
        through_seq = 0
    retained_tail = tuple(
        _copy_message(message)
        for unit in retained
        for message in unit.messages
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


def _find_subsequence(
    haystack: Sequence[dict[str, Any]],
    needle: Sequence[dict[str, Any]],
) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return None


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
