from __future__ import annotations

import asyncio
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from agent.model_runtime.execution_history import active_shell_execution_origins
from agent.model_runtime.types import ModelUsage
from agent.model_runtime.usage import aggregate_usage
from agent.prompting import is_context_frame

if TYPE_CHECKING:
    from agent.provider import LLMProvider

COMPACTION_SCHEMA_VERSION = 1
COMPACTION_TOOL_NAME = "context_compact"
_KEEP_RECENT_PERCENT = 0.20
_SUMMARY_MAX_TOKENS = 8192
_SUMMARY_MAX_RETRIES = 3
_SUMMARY_RETRY_BASE_DELAY_SECONDS = 2.0
_TOOL_RESULT_SUMMARY_CHAR_LIMIT = 2000
_MESSAGE_SUMMARY_CHAR_LIMIT = 8000

CompactionTrigger = Literal["soft_limit", "context_overflow"]

_SUMMARY_PROMPT = """更新当前长任务的上下文压缩摘要。

摘要会替代已经完成的旧工具步骤，供同一个任务后续继续执行。只记录输入中已经出现的事实，不补充猜测，不把计划写成已完成。

必须使用以下标题：
## Goal
## Constraints
## Progress
## Key facts and references
## Decisions
## Validation
## Unfinished work
## Next steps

保留文件路径、符号、命令、错误、数值和验证结果。若输入中存在仍在运行的 shell execution，必须保留 execution_id、命令和当前状态。省略重复探索、无用日志、tool_call_id 和其他协议细节。只输出摘要正文。"""


class ContextCompactionError(RuntimeError):
    """当前 query 无法生成可继续执行的有界上下文。"""


@dataclass(frozen=True)
class ReactCompaction:
    summary: str
    compacted_tool_groups: int
    generation: int
    trigger: CompactionTrigger
    context_window: int
    soft_limit_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": COMPACTION_SCHEMA_VERSION,
            "summary": self.summary,
            "compacted_tool_groups": self.compacted_tool_groups,
            "generation": self.generation,
            "trigger": self.trigger,
            "context_window": self.context_window,
            "soft_limit_tokens": self.soft_limit_tokens,
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
        }


def parse_react_compaction(
    value: object,
    *,
    source: str,
) -> ReactCompaction:
    """Validate and decode persisted compaction metadata."""

    # 1. 校验版本、摘要和枚举字段。
    if not isinstance(value, dict):
        raise ValueError(f"react_compaction 必须是 JSON object: {source}")
    raw = cast(dict[str, object], value)
    if raw.get("schema_version") != COMPACTION_SCHEMA_VERSION:
        raise ValueError(f"react_compaction schema_version 无效: {source}")
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"react_compaction.summary 必须是非空字符串: {source}")
    if len(summary.encode("utf-8")) > 512 * 1024:
        raise ValueError(f"react_compaction.summary 超过 512 KiB: {source}")
    trigger = raw.get("trigger")
    if trigger not in {"soft_limit", "context_overflow"}:
        raise ValueError(f"react_compaction.trigger 无效: {source}")

    # 2. 校验计数和预算，避免损坏元数据改变重放切点。
    integers: dict[str, int] = {}
    for field in (
        "compacted_tool_groups",
        "generation",
        "context_window",
        "soft_limit_tokens",
        "estimated_tokens_before",
        "estimated_tokens_after",
    ):
        item = raw.get(field)
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"react_compaction.{field} 必须是整数: {source}")
        integers[field] = item
    if integers["compacted_tool_groups"] <= 0:
        raise ValueError(f"react_compaction.compacted_tool_groups 必须大于 0: {source}")
    if integers["generation"] <= 0:
        raise ValueError(f"react_compaction.generation 必须大于 0: {source}")
    if integers["context_window"] <= 0 or integers["soft_limit_tokens"] <= 0:
        raise ValueError(f"react_compaction 模型预算必须大于 0: {source}")
    if integers["soft_limit_tokens"] >= integers["context_window"]:
        raise ValueError(f"react_compaction 软水位必须小于模型窗口: {source}")
    if (
        integers["estimated_tokens_before"] < 0
        or integers["estimated_tokens_after"] < 0
    ):
        raise ValueError(f"react_compaction token 估算不能为负数: {source}")

    return ReactCompaction(
        summary=summary.strip(),
        compacted_tool_groups=integers["compacted_tool_groups"],
        generation=integers["generation"],
        trigger=cast(CompactionTrigger, trigger),
        context_window=integers["context_window"],
        soft_limit_tokens=integers["soft_limit_tokens"],
        estimated_tokens_before=integers["estimated_tokens_before"],
        estimated_tokens_after=integers["estimated_tokens_after"],
    )


def build_compaction_messages(
    compaction: ReactCompaction,
    *,
    call_id: str,
) -> list[dict[str, Any]]:
    """Project an internal compaction boundary as a paired provider message."""
    arguments = json.dumps(
        {
            "scope": "current_user_query",
            "compacted_tool_groups": compaction.compacted_tool_groups,
            "generation": compaction.generation,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": COMPACTION_TOOL_NAME,
                        "arguments": arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": compaction.summary,
        },
    ]


def build_replay_compaction_messages(
    compaction: ReactCompaction,
    *,
    message_id: str,
) -> list[dict[str, Any]]:
    """Build the stable compact pair used when SessionDB history is reloaded."""
    return build_compaction_messages(
        compaction,
        call_id=_compaction_call_id(
            f"persisted:{message_id}",
            compaction.generation,
        ),
    )


@dataclass(frozen=True)
class PreparedQueryContext:
    pending_start: int
    estimated_tokens: int
    estimate_quality: Literal["approximate", "exact_plus_delta"]
    compacted: bool
    summary_usage: ModelUsage | None


class _ContextTokenMeter:
    def __init__(self) -> None:
        self._exact_input_tokens: int | None = None
        self._message_count = 0
        self._tool_digest = ""

    def estimate(
        self,
        provider: LLMProvider,
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


class QueryCompactor:
    """Compact completed batches while retaining full execution evidence elsewhere."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        base_messages: list[dict],
        scope_id: str,
        completed_batches: list[list[dict]] | None = None,
        current_query: object | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_messages = [_copy_message(message) for message in base_messages]
        self._scope_id = scope_id
        self._completed_batches = [
            [_copy_message(message) for message in batch]
            for batch in completed_batches or []
        ]
        self._current_query = (
            _bounded_text(current_query, _MESSAGE_SUMMARY_CHAR_LIMIT)
            if current_query is not None
            else _find_current_query(self._base_messages)
        )
        self._compaction: ReactCompaction | None = None
        self._meter = _ContextTokenMeter()

    @property
    def pending_start(self) -> int:
        """返回初始消息中尚未闭合为 tool batch 的起点。"""

        return len(self._base_messages) + sum(
            len(batch) for batch in self._completed_batches
        )

    @property
    def compaction(self) -> ReactCompaction | None:
        return self._compaction

    @property
    def has_compactable_prefix(self) -> bool:
        return self._select_compact_count() > 0

    def persistence_payload(self) -> dict[str, object] | None:
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

    def record_completed_batch(
        self,
        messages: list[dict],
        *,
        batch_start: int,
    ) -> None:
        if batch_start < 0 or batch_start >= len(messages):
            raise ValueError("完整工具批次缺少可记录消息")
        self._completed_batches.append(
            [_copy_message(message) for message in messages[batch_start:]]
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
        """Compact an eligible prefix before the next provider request."""

        # 1. 使用 provider owner 的统一口径判断是否达到软水位。
        estimated, quality = self._meter.estimate(self._provider, messages, tools)
        soft_limit = self._provider.compaction_trigger_tokens
        if not force and (soft_limit <= 0 or estimated < soft_limit):
            return PreparedQueryContext(pending_start, estimated, quality, False, None)

        # 2. 至少保留最近一个完整批次；没有可淘汰前缀时推迟压缩。
        compact_count = self._select_compact_count()
        if compact_count <= 0:
            if force:
                raise ContextCompactionError("context_compaction_no_closed_prefix")
            return PreparedQueryContext(pending_start, estimated, quality, False, None)
        if pending_start < 0 or pending_start > len(messages):
            raise ValueError("pending_start 超出当前 query 消息范围")

        evicted = self._completed_batches[:compact_count]
        retained = self._completed_batches[compact_count:]
        pending = [_copy_message(item) for item in messages[pending_start:]]
        summary, summary_usage = await self._summarize(evicted)
        generation = 1 if self._compaction is None else self._compaction.generation + 1
        compacted_groups = compact_count + (
            self._compaction.compacted_tool_groups
            if self._compaction is not None
            else 0
        )
        candidate = ReactCompaction(
            summary=summary,
            compacted_tool_groups=compacted_groups,
            generation=generation,
            trigger=trigger,
            context_window=self._provider.context_window,
            soft_limit_tokens=soft_limit,
            estimated_tokens_before=estimated,
            estimated_tokens_after=0,
        )

        # 3. 重建临时模型视图；被压缩后的 opaque model_state 不再重放。
        pair = build_compaction_messages(
            candidate,
            call_id=_compaction_call_id(self._scope_id, generation),
        )
        retained = [
            [_strip_opaque_state(message) for message in batch] for batch in retained
        ]
        pending = [_strip_opaque_state(message) for message in pending]
        prefix = [
            *[_copy_message(message) for message in self._base_messages],
            *pair,
            *[message for batch in retained for message in batch],
        ]
        rebuilt = [*prefix, *pending]
        after = self._provider.estimate_context_tokens(rebuilt, tools)
        hard_limit = self._provider.hard_input_tokens
        if hard_limit <= 0 or after >= hard_limit:
            raise ContextCompactionError(
                "context_compaction_insufficient "
                f"estimated={after} hard_limit={hard_limit}"
            )

        self._compaction = ReactCompaction(
            summary=candidate.summary,
            compacted_tool_groups=candidate.compacted_tool_groups,
            generation=candidate.generation,
            trigger=candidate.trigger,
            context_window=candidate.context_window,
            soft_limit_tokens=candidate.soft_limit_tokens,
            estimated_tokens_before=candidate.estimated_tokens_before,
            estimated_tokens_after=after,
        )
        self._completed_batches = retained
        messages[:] = rebuilt
        self._meter.invalidate()
        return PreparedQueryContext(
            len(prefix),
            after,
            "approximate",
            True,
            summary_usage,
        )

    def _select_compact_count(self) -> int:
        """Keep a recent suffix sized from the model window."""
        if len(self._completed_batches) < 2:
            return 0
        keep_budget = max(
            1,
            math.floor(self._provider.context_window * _KEEP_RECENT_PERCENT),
        )
        keep_count = 0
        kept_tokens = 0
        for batch in reversed(self._completed_batches):
            batch_tokens = self._provider.estimate_appended_message_tokens(batch)
            if keep_count > 0 and kept_tokens + batch_tokens > keep_budget:
                break
            keep_count += 1
            kept_tokens += batch_tokens
        selected = min(
            len(self._completed_batches) - 1,
            max(1, len(self._completed_batches) - keep_count),
        )
        active_batch = _active_execution_batch(self._completed_batches)
        return selected if active_batch is None else min(selected, active_batch)

    async def _summarize(
        self,
        evicted: list[list[dict]],
    ) -> tuple[str, ModelUsage | None]:
        """Generate one structured handoff from previous summary and evicted batches."""

        # 1. 序列化受控输入，避免单个工具结果占满 summary 请求。
        sections = [
            _SUMMARY_PROMPT,
            "\n[Current user query]\n",
            self._current_query,
        ]
        if self._compaction is not None:
            sections.extend(
                [
                    "\n[Previous compaction summary]\n",
                    self._compaction.summary,
                ]
            )
        sections.append("\n[New completed steps]\n")
        sections.extend(
            _serialize_message(message) for batch in evicted for message in batch
        )

        # 2. summary 不携带业务工具和主 ReAct cache；语义无效时有界退避重试。
        usages: list[ModelUsage] = []
        for attempt in range(_SUMMARY_MAX_RETRIES + 1):
            response = await self._provider.chat(
                messages=[{"role": "user", "content": "".join(sections)}],
                tools=[],
                model=self._model,
                max_tokens=min(
                    _SUMMARY_MAX_TOKENS,
                    max(512, self._provider.hard_input_tokens // 8),
                ),
                disable_thinking=True,
            )
            if response.usage is not None:
                usages.append(response.usage)
            summary = (response.content or "").strip()
            if summary and not response.tool_calls:
                return summary, aggregate_usage(usages) if usages else None
            if attempt < _SUMMARY_MAX_RETRIES:
                await asyncio.sleep(_SUMMARY_RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ContextCompactionError("context_compaction_summary_invalid")


def _tool_schema_digest(tools: list[dict]) -> str:
    encoded = json.dumps(
        tools,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _active_execution_batch(batches: list[list[dict]]) -> int | None:
    """Return the earliest batch that must remain visible for a live execution."""

    # 1. 保留 tool_call_id 到原始 batch 的稳定映射。
    call_batches: dict[str, int] = {}
    messages: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        messages.extend(cast(list[dict[str, Any]], batch))
        for message in batch:
            calls = message.get("tool_calls")
            if message.get("role") != "assistant" or not isinstance(calls, list):
                continue
            for raw_call in calls:
                if not isinstance(raw_call, dict):
                    continue
                call_id = str(raw_call.get("id") or "")
                if call_id:
                    call_batches[call_id] = batch_index

    # 2. 只 pin 仍在运行 execution 的最早创建 batch。
    active_origins = active_shell_execution_origins(messages)
    active_batches = [
        call_batches[call_id]
        for call_id in active_origins.values()
        if call_id in call_batches
    ]
    return min(active_batches) if active_batches else None


def _compaction_call_id(scope_id: str, generation: int) -> str:
    digest = hashlib.sha256(f"{scope_id}\0{generation}".encode("utf-8")).hexdigest()
    return f"cmp_{digest[:24]}"


def _copy_message(message: dict) -> dict[str, Any]:
    return deepcopy(cast(dict[str, Any], message))


def _strip_opaque_state(message: dict) -> dict[str, Any]:
    clean = _copy_message(message)
    clean.pop("model_state", None)
    return clean


def _find_current_query(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and is_context_frame(content):
            continue
        return _bounded_text(content, _MESSAGE_SUMMARY_CHAR_LIMIT)
    return "(current user query is retained in the active prompt)"


def _serialize_message(message: dict) -> str:
    clean = {
        key: value
        for key, value in message.items()
        if key not in {"model_state", "reasoning_content"}
    }
    content = clean.get("content")
    limit = (
        _TOOL_RESULT_SUMMARY_CHAR_LIMIT
        if clean.get("role") == "tool"
        else _MESSAGE_SUMMARY_CHAR_LIMIT
    )
    clean["content"] = _bounded_text(content, limit)
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    return f"\n{encoded}"


def _bounded_text(value: object, limit: int) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        blocks: list[object] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in {
                "image_url",
                "input_image",
            }:
                blocks.append({"type": item.get("type"), "content": "[image omitted]"})
            else:
                blocks.append(item)
        text = json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value or "")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    marker = f"\n…{omitted} chars omitted from compaction input…\n"
    keep = max(0, limit - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")
