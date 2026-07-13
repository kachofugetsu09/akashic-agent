from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable


class CapabilitySource(StrEnum):
    """标记模型能力值的来源。"""

    CATALOG = "catalog"
    OVERRIDE = "override"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class UsageCoverage(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    cache_write_tokens: int | None = None
    request_count: int = 1
    covered_request_count: int = 0
    coverage: UsageCoverage = UsageCoverage.UNAVAILABLE
    raw_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelCapabilities:
    context_window: int
    max_output_tokens: int
    effective_context_percent: float = 0.9
    supported_reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    input_modalities: tuple[str, ...] = ("text",)
    supports_image_original_detail: bool = False
    supports_tool_calling: bool = True
    supports_parallel_tool_calls: bool = True
    supported_tool_choices: tuple[str, ...] = ("auto", "none", "required")
    supports_streaming: bool = True
    supports_stream_usage: bool = True
    supports_prompt_cache: bool = False
    continuation_mode: str = "messages"
    source: CapabilitySource = CapabilitySource.MANUAL

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window 必须大于 0")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens 必须大于 0")
        if not 0 < self.effective_context_percent <= 1:
            raise ValueError("effective_context_percent 必须在 (0, 1] 内")
        if "text" not in self.input_modalities:
            raise ValueError("input_modalities 必须包含 text")


@dataclass(frozen=True)
class ContinuationState:
    runtime_id: str
    transport: str
    model: str
    items: tuple[dict[str, Any], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("不支持的 model_state schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_id": self.runtime_id,
            "transport": self.transport,
            "model": self.model,
            "items": list(self.items),
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    provider_fields: dict[str, Any] = field(default_factory=dict)
    cache_prompt_tokens: int | None = None
    cache_hit_tokens: int | None = None
    usage: ModelUsage | None = None
    continuation: ContinuationState | None = None


StreamCallback = Callable[[dict[str, str]], Awaitable[None]]


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    model: str
    max_output_tokens: int
    system_prompt: str = ""
    tool_choice: str | dict[str, Any] = "auto"
    reasoning_effort: str | None = None
    prompt_cache_key: str | None = None
    continuation: ContinuationState | None = None
    on_delta: StreamCallback | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    disable_thinking: bool = False
