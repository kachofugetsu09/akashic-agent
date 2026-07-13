from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Protocol


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
    request_count: int = 1
    covered_request_count: int = 0
    coverage: UsageCoverage = UsageCoverage.UNAVAILABLE


@dataclass(frozen=True)
class ModelCapabilities:
    context_window: int
    max_output_tokens: int
    effective_context_percent: float = 0.9
    max_context_window: int | None = None
    supported_reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    input_modalities: tuple[str, ...] = ("text",)
    supports_parallel_tool_calls: bool = True
    supports_reasoning_summaries: bool = False
    use_responses_lite: bool = False

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
    on_delta: StreamCallback | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    disable_thinking: bool = False


class ModelBackend(Protocol):
    async def send(self, request: ModelRequest) -> LLMResponse: ...
