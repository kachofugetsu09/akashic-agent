from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

from agent.plugin_composition.events import ObserveEventKey
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import ModelUsage


class ProviderProjectionError(RuntimeError):
    """An optional request projector failed and must not be silently degraded."""


@dataclass(frozen=True, slots=True)
class SessionHistoryUnit:
    """One immutable, complete logical interaction owned by Session."""

    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    messages: tuple[dict[str, Any], ...]
    message_refs: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.source_from_seq < 0 or self.consolidated_through_seq < self.source_from_seq:
            raise ValueError("history unit seq boundary 无效")
        if not self.source_message_ids:
            raise ValueError("history unit 必须包含 source message ids")
        if not self.messages:
            raise ValueError("history unit 必须包含 model messages")
        if self.message_refs and len(self.message_refs) != len(self.messages):
            raise ValueError("history unit message_refs 必须与 messages 等长")


@dataclass(frozen=True, slots=True)
class RequestHistoryUnit:
    """One immutable Session history unit offered to request projectors."""

    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    messages_json: str
    message_refs: tuple[tuple[str, int], ...]

    def messages(self) -> tuple[dict[str, Any], ...]:
        """Decode a fresh copy so plugins cannot mutate Core's Session view."""

        value = cast(Any, json.loads(self.messages_json))
        if not isinstance(value, list):
            raise ValueError("request history unit messages schema 无效")
        items = cast(list[object], value)
        if not all(isinstance(item, dict) for item in items):
            raise ValueError("request history unit messages schema 无效")
        return tuple(dict(cast(dict[str, Any], item)) for item in items)


@dataclass(frozen=True, slots=True)
class ProviderTurnInput:
    """Source-neutral immutable input for an optional request projector."""

    session_key: str
    session_created_at: str
    history_units: tuple[RequestHistoryUnit, ...]
    access_grant: object | None = None


@dataclass(slots=True)
class ProviderRequestBinding:
    """Bind one projected history to the mutable provider request loop."""

    initial_messages: list[dict[str, Any]]
    history_count: int
    attempt_replay: list[dict[str, Any]]
    prior_tool_groups: int
    channel: str
    chat_id: str
    agent_model: Any
    fallback_model: Any
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class PreparedProviderRequest:
    """Observable result of one optional provider-request preparation."""

    pending_start: int
    estimated_tokens: int
    token_quality: str
    changed: bool
    auxiliary_usages: tuple[ModelUsage, ...] = ()


class ProviderRequestGate(Protocol):
    """Prepare provider calls without exposing the projector's algorithm."""

    @property
    def pending_start(self) -> int: ...

    async def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int | None,
        trigger: str,
        force: bool,
    ) -> PreparedProviderRequest: ...

    def can_retry_context_error(self, *, context_window: int) -> bool: ...

    def record_completed_batch(
        self,
        messages: list[dict[str, Any]],
        *,
        batch_start: int,
    ) -> None: ...

    async def record_response(
        self,
        *,
        message_count: int,
        tools: list[dict[str, Any]],
        usage: ModelUsage | None,
    ) -> None: ...


class ProviderTurnProjection(Protocol):
    """Projected history plus a factory for its request-local gate."""

    @property
    def history(self) -> tuple[dict[str, Any], ...]: ...

    def bind(self, binding: ProviderRequestBinding) -> ProviderRequestGate: ...


class ProviderRequestProjection(Protocol):
    """Optionally replace Session history before a provider request is built."""

    async def open_turn(self, input: ProviderTurnInput) -> ProviderTurnProjection: ...


@dataclass(frozen=True, slots=True)
class ContextProjectionCommitted:
    """Deeply immutable notice carrying only a turn-scoped read grant."""

    session_key: str
    generation: int
    source_ref: str
    scope_channel: str
    scope_chat_id: str
    suppress_post_commit: bool
    access_grant: object


@dataclass(frozen=True, slots=True)
class ContextProjectionFact:
    """Validated durable projection data returned through a scoped grant."""

    session_key: str
    generation: int
    source_ref: str
    checkpoint_json: str
    scope_channel: str
    scope_chat_id: str


class ContextProjectionFacts(Protocol):
    """Read durable committed facts so observers can repair missed delivery."""

    def list_committed(
        self,
        access_grant: object,
        *,
        session_key: str,
    ) -> tuple[ContextProjectionFact, ...]: ...

    def get_committed(
        self,
        access_grant: object,
        *,
        session_key: str,
        source_ref: str,
    ) -> ContextProjectionFact | None: ...


PROVIDER_REQUEST_PROJECTION = ServiceKey[ProviderRequestProjection](
    "provider.request_projection.v1"
)
CONTEXT_PROJECTION_FACTS = ServiceKey[ContextProjectionFacts](
    "session.context_projection.facts.v1"
)
CONTEXT_PROJECTION_COMMITTED = ObserveEventKey[ContextProjectionCommitted](
    "session.context_projection.committed"
)
