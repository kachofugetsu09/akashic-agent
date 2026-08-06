from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} 必须包含时区")
    return value.astimezone(UTC)


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: object, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是 RFC 3339 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} 不是有效的 RFC 3339 时间") from exc
    return _as_utc(parsed, field_name)


def _require_dict(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 JSON object")
    return cast(dict[str, Any], value)


class ThreadSource(StrEnum):
    PROGRAMMATIC = "programmatic"
    CHANNEL = "channel"
    INTERNAL = "internal"


class TurnStatus(StrEnum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TurnStatus.COMPLETED,
            TurnStatus.INTERRUPTED,
            TurnStatus.FAILED,
            TurnStatus.CANCELLED,
        }


class TurnItemKind(StrEnum):
    USER_MESSAGE = "userMessage"
    ASSISTANT_MESSAGE = "assistantMessage"
    REASONING = "reasoning"
    TOOL_CALL = "toolCall"
    ERROR = "error"


@dataclass(frozen=True)
class ThreadRecord:
    id: str
    source: ThreadSource
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ThreadSource(self.source))
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _as_utc(self.updated_at, "updated_at"))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source.value,
            "createdAt": _format_datetime(self.created_at),
            "updatedAt": _format_datetime(self.updated_at),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TurnItem:
    kind: TurnItemKind
    id: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TurnItemKind(self.kind))
        if not self.id:
            raise ValueError("turn item id 不能为空")

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "type": self.kind.value, "data": dict(self.data)}

    @classmethod
    def from_dict(cls, payload: object) -> TurnItem:
        data = _require_dict(payload, "turn item")
        item_id = data.get("id")
        item_type = data.get("type")
        if not isinstance(item_id, str):
            raise ValueError("turn item id 必须是字符串")
        if not isinstance(item_type, str):
            raise ValueError("turn item type 必须是字符串")
        return cls(
            id=item_id,
            kind=TurnItemKind(item_type),
            data=_require_dict(data.get("data"), "turn item data"),
        )


@dataclass(frozen=True)
class TurnUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    request_count: int = 0
    covered_request_count: int = 0
    coverage: str = "unavailable"

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.reasoning_output_tokens,
            self.request_count,
            self.covered_request_count,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("turn usage token/count 字段不得为负数")
        if self.coverage not in {"exact", "partial", "unavailable"}:
            raise ValueError(f"turn usage coverage 无效: {self.coverage}")

    def to_dict(self) -> dict[str, object]:
        return {
            "inputTokens": self.input_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "outputTokens": self.output_tokens,
            "reasoningOutputTokens": self.reasoning_output_tokens,
            "requestCount": self.request_count,
            "coveredRequestCount": self.covered_request_count,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TurnUsage:
        data = _require_dict(payload, "turn usage")

        # 1. JSON number 必须是非布尔整数，避免 SQLite 损坏被隐式归一化。
        values: dict[str, int | None] = {}
        for wire_name in (
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
        ):
            value = data.get(wire_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"turn usage {wire_name} 必须是整数或 null")
            values[wire_name] = cast(int | None, value)
        for wire_name in ("requestCount", "coveredRequestCount"):
            value = data.get(wire_name, 0)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"turn usage {wire_name} 必须是整数")
            values[wire_name] = value
        coverage = data.get("coverage", "unavailable")
        if not isinstance(coverage, str):
            raise ValueError("turn usage coverage 必须是字符串")

        # 2. 交给领域构造函数校验范围和枚举值。
        return cls(
            input_tokens=values["inputTokens"],
            cached_input_tokens=values["cachedInputTokens"],
            output_tokens=values["outputTokens"],
            reasoning_output_tokens=values["reasoningOutputTokens"],
            request_count=cast(int, values["requestCount"]),
            covered_request_count=cast(int, values["coveredRequestCount"]),
            coverage=coverage,
        )


@dataclass(frozen=True)
class TurnError:
    type: str
    message: str
    retryable: bool
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": self.type,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.data is not None:
            payload["data"] = dict(self.data)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> TurnError:
        data = _require_dict(payload, "turn error")
        error_data = data.get("data")
        if error_data is not None:
            error_data = _require_dict(error_data, "turn error data")
        retryable = data.get("retryable")
        if not isinstance(retryable, bool):
            raise ValueError("turn error retryable 必须是布尔值")
        error_type = data.get("type")
        message = data.get("message")
        if not isinstance(error_type, str) or not error_type:
            raise ValueError("turn error type 必须是非空字符串")
        if not isinstance(message, str) or not message:
            raise ValueError("turn error message 必须是非空字符串")
        return cls(
            type=error_type,
            message=message,
            retryable=retryable,
            data=cast(dict[str, Any] | None, error_data),
        )


@dataclass(frozen=True)
class TurnRequest:
    thread_id: str
    input: str
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def to_dict(self) -> dict[str, object]:
        return {
            "threadId": self.thread_id,
            "input": self.input,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TurnRecord:
    id: str
    thread_id: str
    status: TurnStatus
    input: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    items: list[TurnItem] = field(default_factory=list[TurnItem])
    usage: TurnUsage | None = None
    error: TurnError | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    final_response: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", TurnStatus(self.status))
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))
        if self.started_at is not None:
            object.__setattr__(
                self, "started_at", _as_utc(self.started_at, "started_at")
            )
        if self.completed_at is not None:
            object.__setattr__(
                self, "completed_at", _as_utc(self.completed_at, "completed_at")
            )

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return max(0, int((self.completed_at - self.started_at).total_seconds() * 1000))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "threadId": self.thread_id,
            "status": self.status.value,
            "input": self.input,
            "metadata": dict(self.metadata),
            "createdAt": _format_datetime(self.created_at),
            "startedAt": _format_datetime(self.started_at),
            "completedAt": _format_datetime(self.completed_at),
            "durationMs": self.duration_ms,
            "finalResponse": self.final_response,
            "items": [item.to_dict() for item in self.items],
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
        }


@dataclass(frozen=True)
class TurnResult:
    id: str
    thread_id: str
    status: TurnStatus
    started_at: datetime | None
    completed_at: datetime | None
    final_response: str | None
    items: list[TurnItem]
    usage: TurnUsage | None
    error: TurnError | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", TurnStatus(self.status))
        if self.started_at is not None:
            object.__setattr__(
                self, "started_at", _as_utc(self.started_at, "started_at")
            )
        if self.completed_at is not None:
            object.__setattr__(
                self, "completed_at", _as_utc(self.completed_at, "completed_at")
            )

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return max(0, int((self.completed_at - self.started_at).total_seconds() * 1000))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "threadId": self.thread_id,
            "status": self.status.value,
            "startedAt": _format_datetime(self.started_at),
            "completedAt": _format_datetime(self.completed_at),
            "durationMs": self.duration_ms,
            "finalResponse": self.final_response,
            "items": [item.to_dict() for item in self.items],
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
        }

    @classmethod
    def from_record(cls, record: TurnRecord) -> TurnResult:
        return cls(
            id=record.id,
            thread_id=record.thread_id,
            status=record.status,
            started_at=record.started_at,
            completed_at=record.completed_at,
            final_response=record.final_response,
            items=list(record.items),
            usage=record.usage,
            error=record.error,
        )


def parse_rfc3339(value: object, field_name: str) -> datetime | None:
    """在持久化边界解析 RFC 3339 UTC 时间。"""
    return _parse_datetime(value, field_name)
