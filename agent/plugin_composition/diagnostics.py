from __future__ import annotations

import asyncio
import logging
import math
import re
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from types import TracebackType
from typing import Any, ContextManager, Generator, Literal, Protocol, cast

from agent.control.context import running_turn_id
from core.common.diagnostic_log import log_event
from core.error_context import current_client_message_id, current_session_key

_NAME = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_UNITS = frozenset({"bytes", "count", "ratio", "seconds", "tokens"})
_CURRENT_OPERATION: ContextVar[str | None] = ContextVar(
    "akashic_plugin_diagnostic_operation",
    default=None,
)
_CONTEXT_SEAL = object()
_logger = logging.getLogger("akashic.plugin.diagnostics")

ObservationKind = Literal["entrypoint", "internal"]


class PluginDiagnosticContext(Protocol):
    """Opaque Core-issued causal token for an explicit plugin handoff."""


class PluginDiagnostics(Protocol):
    """Supported diagnostics surface with identity supplied only by Context."""

    def operation(self, name: str) -> ContextManager[object]: ...

    def capture(self) -> PluginDiagnosticContext | None: ...

    def resume(
        self,
        context: PluginDiagnosticContext | None,
    ) -> ContextManager[None]: ...

    def measure(
        self,
        name: str,
        value: int | float,
        *,
        unit: str = "count",
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _PluginDiagnosticContext:
    """Carry one Core-issued causal parent across an explicit handoff."""

    plugin_id: str
    generation_id: str
    fiber: str
    operation_id: str
    session_id: str
    turn_id: str
    client_message_id: str
    _seal: object = field(repr=False, compare=False)

    def check_owner(
        self,
        plugin_id: str,
        generation_id: str,
        fiber: str,
    ) -> None:
        """Reject forged or cross-Fiber context at the resume boundary."""

        if self._seal is not _CONTEXT_SEAL:
            raise ValueError("plugin diagnostic context 不是 Core 签发的 token")
        if (self.plugin_id, self.generation_id, self.fiber) != (
            plugin_id,
            generation_id,
            fiber,
        ):
            raise ValueError("plugin diagnostic context 不属于当前 Fiber")


class PluginOperation:
    """Record one nested plugin operation without owning its control flow."""

    def __init__(
        self,
        *,
        plugin_id: str,
        generation_id: str,
        fiber: str,
        operation: str,
        observation_kind: ObservationKind,
        plugin_entrypoint: str = "",
    ) -> None:
        self.plugin_id = _text(plugin_id, "plugin_id")
        self.generation_id = _text(generation_id, "generation_id")
        self.fiber = _text(fiber, "fiber")
        self.operation = _identifier(operation, "operation")
        self.observation_kind = observation_kind
        self.plugin_entrypoint = plugin_entrypoint
        self.operation_id = secrets.token_hex(8)
        self.parent_operation_id = _CURRENT_OPERATION.get() or ""
        self._started_at: float | None = None
        self._finished = False

    def start(self) -> PluginOperation:
        """Emit the start fact once and return this operation."""

        if self._started_at is not None:
            raise RuntimeError("plugin operation 已经开始")
        self._started_at = monotonic()
        _emit_operation(self, "start", outcome="started")
        return self

    @contextmanager
    def bind(self) -> Generator[None]:
        """Bind child diagnostics to this operation for the current task."""

        if self._started_at is None or self._finished:
            raise RuntimeError("plugin operation 不可绑定")
        token = _CURRENT_OPERATION.set(self.operation_id)
        try:
            yield
        finally:
            _CURRENT_OPERATION.reset(token)

    def finish(self, error: BaseException | None = None) -> None:
        """Emit exactly one terminal fact while preserving the caller error."""

        if self._started_at is None:
            raise RuntimeError("plugin operation 尚未开始")
        if self._finished:
            raise RuntimeError("plugin operation 已经结束")
        self._finished = True
        if isinstance(error, asyncio.CancelledError):
            event = "cancelled"
            outcome = "cancelled"
            level = logging.INFO
        elif error is not None:
            event = "error"
            outcome = "error"
            level = logging.ERROR
        else:
            event = "done"
            outcome = "success"
            level = logging.INFO
        _emit_operation(
            self,
            event,
            outcome=outcome,
            duration_ms=(monotonic() - self._started_at) * 1000,
            error=error,
            level=level,
        )


class _OperationContext:
    def __init__(self, operation: PluginOperation) -> None:
        self._operation = operation
        self._binding: ContextManager[None] | None = None

    def __enter__(self) -> PluginOperation:
        _ = self._operation.start()
        self._binding = self._operation.bind()
        self._binding.__enter__()
        return self._operation

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        _ = traceback
        assert self._binding is not None
        _ = self._binding.__exit__(error_type, error, traceback)
        self._operation.finish(error)
        return False


class CorePluginDiagnostics:
    """Expose bounded plugin-owned diagnostics with Core-assigned identity."""

    def __init__(
        self,
        *,
        plugin_id: str,
        generation_id: str,
        fiber: str,
    ) -> None:
        self._plugin_id = _text(plugin_id, "plugin_id")
        self._generation_id = _text(generation_id, "generation_id")
        self._fiber = _text(fiber, "fiber")

    def operation(self, name: str) -> _OperationContext:
        """Describe one plugin-owned internal operation."""

        return _OperationContext(
            self._new_operation(name, observation_kind="internal")
        )

    def capture(self) -> PluginDiagnosticContext | None:
        """Capture the current parent for a plugin-owned queue handoff."""

        operation_id = _CURRENT_OPERATION.get()
        if operation_id is None:
            return None
        return _PluginDiagnosticContext(
            plugin_id=self._plugin_id,
            generation_id=self._generation_id,
            fiber=self._fiber,
            operation_id=operation_id,
            session_id=current_session_key.get() or "",
            turn_id=running_turn_id.get(),
            client_message_id=current_client_message_id.get(),
            _seal=_CONTEXT_SEAL,
        )

    @contextmanager
    def resume(self, context: PluginDiagnosticContext | None) -> Generator[None]:
        """Resume only a context captured by this exact plugin Fiber."""

        if context is None:
            yield
            return
        if not isinstance(context, _PluginDiagnosticContext):
            raise ValueError("plugin diagnostic context 不是 Core 签发的 token")
        context.check_owner(
            self._plugin_id,
            self._generation_id,
            self._fiber,
        )
        operation_token = _CURRENT_OPERATION.set(context.operation_id)
        session_token = current_session_key.set(context.session_id or None)
        turn_token = running_turn_id.set(context.turn_id)
        client_message_token = current_client_message_id.set(
            context.client_message_id
        )
        try:
            yield
        finally:
            current_client_message_id.reset(client_message_token)
            running_turn_id.reset(turn_token)
            current_session_key.reset(session_token)
            _CURRENT_OPERATION.reset(operation_token)

    def measure(
        self,
        name: str,
        value: int | float,
        *,
        unit: str = "count",
    ) -> None:
        """Emit one finite numeric measurement without arbitrary labels."""

        measurement = _identifier(name, "measurement")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("plugin measurement value 必须是数字")
        if not math.isfinite(value):
            raise ValueError("plugin measurement value 必须是有限数字")
        if unit not in _UNITS:
            raise ValueError(f"plugin measurement unit 无效: {unit}")
        _safe_log_event(
            logging.INFO,
            "plugin.measurement",
            plugin_id=self._plugin_id,
            generation_id=self._generation_id,
            fiber=self._fiber,
            operation_id=_CURRENT_OPERATION.get() or "",
            measurement=measurement,
            measurement_value=value,
            measurement_unit=unit,
            observation_kind="internal",
            outcome="observed",
        )

    def _new_operation(
        self,
        name: str,
        *,
        observation_kind: ObservationKind,
        plugin_entrypoint: str = "",
    ) -> PluginOperation:
        return PluginOperation(
            plugin_id=self._plugin_id,
            generation_id=self._generation_id,
            fiber=self._fiber,
            operation=name,
            observation_kind=observation_kind,
            plugin_entrypoint=plugin_entrypoint,
        )


def plugin_entrypoint(
    *,
    plugin_id: str,
    generation_id: str,
    fiber: str,
    operation: str,
    entrypoint: str = "",
) -> _OperationContext:
    """Build one Core-owned entrypoint context for a direct dispatch seam."""

    return _OperationContext(
        PluginOperation(
            plugin_id=plugin_id,
            generation_id=generation_id,
            fiber=fiber,
            operation=operation,
            observation_kind="entrypoint",
            plugin_entrypoint=entrypoint,
        )
    )


def start_plugin_entrypoint(
    *,
    plugin_id: str,
    generation_id: str,
    fiber: str,
    operation: str,
    entrypoint: str = "",
) -> PluginOperation:
    """Start an entrypoint whose callback and awaitable settle separately."""

    return PluginOperation(
        plugin_id=plugin_id,
        generation_id=generation_id,
        fiber=fiber,
        operation=operation,
        observation_kind="entrypoint",
        plugin_entrypoint=entrypoint,
    ).start()


def _emit_operation(
    operation: PluginOperation,
    terminal: str,
    *,
    outcome: str,
    duration_ms: float | None = None,
    error: BaseException | None = None,
    level: int = logging.INFO,
) -> None:
    fields: dict[str, object] = {
        "plugin_id": operation.plugin_id,
        "generation_id": operation.generation_id,
        "fiber": operation.fiber,
        "operation": operation.operation,
        "operation_id": operation.operation_id,
        "parent_operation_id": operation.parent_operation_id,
        "observation_kind": operation.observation_kind,
        "outcome": outcome,
    }
    if operation.plugin_entrypoint:
        fields["plugin_entrypoint"] = operation.plugin_entrypoint
    if duration_ms is not None:
        fields["duration_ms"] = round(duration_ms, 3)
    if error is not None:
        fields["error_type"] = type(error).__name__
    _safe_log_event(
        level,
        f"plugin.operation.{terminal}",
        **fields,
    )


def _safe_log_event(level: int, event: str, **fields: object) -> None:
    """Keep a broken diagnostic sink outside plugin business semantics."""

    try:
        log_event(
            _logger,
            level,
            event,
            session_id=current_session_key.get() or "",
            turn_id=running_turn_id.get(),
            client_message_id=current_client_message_id.get(),
            **cast(Any, fields),
        )
    except Exception:
        return


def _identifier(value: object, field: str) -> str:
    text = _text(value, field)
    if _NAME.fullmatch(text) is None:
        raise ValueError(f"{field} 无效: {text}")
    return text


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} 必须是字符串")
    if not value or value.strip() != value:
        raise ValueError(f"{field} 必须是非空且无首尾空白的字符串")
    return value
