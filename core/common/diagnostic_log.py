from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

_DIAG_FIELDS = (
    "event",
    "flow",
    "phase",
    "session",
    "turn",
    "tick",
    "action",
    "reason",
    "duration_ms",
    "counts",
    "error_type",
    "error_fp",
    "note",
)

_STRUCTURED_FIELDS = frozenset(
    {
        "action",
        "boot_id",
        "command_bytes",
        "command_fp",
        "counts",
        "cwd",
        "duration_ms",
        "error_fp",
        "error_type",
        "event",
        "execution_id",
        "exit_code",
        "finish_reason",
        "flow",
        "manager_id",
        "method",
        "operation",
        "outcome",
        "output_bytes",
        "phase",
        "reason",
        "release_commit",
        "request_id",
        "session",
        "source",
        "tick",
        "toolchain_digest",
        "turn",
        "tty",
    }
)
_SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[-_]?key|token|password|passwd|secret|cookie)"
    r"(\s*[:=]\s*|\s+)([^\s,;]+)"
)
_DIAGNOSTIC_EVENT_PATTERN = re.compile(
    r"^\[(?P<operation>[^]]+)](?:\s+\S+)*\s+event=(?P<event>[^\s]+)"
)
_MAX_LOG_TEXT = 4096

diagnostic_session: ContextVar[str | None] = ContextVar(
    "diagnostic_session", default=None
)
diagnostic_flow: ContextVar[str | None] = ContextVar("diagnostic_flow", default=None)
diagnostic_phase: ContextVar[str | None] = ContextVar("diagnostic_phase", default=None)
diagnostic_turn: ContextVar[str | None] = ContextVar("diagnostic_turn", default=None)
diagnostic_tick: ContextVar[str | None] = ContextVar("diagnostic_tick", default=None)
diagnostic_request: ContextVar[str | None] = ContextVar(
    "diagnostic_request", default=None
)
diagnostic_execution: ContextVar[str | None] = ContextVar(
    "diagnostic_execution", default=None
)


class JsonLogFormatter(logging.Formatter):
    """Serialize one bounded event with stable correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        document: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "service": os.environ.get("AKASHIC_SERVICE_NAME", "akashic"),
            "logger": record.name,
            "message": _redact(record.getMessage()),
            "pid": record.process,
        }
        document.update(
            {key: value for key, value in current_diagnostic_context().items() if value}
        )
        fields = getattr(record, "akashic_fields", {})
        if isinstance(fields, Mapping):
            for key, value in fields.items():
                if key in _STRUCTURED_FIELDS and value is not None and value != "":
                    document[key] = _bounded_value(value)
        if "event" not in document:
            diagnostic = _DIAGNOSTIC_EVENT_PATTERN.match(record.getMessage())
            if diagnostic is not None:
                document["event"] = diagnostic.group("event")
                document["operation"] = diagnostic.group("operation")
        release_commit = os.environ.get("AKASHIC_RUNTIME_COMMIT")
        boot_id = os.environ.get("AKASHIC_BOOT_ID")
        if release_commit and "release_commit" not in document:
            document["release_commit"] = release_commit
        if boot_id and "boot_id" not in document:
            document["boot_id"] = boot_id
        if record.exc_info:
            document["exception"] = _redact(self.formatException(record.exc_info))
        return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def configure_logging() -> None:
    """Configure stderr logging from the process environment."""

    level_name = os.environ.get("AKASHIC_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"AKASHIC_LOG_LEVEL 无效: {level_name}")
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("AKASHIC_LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    message: str = "",
    exc_info: bool = False,
    **fields: object,
) -> None:
    """Emit one allow-listed event without attaching arbitrary payloads."""

    unknown = set(fields) - _STRUCTURED_FIELDS
    if unknown:
        raise ValueError(f"未知结构化日志字段: {', '.join(sorted(unknown))}")
    logger.log(
        level,
        message or event,
        extra={"akashic_fields": {"event": event, **fields}},
        exc_info=exc_info,
    )


def diagnostic_line(method: str, **fields: object) -> str:
    parts = [f"[{method}]"]
    for key in _DIAG_FIELDS:
        value = _clean(fields.get(key, "-"))
        if key == "note":
            value = f'"{value}"'
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def diagnostic_context(
    *,
    session: str | None = None,
    flow: str | None = None,
    phase: str | None = None,
    turn: str | None = None,
    tick: str | None = None,
    request_id: str | None = None,
    execution_id: str | None = None,
) -> Iterator[None]:
    tokens = []
    if session is not None:
        tokens.append((diagnostic_session, diagnostic_session.set(session)))
    if flow is not None:
        tokens.append((diagnostic_flow, diagnostic_flow.set(flow)))
    if phase is not None:
        tokens.append((diagnostic_phase, diagnostic_phase.set(phase)))
    if turn is not None:
        tokens.append((diagnostic_turn, diagnostic_turn.set(turn)))
    if tick is not None:
        tokens.append((diagnostic_tick, diagnostic_tick.set(tick)))
    if request_id is not None:
        tokens.append((diagnostic_request, diagnostic_request.set(request_id)))
    if execution_id is not None:
        tokens.append((diagnostic_execution, diagnostic_execution.set(execution_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_diagnostic_context() -> dict[str, str]:
    return {
        "session": diagnostic_session.get() or "",
        "flow": diagnostic_flow.get() or "",
        "phase": diagnostic_phase.get() or "",
        "turn": diagnostic_turn.get() or "",
        "tick": diagnostic_tick.get() or "",
        "request_id": diagnostic_request.get() or "",
        "execution_id": diagnostic_execution.get() or "",
    }


def _clean(value: object) -> str:
    text = str(value if value is not None else "-").replace("\n", " ").strip()
    if not text:
        return "-"
    return text.replace('"', "'")


def _redact(value: object) -> str:
    text = str(value).replace("\x00", "�")
    text = _SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text
    )
    if len(text) > _MAX_LOG_TEXT:
        return f"{text[:_MAX_LOG_TEXT]}…[truncated]"
    return text


def _bounded_value(value: object) -> object:
    if isinstance(value, (bool, int, float)):
        return value
    return _redact(value)
