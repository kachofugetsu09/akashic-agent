from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sys
import threading
import traceback
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Callable

from core.common.diagnostic_log import current_diagnostic_context

from .events import GlobalErrorTrace

_SKIP_LOGGER_PREFIXES = ("plugin.observe", "observe.", "plugins.observe")
_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+\b")
_MESSAGE_LIMIT = 500
_TRACEBACK_LIMIT = 4000


class GlobalErrorCollector:
    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._handler = _GlobalErrorLogHandler(self)
        self._old_sys_excepthook: Callable[..., object] | None = None
        self._old_threading_excepthook: Callable[..., object] | None = None
        self._old_asyncio_handler: Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        logging.getLogger().addHandler(self._handler)
        self._old_sys_excepthook = sys.excepthook
        sys.excepthook = self._handle_sys_exception
        self._old_threading_excepthook = threading.excepthook
        threading.excepthook = self._handle_thread_exception
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        if self._loop is not None:
            self._old_asyncio_handler = self._loop.get_exception_handler()
            self._loop.set_exception_handler(self._handle_asyncio_exception)

    def close(self) -> None:
        if not self._installed:
            return
        self._installed = False
        logging.getLogger().removeHandler(self._handler)
        if self._old_sys_excepthook is not None:
            sys.excepthook = self._old_sys_excepthook
        if self._old_threading_excepthook is not None:
            threading.excepthook = self._old_threading_excepthook
        if self._loop is not None:
            self._loop.set_exception_handler(self._old_asyncio_handler)

    def emit_log_record(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR or _skip_logger(record.name):
            return
        exc_type: type[BaseException] | None = None
        exc_value: BaseException | None = None
        exc_tb: TracebackType | None = None
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
        message = record.getMessage()
        self._emit(
            source="log",
            logger_name=record.name,
            level=record.levelname,
            message=message,
            exc_type=exc_type,
            exc_value=exc_value,
            exc_tb=exc_tb,
        )

    def _handle_sys_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        self._emit(
            source="uncaught",
            logger_name="sys.excepthook",
            level="CRITICAL",
            message=str(exc_value),
            exc_type=exc_type,
            exc_value=exc_value,
            exc_tb=exc_tb,
        )
        if self._old_sys_excepthook is not None:
            self._old_sys_excepthook(exc_type, exc_value, exc_tb)

    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        self._emit(
            source="thread",
            logger_name=str(getattr(args.thread, "name", "") or "threading.excepthook"),
            level="CRITICAL",
            message=str(args.exc_value),
            exc_type=args.exc_type,
            exc_value=args.exc_value,
            exc_tb=args.exc_traceback,
        )
        if self._old_threading_excepthook is not None:
            self._old_threading_excepthook(args)

    def _handle_asyncio_exception(
        self,
        loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        exc = context.get("exception")
        message = str(context.get("message") or exc or "asyncio exception")
        self._emit(
            source="asyncio",
            logger_name="asyncio",
            level="ERROR",
            message=message,
            exc_type=type(exc) if isinstance(exc, BaseException) else None,
            exc_value=exc if isinstance(exc, BaseException) else None,
            exc_tb=exc.__traceback__ if isinstance(exc, BaseException) else None,
        )
        if self._old_asyncio_handler is not None:
            self._old_asyncio_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    def _emit(
        self,
        *,
        source: str,
        logger_name: str,
        level: str,
        message: str,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        error_type = exc_type.__name__ if exc_type is not None else "LogError"
        traceback_text = _format_traceback(exc_type, exc_value, exc_tb)
        fingerprint = _fingerprint(error_type, message, traceback_text)
        diag = current_diagnostic_context()
        session = diag["session"]
        self._writer.emit(
            GlobalErrorTrace(
                fingerprint=fingerprint,
                bucket=now[:13],
                source=source,
                logger_name=logger_name,
                error_type=error_type,
                message=message[:_MESSAGE_LIMIT],
                traceback_text=traceback_text[:_TRACEBACK_LIMIT],
                level=level,
                first_ts=now,
                last_ts=now,
                count=1,
                session_keys=[session] if session else [],
                flow=diag["flow"],
                phase=diag["phase"],
                turn=diag["turn"],
                tick=diag["tick"],
            )
        )


class _GlobalErrorLogHandler(logging.Handler):
    def __init__(self, collector: GlobalErrorCollector) -> None:
        super().__init__(level=logging.ERROR)
        self._collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        self._collector.emit_log_record(record)


def _skip_logger(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _SKIP_LOGGER_PREFIXES)


def _format_traceback(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    exc_tb: TracebackType | None,
) -> str:
    if exc_type is None:
        return ""
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def _fingerprint(error_type: str, message: str, traceback_text: str) -> str:
    normalized = _NUM_RE.sub("<n>", _HEX_RE.sub("<hex>", message))
    frame = _top_app_frame(traceback_text)
    raw = f"{error_type}|{normalized}|{frame}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _top_app_frame(traceback_text: str) -> str:
    for line in traceback_text.splitlines():
        if "/akasic-agent/" in line and "/plugins/observe/" not in line:
            return line.strip()
    return ""
