"""Long exception records retain the cause while keeping secrets out of JSON."""
from __future__ import annotations

import json
import logging

import pytest

from core.common.diagnostic_log import AkashicJsonFormatter


def format_error(error: BaseException) -> dict[str, object]:
    """Exercise the configured JSON formatter with a real exception chain."""
    formatter = AkashicJsonFormatter(
        ("levelname", "name", "message", "process"),
        rename_fields={"levelname": "level", "name": "logger", "process": "pid", "exc_info": "exception"},
    )
    record = logging.LogRecord("test", logging.ERROR, __file__, 1, "operation failed", (),
                               (type(error), error, error.__traceback__))
    return json.loads(formatter.format(record))


@pytest.mark.parametrize("grouped", [False, True])
def test_long_exception_keeps_final_error_and_redacts_both_ends(grouped: bool) -> None:
    """Nested failures must retain the final cause after bounded redaction."""
    cause: Exception = ValueError("SOURCE password=head-secret")
    for index in range(80):
        try:
            raise RuntimeError(f"provider frame {index} " + "detail " * 20) from cause
        except RuntimeError as error:
            cause = error
    final = ValueError("FINAL_CAUSE api_key=tail-secret")
    try:
        if grouped:
            raise ExceptionGroup("task failure", [final]) from cause
        raise final from cause
    except Exception as error:
        document = format_error(error)
    text = str(document["exception"])
    assert "ValueError: SOURCE password=[REDACTED]" in text
    assert "ValueError: FINAL_CAUSE api_key=[REDACTED]" in text
    assert "head-secret" not in text and "tail-secret" not in text
    assert "[truncated]" in text
    assert len(text) <= 4096
    if grouped:
        assert "ExceptionGroup: task failure" in text


def test_short_exception_keeps_full_traceback() -> None:
    try:
        raise ValueError("short failure")
    except ValueError as error:
        document = format_error(error)
    assert "Traceback (most recent call last)" in str(document["exception"])
    assert "ValueError: short failure" in str(document["exception"])
    assert "[truncated]" not in str(document["exception"])
