from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import pytest

from bus.events import InboundMessage
from agent.core.passive_turn import _turn_log_id
from core.common.diagnostic_log import AkashicJsonFormatter
from core.common.diagnostic_log import configure_logging
from core.common.diagnostic_log import diagnostic_context
from core.common.diagnostic_log import diagnostic_line
from core.common.diagnostic_log import log_event


def test_json_logging_emits_joinable_bounded_event(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AKASHIC_LOG_FORMAT", "json")
    monkeypatch.setenv("AKASHIC_SERVICE_NAME", "akashic-test")
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-1")
    configure_logging()

    with diagnostic_context(
        session="session-1",
        turn="turn-1",
        request_id="request-1",
    ):
        log_event(
            logging.getLogger("test.observability"),
            logging.INFO,
            "test.completed",
            content_fp="content-123",
            duration_ms=12,
            outcome="completed",
            message="Authorization: Bearer-secret token=private-value",
        )

    document = json.loads(capsys.readouterr().err)
    assert datetime.fromisoformat(document["timestamp"]).utcoffset() == timedelta(0)
    assert document["service"] == "akashic-test"
    assert document["event"] == "test.completed"
    assert document["session"] == "session-1"
    assert document["turn"] == "turn-1"
    assert document["request_id"] == "request-1"
    assert document["boot_id"] == "boot-1"
    assert document["duration_ms"] == 12
    assert document["content_fp"] == "content-123"
    assert "Bearer-secret" not in document["message"]
    assert "private-value" not in document["message"]


def test_json_logging_uses_library_formatter_and_drops_arbitrary_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AKASHIC_LOG_FORMAT", "json")
    configure_logging()

    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, AkashicJsonFormatter)

    logging.getLogger("test.observability").info(
        "bounded",
        extra={"arbitrary_payload": "must not be logged"},
    )

    document = json.loads(capsys.readouterr().err)
    assert "arbitrary_payload" not in document


def test_structured_logging_rejects_unowned_fields() -> None:
    with pytest.raises(ValueError, match="未知结构化日志字段"):
        log_event(
            logging.getLogger("test.observability"),
            logging.INFO,
            "test.invalid",
            arbitrary_payload="must not be logged",
        )


def test_json_formatter_promotes_existing_diagnostic_events(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AKASHIC_LOG_FORMAT", "json")
    configure_logging()

    logging.getLogger("test.legacy").info(
        diagnostic_line("PassiveTurnPipeline.run", event="phase_error")
    )

    document = json.loads(capsys.readouterr().err)
    assert document["event"] == "phase_error"
    assert document["operation"] == "PassiveTurnPipeline.run"


def test_turn_log_id_prefers_persisted_control_turn() -> None:
    message = InboundMessage(
        channel="cli",
        sender="owner",
        chat_id="chat",
        content="hello",
        metadata={"turnId": "turn-persisted"},
    )

    assert _turn_log_id("cli:chat", message) == "turn-persisted"


def test_turn_log_id_marks_pre_persistence_fallback() -> None:
    message = InboundMessage(
        channel="cli",
        sender="owner",
        chat_id="chat",
        content="hello",
    )

    assert _turn_log_id("cli:chat", message).startswith("local-")
