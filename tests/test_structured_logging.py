from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import TurnCommitted
from agent.core.passive_turn import _turn_log_id
from core.common.diagnostic_log import AkashicJsonFormatter
from core.common.diagnostic_log import configure_logging
from core.common.diagnostic_log import diagnostic_context
from core.common.diagnostic_log import diagnostic_line
from core.common.diagnostic_log import log_event
from core.common.diagnostic_log import turn_milestone
from session.store import SessionStore


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


def test_turn_milestone_text_message_contains_full_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="test.observability"):
        turn_milestone(
            logging.getLogger("test.observability"),
            "tl:send.received",
            session_id="session-1",
            turn_id="turn-1",
            client_message_id="client-1",
            duration_ms=12.3,
            counts="n=1",
            outcome="accepted",
        )

    message = caplog.records[0].getMessage()
    assert "event=tl:send.received" in message
    assert "session_id=session-1" in message
    assert "turn_id=turn-1" in message
    assert "client_message_id=client-1" in message
    assert "duration_ms=12.3" in message
    assert "origin=monotonic" in message
    assert "outcome=accepted" in message
    assert "counts=n=1" in message
    assert "request_id=" not in message
    assert "session=" not in message
    assert "turn=" not in message


def test_turn_milestone_marks_missing_fields_without_fabricating_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="test.observability"):
        turn_milestone(
            logging.getLogger("test.observability"),
            "tl:turn.started",
            session_id="session-1",
        )

    message = caplog.records[0].getMessage()
    assert "turn_id=missing" in message
    assert "client_message_id=missing" in message
    assert "duration_ms=missing" in message
    assert "origin=missing" in message
    assert "outcome=missing" in message
    assert "counts=missing" in message
    assert "duration_ms=0" not in message


def test_turn_milestone_json_extra_uses_same_field_names(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AKASHIC_LOG_FORMAT", "json")
    configure_logging()

    turn_milestone(
        logging.getLogger("test.observability"),
        "tl:send.ack",
        session_id="session-1",
        turn_id="turn-1",
        client_message_id="client-1",
        duration_ms=12.34,
        counts="n=1",
        outcome="accepted",
    )

    document = json.loads(capsys.readouterr().err)
    assert document["event"] == "tl:send.ack"
    assert document["session_id"] == "session-1"
    assert document["turn_id"] == "turn-1"
    assert document["client_message_id"] == "client-1"
    assert document["duration_ms"] == 12.3
    assert document["origin"] == "monotonic"
    assert document["outcome"] == "accepted"
    assert document["counts"] == "n=1"
    assert "request_id" not in document
    assert "session" not in document
    assert "turn" not in document
    assert "duration_ms=12.3" in document["message"]
    assert "client_message_id=client-1" in document["message"]


def test_reply_sent_json_preserves_epoch_integer_and_replay_boolean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AKASHIC_LOG_FORMAT", "json")
    configure_logging()

    for replayed in (False, True):
        turn_milestone(
            logging.getLogger("test.observability"),
            "tl:send.reply_sent",
            session_id="mobile:session-1",
            client_message_id="client-1",
            duration_ms=1.25,
            outcome="receipt_replayed" if replayed else "sent",
            device_id="device-1",
            connection_epoch=7,
            reply_type="message.send.ok",
            receipt_replayed=replayed,
        )

    documents = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]
    assert len(documents) == 2
    assert [document["connection_epoch"] for document in documents] == [7, 7]
    assert [document["receipt_replayed"] for document in documents] == [
        False,
        True,
    ]
    assert all(isinstance(document["connection_epoch"], int) for document in documents)
    assert all(isinstance(document["receipt_replayed"], bool) for document in documents)


@pytest.mark.asyncio
async def test_control_terminal_milestone_uses_inbound_metadata_client_message_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()

    class _Loop:
        async def process_direct_message(
            self,
            _content: str,
            **kwargs: object,
        ) -> OutboundMessage:
            turn_id = str(kwargs["turn_id"])
            await bus.fanout(
                TurnCommitted(
                    session_key="mobile:one",
                    channel="mobile",
                    chat_id="one",
                    input_message="hello",
                    persisted_user_message="hello",
                    assistant_response="done",
                    tools_used=[],
                    turn_id=turn_id,
                )
            )
            return OutboundMessage("mobile", "one", "done")

    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        from bootstrap.control_execution import execute_control_turn

        return await execute_control_turn(cast(Any, _Loop()), bus, request)

    runtime = ConversationRuntime(store, execute)
    with caplog.at_level(logging.INFO, logger="agent.control.runtime"):
        result = await (
            await runtime.start_turn(
                TurnRequest(
                    "mobile:one",
                    "hello",
                    {
                        "channel": "mobile",
                        "chatId": "one",
                        "runtime": "latest",
                        "inboundMetadata": {"client_message_id": "client-1"},
                    },
                )
            )
        ).result()
    assert result.status.value == "completed"
    terminal = next(
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:turn.terminal"
    )
    assert terminal.akashic_fields["session_id"] == "mobile:one"
    assert terminal.akashic_fields["client_message_id"] == "client-1"
    assert terminal.akashic_fields["outcome"] == "completed"
    assert "session_id=mobile:one" in terminal.getMessage()
    assert "client_message_id=client-1" in terminal.getMessage()
    assert "outcome=completed" in terminal.getMessage()
    await runtime.shutdown()
    await bus.aclose()
    store.close()


@pytest.mark.asyncio
async def test_control_terminal_fails_loud_on_malformed_client_message_id(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        raise AssertionError("结构不符的 turn 不应执行")

    runtime = ConversationRuntime(store, execute)
    with pytest.raises(ValueError, match="client_message_id"):
        await runtime.start_turn(
            TurnRequest(
                "mobile:one",
                "hello",
                {
                    "channel": "mobile",
                    "chatId": "one",
                    "inboundMetadata": {"client_message_id": 123},
                },
            )
        )
    await runtime.shutdown()
    await bus.aclose()
    store.close()
