from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from infra.mobile_realtime.protocol import (
    AttachmentDownloadCommand,
    AuthAcceptedControl,
    CONTROL_TYPES,
    EVENT_TYPES,
    GenericControl,
    MessageSendCommand,
    PRE_AUTH_CONTROL_TYPES,
    ProtocolDecodeError,
    ReplyFrame,
    ThinkingDeltaEvent,
    frame_to_json,
    parse_frame,
)
from scripts.generate_mobile_realtime_schema import OUTPUT, build_schema

FIXTURES = Path(__file__).parent / "fixtures" / "frames-v1.json"


def test_golden_frames_round_trip() -> None:
    frames = json.loads(FIXTURES.read_text(encoding="utf-8"))
    parsed = [parse_frame(json.dumps(frame, ensure_ascii=False)) for frame in frames]

    assert isinstance(parsed[0], MessageSendCommand)
    assert isinstance(parsed[1], ReplyFrame)
    assert isinstance(parsed[4], AuthAcceptedControl)
    assert [json.loads(frame_to_json(frame)) for frame in parsed] == frames


def test_resume_rejects_ack_that_cannot_fit_sqlite_sequence() -> None:
    frame = _golden_frame(5)
    frame["payload"]["last_ack"] = (1 << 63) - 2

    with pytest.raises(ValidationError, match="less than or equal"):
        parse_frame(json.dumps(frame))


def test_message_send_rejects_mismatched_session() -> None:
    frame = _golden_frame(0)
    frame["session_id"] = "mobile:other"

    with pytest.raises(ValidationError, match="session_id 必须一致"):
        parse_frame(json.dumps(frame))


def test_attachment_download_validates_offset() -> None:
    frame = {
        "v": 1,
        "kind": "command",
        "type": "attachment.download",
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "connection_epoch": 1,
        "session_id": "mobile:test",
        "payload": {
            "attachment_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
            "offset": 131072,
        },
    }

    parsed = parse_frame(json.dumps(frame))
    assert isinstance(parsed, AttachmentDownloadCommand)
    assert parsed.payload.offset == 131072

    frame["payload"]["offset"] = -1
    with pytest.raises(ValidationError):
        parse_frame(json.dumps(frame))


def test_message_send_rejects_duplicate_or_too_many_media_refs() -> None:
    duplicate = _golden_frame(0)
    duplicate["payload"]["media_refs"] = [
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    ]
    with pytest.raises(ValidationError, match="不能重复"):
        parse_frame(json.dumps(duplicate))

    oversized = _golden_frame(0)
    oversized["payload"]["media_refs"] = [
        f"01ARZ3NDEKTSV4RRFFQ69G5FA{suffix}" for suffix in "0123456789A"
    ]
    with pytest.raises(ValidationError):
        parse_frame(json.dumps(oversized))


def test_message_send_validates_reply_identity() -> None:
    valid = _golden_frame(0)
    valid["payload"]["reply_to"] = {
        "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
    }
    parsed = parse_frame(json.dumps(valid))
    assert isinstance(parsed, MessageSendCommand)
    assert parsed.payload.reply_to is not None
    assert parsed.payload.reply_to.client_message_id == "01ARZ3NDEKTSV4RRFFQ69G5FAW"

    delivery = _golden_frame(0)
    delivery["payload"]["reply_to"] = {"delivery_id": "delivery-1"}
    parsed_delivery = parse_frame(json.dumps(delivery))
    assert isinstance(parsed_delivery, MessageSendCommand)
    assert parsed_delivery.payload.reply_to is not None
    assert parsed_delivery.payload.reply_to.delivery_id == "delivery-1"

    invalid = _golden_frame(0)
    invalid["payload"]["reply_to"] = {
        "message_id": "mobile:test:1",
        "delivery_id": "delivery-1",
    }
    with pytest.raises(ValidationError, match="只能提供一种"):
        parse_frame(json.dumps(invalid))


def test_message_send_accepts_real_client_creation_time() -> None:
    frame = _golden_frame(0)
    frame["payload"]["client_created_at"] = "2026-07-16T12:34:56+08:00"

    parsed = parse_frame(json.dumps(frame))

    assert isinstance(parsed, MessageSendCommand)
    assert parsed.payload.client_created_at == "2026-07-16T12:34:56+08:00"


def test_message_send_validates_model_selection() -> None:
    frame = _golden_frame(0)
    frame["payload"]["model_runtime_id"] = "  model-a  "
    frame["payload"]["model_reasoning_effort"] = " high "

    parsed = parse_frame(json.dumps(frame))

    assert isinstance(parsed, MessageSendCommand)
    assert parsed.payload.model_runtime_id == "model-a"
    assert parsed.payload.model_reasoning_effort == "high"

    frame["payload"]["model_runtime_id"] = ""
    with pytest.raises(ValidationError, match="model_reasoning_effort"):
        parse_frame(json.dumps(frame))


@pytest.mark.parametrize(
    "value",
    (
        "not-a-time",
        "2026-07-16T12:34:56",
        "2026-07-16 12:34:56+08:00",
        "20260716T123456+0800",
        "2026-07-16T12:34:56,5+08:00",
    ),
)
def test_message_send_rejects_invalid_client_creation_time(value: str) -> None:
    frame = _golden_frame(0)
    frame["payload"]["client_created_at"] = value

    with pytest.raises(ValidationError, match="client_created_at"):
        parse_frame(json.dumps(frame))


def test_message_send_accepts_but_drops_legacy_reply_projection() -> None:
    frame = _golden_frame(0)
    frame["payload"]["reply_to"] = {
        "message_id": "mobile:test:1",
        "role": "assistant",
        "preview": "旧版客户端缓存的摘要",
    }

    parsed = parse_frame(json.dumps(frame))

    assert isinstance(parsed, MessageSendCommand)
    assert parsed.payload.reply_to is not None
    assert parsed.payload.reply_to.legacy_role == "assistant"
    assert parsed.payload.reply_to.legacy_preview == "旧版客户端缓存的摘要"
    dumped = parsed.payload.reply_to.model_dump(by_alias=True)
    assert "role" not in dumped
    assert "preview" not in dumped


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("v", 2),
        ("kind", "notification"),
        ("type", "unknown.command"),
        ("id", "not-a-valid-id"),
        ("connection_epoch", 0),
    ),
)
def test_command_rejects_invalid_envelope(field: str, value: object) -> None:
    frame = _golden_frame(0)
    frame[field] = value

    with pytest.raises(ValidationError):
        parse_frame(json.dumps(frame))


def test_event_rejects_unknown_type_and_missing_sequence() -> None:
    frame = _golden_frame(2)
    frame["type"] = "answer.replaced"
    frame.pop("event_seq")

    with pytest.raises(ValidationError):
        parse_frame(json.dumps(frame))


def test_session_list_is_a_valid_server_event() -> None:
    frame = {
        "v": 1,
        "kind": "event",
        "type": "session.list",
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "connection_epoch": 1,
        "event_seq": 1,
        "payload": {"items": []},
    }

    assert parse_frame(json.dumps(frame)).type == "session.list"


def test_command_list_command_and_reply_are_valid() -> None:
    command = {
        "v": 1,
        "kind": "command",
        "type": "command.list",
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "connection_epoch": 1,
        "payload": {},
    }
    reply = {
        **command,
        "kind": "reply",
        "type": "command.list.ok",
        "payload": {
            "items": [
                {"command": "memorystatus", "description": "查看记忆整理状态"},
            ],
        },
    }

    assert parse_frame(json.dumps(command)).type == "command.list"
    assert parse_frame(json.dumps(reply)).type == "command.list.ok"


@pytest.mark.parametrize(
    "command_type",
    (
        "runtime.document.list",
        "runtime.document.get",
        "runtime.capability.list",
        "runtime.mcp.get",
        "scheduler.job.list",
        "scheduler.job.get",
    ),
)
def test_runtime_inspection_command_and_reply_are_valid(command_type: str) -> None:
    command = {
        "v": 1,
        "kind": "command",
        "type": command_type,
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "connection_epoch": 1,
        "payload": {},
    }
    reply = {
        **command,
        "kind": "reply",
        "type": f"{command_type}.ok",
    }

    assert parse_frame(json.dumps(command)).type == command_type
    assert parse_frame(json.dumps(reply)).type == f"{command_type}.ok"


@pytest.mark.parametrize(
    "command_type",
    (
        "plugin.ui.catalog",
        "plugin.ui.asset.get",
        "plugin.ui.query",
        "plugin.ui.cancel",
    ),
)
def test_plugin_ui_commands_are_valid(command_type: str) -> None:
    frame = {
        "v": 1,
        "kind": "command",
        "type": command_type,
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "connection_epoch": 1,
        "payload": {},
    }

    assert parse_frame(json.dumps(frame)).type == command_type


@pytest.mark.parametrize(
    "command_type", ("plugin.ui.list", "plugin.ui.asset", "plugin.ui.call")
)
def test_plugin_ui_v1_commands_are_rejected(command_type: str) -> None:
    frame = {
        "v": 1,
        "kind": "command",
        "type": command_type,
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "connection_epoch": 1,
        "payload": {},
    }

    with pytest.raises(ValueError):
        parse_frame(json.dumps(frame))


def test_delta_process_block_fields_must_appear_together() -> None:
    frame = _golden_frame(2)
    frame["type"] = "react.thinking.delta"
    frame["payload"] = {
        "delta": "思考中",
        "block_id": "thinking:turn-1:0",
        "ordinal": 0,
        "control_turn_id": "turn:logical-1",
    }
    parsed = parse_frame(json.dumps(frame))
    assert isinstance(parsed, ThinkingDeltaEvent)
    assert parsed.payload.block_id == "thinking:turn-1:0"
    assert parsed.payload.control_turn_id == "turn:logical-1"

    frame["payload"].pop("ordinal")
    with pytest.raises(ValidationError, match="必须同时出现"):
        parse_frame(json.dumps(frame))


@pytest.mark.parametrize(
    "reply_type",
    ("message.send", "unknown.ok", "message.send.done"),
)
def test_reply_requires_known_command_suffix(reply_type: str) -> None:
    frame = _golden_frame(1)
    frame["type"] = reply_type

    with pytest.raises(ValidationError, match="reply type"):
        parse_frame(json.dumps(frame))


def test_plugin_ui_catalog_accepts_not_modified_reply() -> None:
    frame = _golden_frame(1)
    frame["type"] = "plugin.ui.catalog.not_modified"
    frame["payload"] = {"catalog_revision": "a" * 64}

    parsed = parse_frame(json.dumps(frame))

    assert isinstance(parsed, ReplyFrame)
    assert parsed.type == "plugin.ui.catalog.not_modified"


def test_auth_accepted_rejects_epoch_mismatch() -> None:
    frame = _golden_frame(4)
    frame["payload"]["connection_epoch"] = 8

    with pytest.raises(ValidationError, match="connection_epoch 必须一致"):
        parse_frame(json.dumps(frame))


def test_pair_claim_is_valid_before_authentication() -> None:
    frame = parse_frame(
        '{"v":1,"kind":"control","type":"pair.claim","payload":{"pairing_id":"p1"}}'
    )

    assert isinstance(frame, GenericControl)
    assert frame.connection_epoch is None
    assert frame.type in PRE_AUTH_CONTROL_TYPES


def test_control_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        parse_frame('{"v":1,"kind":"control","type":"auth.skipped","payload":{}}')


def test_device_revoked_is_control_only() -> None:
    assert "device.revoked" in CONTROL_TYPES
    assert "device.revoked" not in EVENT_TYPES
    control = parse_frame(
        '{"v":1,"kind":"control","type":"device.revoked",'
        '"connection_epoch":7,"payload":{"device_id":"device-1"}}'
    )
    assert isinstance(control, GenericControl)
    with pytest.raises(ValidationError):
        parse_frame(
            '{"v":1,"kind":"event","type":"device.revoked",'
            '"id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","connection_epoch":7,'
            '"event_seq":1,"payload":{"device_id":"device-1"}}'
        )


def test_plugin_ui_changed_is_authenticated_connection_control() -> None:
    frame = parse_frame(
        '{"v":1,"kind":"control","type":"plugin.ui.changed",'
        '"connection_epoch":7,"payload":{}}'
    )

    assert frame.type == "plugin.ui.changed"
    assert frame.connection_epoch == 7


def test_decoder_rejects_ambiguous_or_non_object_json() -> None:
    with pytest.raises(ProtocolDecodeError, match="重复字段"):
        parse_frame('{"v":1,"v":1}')
    with pytest.raises(ProtocolDecodeError, match="顶层必须是 object"):
        parse_frame("[]")
    with pytest.raises(ProtocolDecodeError, match="非标准常量"):
        parse_frame('{"value":NaN}')


def test_models_reject_unknown_fields() -> None:
    frame = _golden_frame(3)
    frame["unexpected"] = True

    with pytest.raises(ValidationError):
        parse_frame(json.dumps(frame))


def test_generated_schema_matches_checked_in_file() -> None:
    encoded = (
        json.dumps(build_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    assert OUTPUT.read_text(encoding="utf-8") == encoded


def _golden_frame(index: int) -> dict[str, Any]:
    frames = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return frames[index]
