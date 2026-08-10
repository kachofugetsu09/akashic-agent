from __future__ import annotations

import json
from typing import Any

from google.protobuf.wrappers_pb2 import BytesValue

PROTOCOL_MAJOR = 1
SERVICE_NAME = "akashic.host.v1.HostBridge"


def encode_message(payload: dict[str, Any]) -> BytesValue:
    """Encode one versioned bridge payload into its protobuf envelope."""

    document = {"protocolMajor": PROTOCOL_MAJOR, **payload}
    return BytesValue(
        value=json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def decode_message(message: BytesValue) -> dict[str, Any]:
    """Decode and validate one bridge payload at the RPC boundary."""

    document = json.loads(message.value.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Host Bridge payload 必须是 object")
    if document.get("protocolMajor") != PROTOCOL_MAJOR:
        raise ValueError(
            "Host Bridge protocol major 不匹配: "
            f"expected={PROTOCOL_MAJOR} actual={document.get('protocolMajor')}"
        )
    return document


def serialize_message(message: BytesValue) -> bytes:
    return message.SerializeToString()


def deserialize_message(payload: bytes) -> BytesValue:
    message = BytesValue()
    message.ParseFromString(payload)
    return message
