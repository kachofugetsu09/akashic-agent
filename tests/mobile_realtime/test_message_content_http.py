from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from infra.mobile_realtime.channel import _fit_mobile_history_payload
from infra.mobile_realtime.gateway import (
    MobileMessageContentHttpError,
    _parse_message_content_range,
    create_mobile_gateway_app,
)
from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.message_content_http import (
    MessageContentTicketError,
    MessageContentTicketIssuer,
)
from infra.mobile_realtime.storage import MobileRealtimeStorage
from session.store import SessionStore


class _Storage:
    def read_device(self, device_id: str) -> object | None:
        if device_id != "device-1":
            return None
        return SimpleNamespace(revoked_at=None)


def _issuer(now: datetime) -> MessageContentTicketIssuer:
    keyset = SimpleNamespace(
        manifest=SimpleNamespace(server_id="server-1"),
        identity_private_key=ec.generate_private_key(ec.SECP256R1()),
    )
    return MessageContentTicketIssuer(
        cast(LoadedKeyset, keyset),
        cast(MobileRealtimeStorage, _Storage()),
        clock=lambda: now,
    )


def test_ticket_binds_device_message_digest_and_length() -> None:
    now = datetime(2026, 8, 2, tzinfo=timezone.utc)
    issuer = _issuer(now)
    grant = issuer.issue(
        device_id="device-1",
        connection_epoch=7,
        session_id="mobile:test",
        message_id="mobile:test:3",
        byte_length=300_000,
        sha256="a" * 64,
    )

    verified = issuer.verify(grant.ticket)

    assert verified.device_id == "device-1"
    assert verified.connection_epoch == 7
    assert verified.session_id == "mobile:test"
    assert verified.message_id == "mobile:test:3"
    assert verified.byte_length == 300_000
    assert verified.sha256 == "a" * 64


def test_ticket_rejects_tampering() -> None:
    now = datetime(2026, 8, 2, tzinfo=timezone.utc)
    issuer = _issuer(now)
    grant = issuer.issue(
        device_id="device-1",
        connection_epoch=7,
        session_id="mobile:test",
        message_id="mobile:test:3",
        byte_length=1,
        sha256="b" * 64,
    )

    with pytest.raises(MessageContentTicketError, match="签名无效"):
        issuer.verify(f"{grant.ticket[:-1]}A")


def test_range_is_single_bounded_and_clamped() -> None:
    assert _parse_message_content_range("bytes=10-19", 100) == (10, 19)
    assert _parse_message_content_range("bytes=90-200", 100) == (90, 99)
    with pytest.raises(MobileMessageContentHttpError, match="单个 bytes Range"):
        _parse_message_content_range(None, 100)
    with pytest.raises(MobileMessageContentHttpError, match="单次下载预算"):
        _parse_message_content_range("bytes=0-262144", 300_000)


def test_http_route_returns_identity_encoded_verified_range_headers() -> None:
    content = "界🌙".encode()
    sha256 = hashlib.sha256(content).hexdigest()

    class Runtime:
        def read_message_content_http(
            self,
            *,
            ticket: str,
            range_header: str | None,
            if_range: str | None,
        ) -> tuple[bytes, int, int, int, str]:
            assert ticket == "ticket"
            assert range_header == f"bytes=0-{len(content) - 1}"
            assert if_range == f'"{sha256}"'
            return content, 0, len(content) - 1, len(content), sha256

    client = TestClient(create_mobile_gateway_app(Runtime()))  # type: ignore[arg-type]
    response = client.get(
        "/mobile/message-content/v1",
        headers={
            "Authorization": "Bearer ticket",
            "Range": f"bytes=0-{len(content) - 1}",
            "If-Range": f'"{sha256}"',
        },
    )

    assert response.status_code == 206
    assert response.content == content
    assert response.headers["content-encoding"] == "identity"
    assert response.headers["content-range"] == f"bytes 0-{len(content) - 1}/{len(content)}"
    assert response.headers["etag"] == f'"{sha256}"'
    assert response.headers["content-digest"].startswith("sha-256=:")
    assert response.headers["repr-digest"].startswith("sha-256=:")


def test_history_externalizes_unicode_content_without_losing_tool_projection() -> None:
    content = "花月🌙" * 80_000
    tool_chain = [
        {
            "reasoning_content": "先检查真实状态",
            "calls": [{"name": "shell", "status": "success", "description": "核对日志"}],
        }
    ]
    payload: dict[str, object] = {
        "items": [{"id": "m", "content": content, "tool_chain": tool_chain}],
        "total": 1,
        "page_size": 10,
        "content_ref_version": 1,
        "after_seq": -1,
        "next_after_seq": 0,
        "snapshot_max_seq": 0,
        "has_more": False,
    }

    _fit_mobile_history_payload(payload, allow_content_refs=True)

    item = payload["items"][0]  # type: ignore[index]
    assert item["content"] is None
    assert item["tool_chain"] == tool_chain
    assert item["content_ref"] == {
        "version": 1,
        "encoding": "utf-8",
        "byte_length": len(content.encode("utf-8")),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "preview": content[:512],
    }


def test_mobile_history_cursor_freezes_append_high_water(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    try:
        store.persist_session(
            "mobile:test",
            created_at="2026-08-02T00:00:00+00:00",
            updated_at="2026-08-02T00:00:00+00:00",
            last_consolidated=0,
            metadata={},
            messages=[
                {
                    "role": "user",
                    "content": "one",
                    "timestamp": "2026-08-02T00:00:00+00:00",
                    "extra": {},
                },
                {
                    "role": "assistant",
                    "content": "two",
                    "timestamp": "2026-08-02T00:00:01+00:00",
                    "extra": {},
                },
            ],
        )
        total, snapshot = store.mobile_history_snapshot("mobile:test")
        store.persist_session(
            "mobile:test",
            created_at="2026-08-02T00:00:00+00:00",
            updated_at="2026-08-02T00:00:02+00:00",
            last_consolidated=0,
            metadata={},
            messages=[
                {
                    "role": "assistant",
                    "content": "later",
                    "timestamp": "2026-08-02T00:00:02+00:00",
                    "extra": {},
                },
            ],
        )

        page = store.list_mobile_history_page(
            session_key="mobile:test",
            after_seq=-1,
            through_seq=snapshot,
            page_size=10,
        )

        assert total == 2
        assert snapshot == 1
        assert [item["content"] for item in page] == ["one", "two"]
    finally:
        store.close()
