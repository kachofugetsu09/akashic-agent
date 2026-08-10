from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pytest

from scripts.akashic_release.mobile_pair import pair_mobile


def _offer(now: datetime) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "server_id": "server-1",
        "server_application_key_fingerprint": "fingerprint",
        "server_application_public_key": "public-key",
        "lan_endpoints": ["wss://akashic.local:6323/ws"],
        "tunnel_endpoints": ["wss://mobile.huashen258.cc/ws"],
        "tls_spki_pins": ["pin"],
        "pairing_id": "pairing-1",
        "one_time_secret": "secret-must-only-exist-inside-qr",
        "expires_at": (now + timedelta(minutes=8)).isoformat(),
    }


def test_terminal_pairing_renders_qr_and_requires_matching_code(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 11, 2, 30, tzinfo=timezone.utc)
    environment = tmp_path / "runtime.env"
    environment.write_text("AKASHIC_PUBLISHED_WEB_PORT=2236\n", encoding="utf-8")
    calls: list[tuple[str, str, object]] = []
    status_values: list[dict[str, object]] = [
        {"pairing_id": "pairing-1", "status": "waiting_for_phone"},
        {
            "pairing_id": "pairing-1",
            "status": "waiting_for_desktop_confirmation",
            "device_name": "Pixel 9",
            "confirmation_code": "482913",
            "capabilities": ["stream-v1"],
        },
    ]
    statuses = iter(status_values)

    def request(method: str, url: str, payload: object) -> dict[str, object]:
        calls.append((method, url, payload))
        if url.endswith("/api/chat/mobile-pairing"):
            return _offer(now)
        if method == "GET":
            return next(statuses)
        return {"device_id": "device-1", "display_name": "Pixel 9"}

    output = StringIO()
    result = pair_mobile(
        environment,
        input_fn=lambda _prompt: "482913",
        output=output,
        request_json=request,
        sleep=lambda _seconds: None,
        now=lambda: now,
    )

    assert result == {
        "status": "paired",
        "deviceId": "device-1",
        "displayName": "Pixel 9",
    }
    assert "约 8 分钟" in output.getvalue()
    assert "服务端确认码：482913" in output.getvalue()
    assert "secret-must-only-exist-inside-qr" not in output.getvalue()
    assert all("127.0.0.1:2236" in url for _method, url, _payload in calls)
    assert calls[-1][2] == {"confirmation_code": "482913"}


def test_terminal_pairing_rejects_mismatched_confirmation_without_approval(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 11, 2, 30, tzinfo=timezone.utc)
    environment = tmp_path / "runtime.env"
    environment.write_text("AKASHIC_PUBLISHED_WEB_PORT=2236\n", encoding="utf-8")
    calls: list[str] = []

    def request(method: str, url: str, _payload: object) -> dict[str, object]:
        calls.append(method)
        if url.endswith("/api/chat/mobile-pairing"):
            return _offer(now)
        return {
            "pairing_id": "pairing-1",
            "status": "waiting_for_desktop_confirmation",
            "device_name": "Unknown phone",
            "confirmation_code": "482913",
        }

    with pytest.raises(RuntimeError, match="未批准"):
        pair_mobile(
            environment,
            input_fn=lambda _prompt: "000000",
            output=StringIO(),
            request_json=request,
            sleep=lambda _seconds: None,
            now=lambda: now,
        )

    assert calls == ["POST", "GET"]
