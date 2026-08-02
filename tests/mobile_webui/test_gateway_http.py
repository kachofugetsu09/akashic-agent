from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import WebSocket
import httpx
import pytest

from agent.config_models import MobileRealtimeConfig
from infra.mobile_realtime.gateway import (
    ActiveMobileConnection,
    MobileGatewayRuntime,
    MobileWebUiHttpError,
    create_mobile_gateway_app,
)
from infra.mobile_realtime.storage import DeviceRecord
from infra.mobile_webui.manifest import manifest_from_directory
from infra.mobile_webui.store import MobileWebUiStore


_SOURCE = {
    "source_repository": "https://github.com/example/repo",
    "source_commit": "a" * 40,
    "source_tree": "b" * 40,
    "input_digest": "c" * 64,
    "build_context_digest": "d" * 64,
    "dirty_provenance": None,
    "reproducible": True,
    "builder_identity": {
        "node_version": "v22.23.1",
        "npm_version": "10.9.0",
        "package_lock_digest": "e" * 64,
        "build_script_digest": "f" * 64,
    },
}


def _websocket_stub() -> WebSocket:
    return cast(WebSocket, object())


class _FakeKeyset:
    class Manifest:
        server_id = "server-1"

    manifest = Manifest()
    identity_private_key = ec.generate_private_key(ec.SECP256R1())


class _FakeStorage:
    def __init__(self, device: DeviceRecord) -> None:
        self.device = device

    def read_device(self, device_id: str) -> DeviceRecord | None:
        return self.device if device_id == self.device.device_id else None


def _runtime(tmp_path: Path) -> tuple[MobileGatewayRuntime, MobileWebUiStore, _FakeStorage, str, str]:
    build = tmp_path / "build"
    build.mkdir()
    (build / "mobile.html").write_bytes(b"<html>mobile</html>")
    manifest, contents = manifest_from_directory(build, **_SOURCE)
    store = MobileWebUiStore(tmp_path / "publication", server_id="server-1")
    release = store.publish(manifest, contents, stable=True, preview=False)
    device = DeviceRecord(
        "device-1",
        "pub",
        "Pixel",
        datetime.now(timezone.utc),
        None,
        ("mobile-webui-ota-v1",),
    )
    storage = _FakeStorage(device)
    runtime = MobileGatewayRuntime(
        config=MobileRealtimeConfig(),
        storage=storage,  # type: ignore[arg-type]
        pairing=object(),  # type: ignore[arg-type]
        authenticator=object(),  # type: ignore[arg-type]
        inbox=object(),  # type: ignore[arg-type]
        approvals=object(),  # type: ignore[arg-type]
        keyset=_FakeKeyset(),  # type: ignore[arg-type]
        publication=store,
    )
    runtime._connections[device.device_id] = ActiveMobileConnection(
        _websocket_stub(),
        7,
        asyncio.Lock(),
        deque(),
        True,
        None,
        device.capabilities,
    )
    assert release.stable is not None
    grant = runtime.webui_http_tickets.issue(
        device_id=device.device_id,
        connection_epoch=7,
        release=release,
        target_key=release.stable.target_key,
    )
    return runtime, store, storage, grant.ticket, release.stable.manifest_digest


@pytest.mark.asyncio
async def test_webui_http_headers_ranges_and_ticket_lifecycle(tmp_path: Path) -> None:
    runtime, store, storage, ticket, manifest_digest = _runtime(tmp_path)
    app = create_mobile_gateway_app(runtime)
    headers = {"Authorization": f"Bearer {ticket}"}
    runtime.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            manifest_response = await client.get(f"/mobile/webui/v1/manifest/{manifest_digest}", headers=headers)
            assert manifest_response.status_code == 200
            assert manifest_response.headers["cache-control"] == "no-store, no-transform"
            assert manifest_response.headers["etag"] == f'"{manifest_digest}"'
            assert manifest_response.headers["content-digest"].startswith("sha-256=:")
            assert manifest_response.headers["repr-digest"] == f"sha-256=:{base64.b64encode(bytes.fromhex(manifest_digest)).decode()}:"

            manifest = manifest_response.json()
            blob_digest = manifest["files"][0]["sha256"]
            blob_response = await client.get(f"/mobile/webui/v1/blob/{blob_digest}", headers=headers)
            assert blob_response.status_code == 200
            assert blob_response.headers["cache-control"] == "private, max-age=31536000, immutable, no-transform"
            assert blob_response.headers["etag"] == f'"{blob_digest}"'
            assert "content-range" not in blob_response.headers
            ranged = await client.get(
                f"/mobile/webui/v1/blob/{blob_digest}",
                headers={**headers, "Range": "bytes=0-3", "If-Range": f'"{blob_digest}"'},
            )
            assert ranged.status_code == 206
            assert ranged.headers["content-range"] == f"bytes 0-3/{len(blob_response.content)}"
            assert ranged.headers["repr-digest"] == f"sha-256=:{base64.b64encode(bytes.fromhex(blob_digest)).decode()}:"
            invalid_range = await client.get(
                f"/mobile/webui/v1/blob/{blob_digest}",
                headers={**headers, "Range": "bytes=999-1000"},
            )
            assert invalid_range.status_code == 416
            assert invalid_range.json()["error"]["code"] == "invalid_range"
            not_member = await client.get(f"/mobile/webui/v1/blob/{'0' * 64}", headers=headers)
            assert not_member.status_code == 404
            assert not_member.json()["error"]["code"] == "resource_not_found"

            runtime._connections["device-1"] = ActiveMobileConnection(
                _websocket_stub(),
                8,
                asyncio.Lock(),
                deque(),
                True,
                None,
                ("mobile-webui-ota-v1",),
            )
            stale_epoch = await client.get(f"/mobile/webui/v1/manifest/{manifest_digest}", headers=headers)
            assert stale_epoch.status_code == 401
            assert stale_epoch.json()["error"]["code"] == "invalid_ticket"

            runtime._connections["device-1"] = ActiveMobileConnection(
                _websocket_stub(),
                7,
                asyncio.Lock(),
                deque(),
                True,
                None,
                ("mobile-webui-ota-v1",),
            )
            fresh_release = store.get_release()
            assert fresh_release.stable is not None
            fresh_grant = runtime.webui_http_tickets.issue(
                device_id="device-1",
                connection_epoch=7,
                release=fresh_release,
                target_key=fresh_release.stable.target_key,
            )
            second_build = tmp_path / "second-build"
            second_build.mkdir()
            (second_build / "mobile.html").write_bytes(b"<html>new</html>")
            second_manifest, second_contents = manifest_from_directory(second_build, **_SOURCE)
            store.publish(second_manifest, second_contents, preview=True)
            changed = await client.get(
                f"/mobile/webui/v1/manifest/{manifest_digest}",
                headers={"Authorization": f"Bearer {fresh_grant.ticket}"},
            )
            assert changed.status_code == 409
            assert changed.json()["error"]["code"] == "target_changed"
            storage.device = replace(storage.device, revoked_at=datetime.now(timezone.utc))
            revoked = await client.get(f"/mobile/webui/v1/manifest/{manifest_digest}", headers=headers)
            assert revoked.status_code == 401
            assert revoked.json()["error"]["code"] == "invalid_ticket"
    finally:
        await runtime.stop()
        store.close()


@pytest.mark.asyncio
async def test_blob_http_rechecks_selection_after_body_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, store, _storage, ticket, manifest_digest = _runtime(tmp_path)
    runtime.start()
    try:
        manifest = store.get_manifest(manifest_digest)
        blob_digest = manifest.files[0].sha256
        second_build = tmp_path / "race-build"
        second_build.mkdir()
        (second_build / "mobile.html").write_bytes(b"<html>race</html>")
        second_manifest, second_contents = manifest_from_directory(second_build, **_SOURCE)
        original_read_bytes = Path.read_bytes
        reads = 0

        def hooked_read_bytes(path: Path) -> bytes:
            nonlocal reads
            data = original_read_bytes(path)
            if path == store.blob_path(blob_digest):
                reads += 1
                if reads == 2:
                    store.publish(second_manifest, second_contents, preview=True)
            return data

        monkeypatch.setattr(Path, "read_bytes", hooked_read_bytes)
        with pytest.raises(MobileWebUiHttpError, match="release 已变化") as error:
            runtime.read_webui_blob_http(
                ticket=ticket,
                blob_digest=blob_digest,
                range_header=None,
                if_range=None,
            )
        assert error.value.code == "target_changed"
        assert reads >= 2
    finally:
        await runtime.stop()
        store.close()


@pytest.mark.asyncio
async def test_manifest_http_rechecks_release_epoch_after_body_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, store, _storage, ticket, manifest_digest = _runtime(tmp_path)
    runtime.start()
    try:
        original_get_manifest = store.get_manifest
        switched = False

        def hooked_get_manifest(digest: str):
            nonlocal switched
            manifest = original_get_manifest(digest)
            if not switched:
                switched = True
                next_epoch = str(uuid4())
                store._db.execute("UPDATE webui_meta SET value = ? WHERE key = 'release_epoch'", (next_epoch,))
                store._db.execute(
                    "UPDATE webui_release_state SET release_epoch = ? WHERE singleton = 1",
                    (next_epoch,),
                )
            return manifest

        monkeypatch.setattr(store, "get_manifest", hooked_get_manifest)
        with pytest.raises(MobileWebUiHttpError, match="release 已变化") as error:
            runtime.read_webui_manifest_http(ticket=ticket, manifest_digest=manifest_digest)
        assert error.value.code == "target_changed"
        assert switched
    finally:
        await runtime.stop()
        store.close()
