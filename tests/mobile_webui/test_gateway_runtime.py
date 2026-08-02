from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from fastapi import WebSocket
import pytest
from pydantic import ValidationError

from infra.mobile_realtime.gateway import ActiveMobileConnection, MobileGatewayRuntime
from infra.mobile_realtime.storage import DeviceRecord
from infra.mobile_realtime.protocol import parse_frame
from infra.mobile_webui.manifest import WebUiManifest, manifest_from_directory
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


def _manifest(root: Path, text: bytes) -> tuple[WebUiManifest, dict[str, bytes]]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mobile.html"
    path.write_bytes(text)
    manifest, contents = manifest_from_directory(root, **_SOURCE)
    return manifest, contents


def _websocket_stub(value: object) -> WebSocket:
    return cast(WebSocket, value)


def _watcher_runtime(store: MobileWebUiStore) -> MobileGatewayRuntime:
    runtime = object.__new__(MobileGatewayRuntime)
    runtime.publication = store
    runtime._publication_monitor_task = None
    runtime._publication_selection_digest = store.get_release_light().selection_digest
    runtime._connections = {}
    runtime._delivery_lock = asyncio.Lock()
    return runtime


@pytest.mark.asyncio
async def test_publication_watcher_deduplicates_and_filters_capability(tmp_path: Path) -> None:
    first, first_contents = _manifest(tmp_path / "first", b"first")
    second, second_contents = _manifest(tmp_path / "second", b"second")
    third, third_contents = _manifest(tmp_path / "third", b"third")
    store = MobileWebUiStore(tmp_path / "store", server_id="server-1")
    runtime = None
    try:
        store.publish(first, first_contents, stable=True, preview=False)
        runtime = _watcher_runtime(store)
        sent: list[tuple[str, str, str, int]] = []

        async def send_control(*, control_type: str, payload: dict[str, object], device_id: str, connection_epoch: int) -> None:
            sent.append((control_type, str(payload["selection_digest"]), device_id, connection_epoch))

        runtime.publish_connection_control = send_control
        runtime._connections = {
            "enabled": ActiveMobileConnection(_websocket_stub(object()), 7, asyncio.Lock(), deque(), True, None, ("mobile-webui-ota-v1",)),
            "disabled": ActiveMobileConnection(_websocket_stub(object()), 8, asyncio.Lock(), deque(), True, None, ("other",)),
            "not-ready": ActiveMobileConnection(_websocket_stub(object()), 9, asyncio.Lock(), deque(), False, None, ("mobile-webui-ota-v1",)),
        }
        runtime.start()
        runtime.start()
        await asyncio.sleep(0.05)
        assert sent == []

        second_release = store.publish(second, second_contents, preview=True)
        await asyncio.sleep(0.65)
        assert sent == [("mobile.webui.release.changed", second_release.selection_digest, "enabled", 7)]
        await asyncio.sleep(0.6)
        assert len(sent) == 1

        store.publish(second, second_contents, preview=True)
        await asyncio.sleep(0.6)
        assert len(sent) == 1

        third_release = store.publish(third, third_contents, preview=True)
        await asyncio.sleep(0.6)
        assert sent[-1] == ("mobile.webui.release.changed", third_release.selection_digest, "enabled", 7)
        assert len(sent) == 2
    finally:
        if runtime is not None:
            await runtime.stop()
        store.close()


@pytest.mark.asyncio
async def test_control_delivery_rejects_replaced_epoch_and_evicts_failed_socket(tmp_path: Path) -> None:
    manifest, contents = _manifest(tmp_path / "build", b"content")
    store = MobileWebUiStore(tmp_path / "store", server_id="server-1")
    runtime = None
    try:
        store.publish(manifest, contents, stable=True, preview=False)
        runtime = _watcher_runtime(store)

        class Socket:
            def __init__(self, *, fail: bool = False) -> None:
                self.fail = fail
                self.sent: list[str] = []
                self.closed = False

            async def send_text(self, value: str) -> None:
                if self.fail:
                    raise RuntimeError("closed")
                self.sent.append(value)

            async def close(self, **_kwargs: object) -> None:
                self.closed = True

        old_socket = Socket()
        old = ActiveMobileConnection(_websocket_stub(old_socket), 7, asyncio.Lock(), deque(), True, None, ("mobile-webui-ota-v1",))
        replacement = ActiveMobileConnection(_websocket_stub(Socket()), 8, asyncio.Lock(), deque(), True, None, ("mobile-webui-ota-v1",))
        runtime._connections = {"device": old}
        runtime._connections["device"] = replacement
        await runtime.publish_connection_control(
            control_type="mobile.webui.release.changed",
            payload={"server_id": "server-1", "selection_digest": "a" * 64},
            device_id="device",
            connection_epoch=7,
        )
        assert old_socket.sent == []

        failing_socket = Socket(fail=True)
        failing = ActiveMobileConnection(_websocket_stub(failing_socket), 9, asyncio.Lock(), deque(), True, None, ("mobile-webui-ota-v1",))
        runtime._connections["device"] = failing
        await runtime.publish_connection_control(
            control_type="mobile.webui.release.changed",
            payload={"server_id": "server-1", "selection_digest": "b" * 64},
            device_id="device",
            connection_epoch=9,
        )
        await asyncio.sleep(0)
        assert "device" not in runtime._connections
        assert failing_socket.closed
    finally:
        if runtime is not None:
            await runtime.stop()
        store.close()


@pytest.mark.asyncio
async def test_revoke_device_commits_offline_and_notifies_active_connection() -> None:
    device = DeviceRecord("device", "pub", "Pixel", datetime.now(timezone.utc), None, ())

    class Storage:
        def __init__(self) -> None:
            self.device = device

        def revoke_device(self, device_id: str, *, revoked_at: datetime) -> DeviceRecord:
            assert device_id == self.device.device_id
            if self.device.revoked_at is None:
                self.device = replace(self.device, revoked_at=revoked_at)
            return self.device

    class Socket:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.sent: list[str] = []
            self.closed: list[dict[str, object]] = []

        async def send_text(self, value: str) -> None:
            if self.fail:
                raise RuntimeError("closed")
            self.sent.append(value)

        async def close(self, **kwargs: object) -> None:
            self.closed.append(kwargs)

    runtime = object.__new__(MobileGatewayRuntime)
    runtime.storage = Storage()
    runtime._delivery_lock = asyncio.Lock()
    socket = Socket()
    runtime._connections = {
        device.device_id: ActiveMobileConnection(_websocket_stub(socket), 7, asyncio.Lock(), deque(), True, None, ())
    }
    revoked = await runtime.revoke_device(device.device_id)
    assert revoked.revoked_at is not None
    assert runtime.storage.device.revoked_at == revoked.revoked_at
    assert device.device_id not in runtime._connections
    assert socket.closed == [{"code": 4403, "reason": "设备已撤销"}]
    frame = parse_frame(socket.sent[0])
    assert frame.kind == "control"
    assert frame.type == "device.revoked"
    assert frame.connection_epoch == 7

    offline = await runtime.revoke_device("device")
    assert offline.revoked_at == revoked.revoked_at

    failing_socket = Socket(fail=True)
    runtime._connections[device.device_id] = ActiveMobileConnection(
        _websocket_stub(failing_socket),
        8,
        asyncio.Lock(),
        deque(),
        True,
        None,
        (),
    )
    failed = await runtime.revoke_device(device.device_id)
    assert failed.revoked_at == revoked.revoked_at
    assert device.device_id not in runtime._connections
    assert failing_socket.closed == [{"code": 4403, "reason": "设备已撤销"}]


def test_device_revoked_cannot_enter_durable_event_enqueue() -> None:
    runtime = object.__new__(MobileGatewayRuntime)

    class Inbox:
        def enqueue(self, **_kwargs: object) -> None:
            raise AssertionError("invalid durable event must fail before inbox write")

    runtime.inbox = Inbox()
    with pytest.raises(ValidationError):
        runtime._enqueue_event(
            device_id="device",
            event_type="device.revoked",
            payload={"device_id": "device"},
        )
