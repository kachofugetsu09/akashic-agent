from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

import httpcore
import pytest

from infra.channels.base import AttachmentStore
from infra.mobile_realtime.remote_media import (
    PinnedNetworkBackend,
    RemoteMediaError,
    snapshot_remote_media,
)


PUBLIC_IP = "93.184.216.34"


class _RawBackendFactory:
    def __init__(self, responses: list[bytes]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, host: str, address: str) -> httpcore.AsyncNetworkBackend:
        self.calls.append((host, address))
        return httpcore.AsyncMockBackend([next(self._responses)])


async def _public_resolver(host: str, port: int) -> tuple[str, ...]:
    return (PUBLIC_IP,)


@pytest.mark.asyncio
async def test_downloads_stream_to_persistent_snapshot_with_safe_metadata(
    tmp_path: Path,
) -> None:
    factory = _RawBackendFactory(
        [
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 7\r\n"
            b"Content-Type: image/png; charset=binary\r\n"
            b"Content-Disposition: attachment; filename=\"../cat.png\"\r\n"
            b"Connection: close\r\n\r\npayload"
        ]
    )

    snapshot = await snapshot_remote_media(
        "https://media.example/original.bin",
        AttachmentStore(tmp_path / "uploads"),
        max_bytes=32,
        resolver=_public_resolver,
        backend_factory=factory,
    )

    assert snapshot.path.parent == tmp_path / "uploads"
    assert snapshot.path.read_bytes() == b"payload"
    assert snapshot.filename == "cat.png"
    assert snapshot.content_type == "image/png"
    assert snapshot.size_bytes == 7
    assert factory.calls == [("media.example", PUBLIC_IP)]
    assert not list((tmp_path / "uploads").glob("*.part"))


@pytest.mark.asyncio
async def test_revalidates_and_repins_every_redirect(tmp_path: Path) -> None:
    factory = _RawBackendFactory(
        [
            b"HTTP/1.1 302 Found\r\nLocation: https://cdn.example/cat.gif\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Type: image/gif\r\nContent-Length: 3\r\n"
            b"Connection: close\r\n\r\ngif",
        ]
    )
    resolved: list[tuple[str, int]] = []

    async def resolver(host: str, port: int) -> tuple[str, ...]:
        resolved.append((host, port))
        return (PUBLIC_IP,)

    snapshot = await snapshot_remote_media(
        "http://origin.example/start",
        AttachmentStore(tmp_path / "uploads"),
        max_bytes=32,
        resolver=resolver,
        backend_factory=factory,
    )

    assert snapshot.filename == "cat.gif"
    assert resolved == [("origin.example", 80), ("cdn.example", 443)]
    assert factory.calls == [
        ("origin.example", PUBLIC_IP),
        ("cdn.example", PUBLIC_IP),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        ("10.0.0.1",),
        ("169.254.169.254",),
        (PUBLIC_IP, "192.168.1.2"),
        ("::1",),
    ],
)
async def test_rejects_when_any_resolved_address_is_not_public(
    tmp_path: Path,
    addresses: tuple[str, ...],
) -> None:
    async def resolver(host: str, port: int) -> tuple[str, ...]:
        return addresses

    with pytest.raises(RemoteMediaError, match="非公网地址"):
        await snapshot_remote_media(
            "https://media.example/file.png",
            AttachmentStore(tmp_path / "uploads"),
            max_bytes=32,
            resolver=resolver,
            backend_factory=_RawBackendFactory([]),
        )


@pytest.mark.asyncio
async def test_oversized_stream_fails_and_removes_partial_file(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    factory = _RawBackendFactory(
        [
            b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            b"4\r\n1234\r\n4\r\n5678\r\n0\r\n\r\n"
        ]
    )

    with pytest.raises(RemoteMediaError, match="大小上限"):
        await snapshot_remote_media(
            "https://media.example/data.bin",
            AttachmentStore(upload_root),
            max_bytes=6,
            resolver=_public_resolver,
            backend_factory=factory,
        )

    assert list(upload_root.iterdir()) == []


@pytest.mark.asyncio
async def test_rejects_invalid_scheme_mime_and_redirect_limit(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "uploads")
    with pytest.raises(RemoteMediaError, match="只允许"):
        await snapshot_remote_media("file:///etc/passwd", store, max_bytes=32)

    invalid_mime = _RawBackendFactory(
        [
            b"HTTP/1.1 200 OK\r\nContent-Type: invalid\r\nContent-Length: 1\r\n"
            b"Connection: close\r\n\r\nx"
        ]
    )
    with pytest.raises(RemoteMediaError, match="Content-Type"):
        await snapshot_remote_media(
            "https://media.example/a",
            store,
            max_bytes=32,
            resolver=_public_resolver,
            backend_factory=invalid_mime,
        )


@pytest.mark.asyncio
async def test_total_deadline_and_dns_failures_are_explicit(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "uploads")

    async def slow_resolver(host: str, port: int) -> tuple[str, ...]:
        await asyncio.sleep(1)
        return (PUBLIC_IP,)

    with pytest.raises(RemoteMediaError, match="总时限"):
        await snapshot_remote_media(
            "https://media.example/a",
            AttachmentStore(tmp_path / "slow"),
            max_bytes=32,
            timeout_seconds=0.01,
            resolver=slow_resolver,
        )

    async def failed_resolver(host: str, port: int) -> tuple[str, ...]:
        raise OSError("name lookup failed")

    with pytest.raises(RemoteMediaError, match="DNS 解析失败"):
        await snapshot_remote_media(
            "https://media.example/a",
            AttachmentStore(tmp_path / "dns"),
            max_bytes=32,
            resolver=failed_resolver,
        )

    redirect = _RawBackendFactory(
        [
            b"HTTP/1.1 302 Found\r\nLocation: /again\r\nContent-Length: 0\r\n"
            b"Connection: close\r\n\r\n"
        ]
    )
    with pytest.raises(RemoteMediaError, match="重定向次数超限"):
        await snapshot_remote_media(
            "https://media.example/a",
            store,
            max_bytes=32,
            max_redirects=0,
            resolver=_public_resolver,
            backend_factory=redirect,
        )


class _CaptureBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, stream: httpcore.AsyncNetworkStream | None = None) -> None:
        self.connected: tuple[str, int] | None = None
        self._stream = stream or httpcore.AsyncMockStream([])

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connected = (host, port)
        return self._stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise AssertionError("不应连接 Unix socket")

    async def sleep(self, seconds: float) -> None:
        return None


@pytest.mark.asyncio
async def test_pinned_backend_connects_only_to_validated_address() -> None:
    delegate = _CaptureBackend()
    backend = PinnedNetworkBackend("media.example", PUBLIC_IP, delegate)

    await backend.connect_tcp("media.example", 443)

    assert delegate.connected == (PUBLIC_IP, 443)
    with pytest.raises(RemoteMediaError, match="偏离"):
        await backend.connect_tcp("rebind.example", 443)


class _TlsCaptureStream(httpcore.AsyncMockStream):
    def __init__(self) -> None:
        super().__init__(
            [
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n"
                b"Connection: close\r\n\r\n"
            ]
        )
        self.server_hostname: str | None = None

    async def start_tls(
        self,
        ssl_context: object,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.server_hostname = server_hostname
        return self


@pytest.mark.asyncio
async def test_pinning_keeps_original_hostname_for_https_tls() -> None:
    stream = _TlsCaptureStream()
    delegate = _CaptureBackend(stream)
    backend = PinnedNetworkBackend("media.example", PUBLIC_IP, delegate)
    pool = httpcore.AsyncConnectionPool(network_backend=backend)
    try:
        response = await pool.request(
            "GET",
            "https://media.example/file",
            headers=[(b"host", b"media.example")],
        )
    finally:
        await pool.aclose()

    assert response.status == 200
    assert delegate.connected == (PUBLIC_IP, 443)
    assert stream.server_hostname == "media.example"
