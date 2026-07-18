from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import mimetypes
import os
import re
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpcore

from infra.channels.base import AttachmentStore


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_CONTENT_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$"
)
_SAFE_SUFFIX_PATTERN = re.compile(r"^\.[A-Za-z0-9]{1,12}$")
_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_DEFAULT_FILENAME = "remote-media"


class RemoteMediaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RemoteMediaSnapshot:
    path: Path
    filename: str
    content_type: str
    size_bytes: int
    sha256: str


AddressResolver = Callable[[str, int], Awaitable[tuple[str, ...]]]
BackendFactory = Callable[[str, str], httpcore.AsyncNetworkBackend]


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """把单次请求固定到已校验地址，同时保留原主机名供 TLS 校验。"""

    def __init__(
        self,
        expected_host: str,
        pinned_address: str,
        delegate: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._expected_host = expected_host
        self._pinned_address = pinned_address
        self._delegate = delegate or cast(
            httpcore.AsyncNetworkBackend,
            httpcore.AnyIOBackend(),
        )

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if host != self._expected_host:
            raise RemoteMediaError(f"连接主机偏离已校验目标: {host}")
        return await self._delegate.connect_tcp(
            self._pinned_address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise RemoteMediaError("远程媒体不允许 Unix socket")

    async def sleep(self, seconds: float) -> None:
        _ = await self._delegate.sleep(seconds)


async def snapshot_remote_media(
    url: str,
    attachment_store: AttachmentStore,
    *,
    max_bytes: int,
    max_redirects: int = 4,
    timeout_seconds: float = 15.0,
    resolver: AddressResolver | None = None,
    backend_factory: BackendFactory | None = None,
) -> RemoteMediaSnapshot:
    """安全下载远程媒体，并原子落盘为持久快照。"""

    if max_bytes <= 0:
        raise ValueError("max_bytes 必须大于 0")
    if max_redirects < 0:
        raise ValueError("max_redirects 不能小于 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")
    resolve = resolver or _resolve_addresses
    create_backend = backend_factory or _create_pinned_backend

    # 1. 总时限覆盖 DNS、重定向和整个响应体，避免慢速滴流占用任务
    try:
        async with asyncio.timeout(timeout_seconds):
            return await _download_remote_media(
                url=url,
                attachment_store=attachment_store,
                max_bytes=max_bytes,
                max_redirects=max_redirects,
                timeout_seconds=timeout_seconds,
                resolve=resolve,
                create_backend=create_backend,
            )
    except TimeoutError as error:
        raise RemoteMediaError("远程媒体下载超过总时限") from error


async def _download_remote_media(
    *,
    url: str,
    attachment_store: AttachmentStore,
    max_bytes: int,
    max_redirects: int,
    timeout_seconds: float,
    resolve: AddressResolver,
    create_backend: BackendFactory,
) -> RemoteMediaSnapshot:
    """逐跳校验并下载一个远程媒体响应。"""

    # 1. 每一跳重新解析、校验并固定连接地址
    current_url = _normalize_url(url)
    for redirect_count in range(max_redirects + 1):
        parts = urlsplit(current_url)
        host = _ascii_hostname(parts.hostname)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            addresses = await resolve(host, port)
        except OSError as error:
            raise RemoteMediaError(f"远程媒体 DNS 解析失败: {error}") from error
        pinned_address = _require_public_addresses(addresses)[0]
        backend = create_backend(host, pinned_address)
        pool = httpcore.AsyncConnectionPool(
            network_backend=backend,
            retries=0,
            max_connections=1,
            max_keepalive_connections=0,
        )
        try:
            async with pool.stream(
                "GET",
                current_url,
                headers=[
                    (b"host", _host_header(host, port, parts.scheme).encode("ascii")),
                    (b"accept", b"*/*"),
                    (b"user-agent", b"Akasic-Mobile-Media/1"),
                    (b"connection", b"close"),
                ],
                extensions={
                    "timeout": {
                        "connect": timeout_seconds,
                        "read": timeout_seconds,
                        "write": timeout_seconds,
                        "pool": timeout_seconds,
                    }
                },
            ) as response:
                if response.status in _REDIRECT_STATUSES:
                    if redirect_count == max_redirects:
                        raise RemoteMediaError("远程媒体重定向次数超限")
                    location = _single_header(response.headers, b"location", required=True)
                    current_url = _normalize_url(urljoin(current_url, location))
                    continue
                if response.status != 200:
                    raise RemoteMediaError(f"远程媒体响应状态无效: {response.status}")
                return await _persist_response(
                    response,
                    current_url,
                    attachment_store,
                    max_bytes=max_bytes,
                )
        except (
            httpcore.NetworkError,
            httpcore.ProtocolError,
            httpcore.TimeoutException,
        ) as error:
            raise RemoteMediaError(f"远程媒体下载失败: {error}") from error
        finally:
            await pool.aclose()
    raise AssertionError("重定向循环必须在限定次数内结束")


async def _persist_response(
    response: httpcore.Response,
    source_url: str,
    attachment_store: AttachmentStore,
    *,
    max_bytes: int,
) -> RemoteMediaSnapshot:
    """流式校验响应，并把完整内容原子提交到持久目录。"""

    # 1. 在 HTTP 边界校验长度、类型和展示文件名
    declared_length = _content_length(response.headers)
    if declared_length is not None and declared_length > max_bytes:
        raise RemoteMediaError(f"远程媒体超过大小上限: {max_bytes}")
    content_type = _content_type(response.headers)
    filename = _response_filename(response.headers, source_url)
    suffix = Path(filename).suffix
    if not _SAFE_SUFFIX_PATTERN.fullmatch(suffix):
        suffix = mimetypes.guess_extension(content_type) or ""
    if not _SAFE_SUFFIX_PATTERN.fullmatch(suffix):
        suffix = ""
    final_path = attachment_store.create_persistent_path("mobile_remote_", suffix)
    partial_path = final_path.with_name(f"{final_path.name}.part")

    # 2. 流式写入并同时实施实际字节上限
    size_bytes = 0
    digest = hashlib.sha256()
    try:
        with partial_path.open("xb") as stream:
            async for chunk in response.aiter_stream():
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise RemoteMediaError(f"远程媒体超过大小上限: {max_bytes}")
                _ = stream.write(chunk)
                digest.update(chunk)
            if size_bytes == 0:
                raise RemoteMediaError("远程媒体内容为空")
            if declared_length is not None and size_bytes != declared_length:
                raise RemoteMediaError(
                    f"远程媒体长度不一致: expected={declared_length} actual={size_bytes}"
                )
            stream.flush()
            os.fsync(stream.fileno())

        # 3. 原子提交文件并同步目录元数据
        os.replace(partial_path, final_path)
        directory_fd = os.open(final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return RemoteMediaSnapshot(
            path=final_path,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )
    except BaseException:
        partial_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


async def _resolve_addresses(host: str, port: int) -> tuple[str, ...]:
    """解析目标的全部 TCP 地址，供调用方统一执行公网校验。"""

    records = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    addresses: list[str] = []
    for record in records:
        address = record[4][0]
        if not isinstance(address, str):
            raise RemoteMediaError(f"DNS 返回了非文本地址: {address!r}")
        addresses.append(address)
    return tuple(dict.fromkeys(addresses))


def _create_pinned_backend(host: str, address: str) -> httpcore.AsyncNetworkBackend:
    return PinnedNetworkBackend(host, address)


def _require_public_addresses(addresses: tuple[str, ...]) -> tuple[str, ...]:
    if not addresses:
        raise RemoteMediaError("远程媒体主机没有可用地址")
    normalized: list[str] = []
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as error:
            raise RemoteMediaError(f"DNS 返回了无效地址: {raw_address}") from error
        if not (
            address.is_global
            and not address.is_private
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_multicast
            and not address.is_reserved
            and not address.is_unspecified
        ):
            raise RemoteMediaError(f"远程媒体主机解析到非公网地址: {address}")
        normalized.append(address.compressed)
    return tuple(normalized)


def _normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        raise RemoteMediaError("远程媒体 URL 不能为空")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise RemoteMediaError("远程媒体只允许 http/https")
    if parts.username is not None or parts.password is not None:
        raise RemoteMediaError("远程媒体 URL 不允许用户凭据")
    host = _ascii_hostname(parts.hostname)
    try:
        port = parts.port
    except ValueError as error:
        raise RemoteMediaError("远程媒体 URL 端口无效") from error
    authority_host = f"[{host}]" if ":" in host else host
    authority = authority_host if port is None else f"{authority_host}:{port}"
    path = parts.path or "/"
    return urlunsplit((parts.scheme, authority, path, parts.query, ""))


def _ascii_hostname(hostname: str | None) -> str:
    if hostname is None:
        raise RemoteMediaError("远程媒体 URL 缺少主机名")
    try:
        return hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise RemoteMediaError("远程媒体主机名无效") from error


def _host_header(host: str, port: int, scheme: str) -> str:
    authority = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    return authority if port == default_port else f"{authority}:{port}"


def _single_header(
    headers: list[tuple[bytes, bytes]],
    name: bytes,
    *,
    required: bool = False,
) -> str | None:
    values = [value for key, value in headers if key.lower() == name]
    if len(values) > 1:
        raise RemoteMediaError(f"远程媒体响应重复头字段: {name.decode('ascii')}")
    if not values:
        if required:
            raise RemoteMediaError(f"远程媒体响应缺少头字段: {name.decode('ascii')}")
        return None
    try:
        value = values[0].decode("latin-1").strip()
    except UnicodeDecodeError as error:
        raise RemoteMediaError(f"远程媒体响应头无法解码: {name!r}") from error
    if not value or "\r" in value or "\n" in value or "\x00" in value:
        raise RemoteMediaError(f"远程媒体响应头无效: {name.decode('ascii')}")
    return value


def _content_length(headers: list[tuple[bytes, bytes]]) -> int | None:
    value = _single_header(headers, b"content-length")
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise RemoteMediaError("远程媒体 Content-Length 无效")
    return int(value)


def _content_type(headers: list[tuple[bytes, bytes]]) -> str:
    value = _single_header(headers, b"content-type")
    if value is None:
        return _DEFAULT_CONTENT_TYPE
    media_type = value.split(";", 1)[0].strip().lower()
    if not _CONTENT_TYPE_PATTERN.fullmatch(media_type):
        raise RemoteMediaError("远程媒体 Content-Type 无效")
    return media_type


def _response_filename(headers: list[tuple[bytes, bytes]], source_url: str) -> str:
    disposition = _single_header(headers, b"content-disposition")
    candidate: str | None = None
    if disposition is not None:
        message = Message()
        message["content-disposition"] = disposition
        candidate = message.get_filename()
    if not candidate:
        candidate = unquote(urlsplit(source_url).path.rsplit("/", 1)[-1])
    return _safe_basename(candidate or _DEFAULT_FILENAME)


def _safe_basename(value: str) -> str:
    candidate = value.replace("\\", "/").rsplit("/", 1)[-1].strip().strip(".")
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        raise RemoteMediaError("远程媒体文件名包含控制字符")
    if not candidate:
        return _DEFAULT_FILENAME
    encoded = candidate.encode("utf-8")
    if len(encoded) <= 240:
        return candidate
    shortened = encoded[:240]
    while True:
        try:
            return shortened.decode("utf-8")
        except UnicodeDecodeError:
            shortened = shortened[:-1]
