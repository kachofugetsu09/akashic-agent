"""App-server 端点地址解析；客户端命令可用，不依赖服务端实现。"""
from __future__ import annotations

import ipaddress
import os
import zlib
from pathlib import Path


def parse_loopback_tcp(endpoint: str) -> tuple[str, int] | None:
    if endpoint.startswith("/") or endpoint.count(":") != 1:
        return None
    host, raw_port = endpoint.rsplit(":", 1)
    try:
        address = ipaddress.ip_address(host)
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"无效 app-server TCP endpoint: {endpoint}") from exc
    if not address.is_loopback:
        raise ValueError(f"app-server TCP 只允许 loopback: {endpoint}")
    if not 0 <= port <= 65535:
        raise ValueError(f"app-server TCP 端口无效: {port}")
    return host, port


def is_tcp_endpoint(endpoint: str) -> bool:
    return parse_loopback_tcp(endpoint) is not None


def resolve_endpoint(value: str, workspace: Path) -> str:
    """保留显式端点；Windows 按原路径稳定派生 loopback 端口。"""
    text = value.strip()
    if os.name != "nt":
        return text or str(workspace / "akashic.sock")
    host, separator, port = text.rpartition(":")
    if separator and host:
        try:
            int(port)
        except ValueError:
            pass
        else:
            return text
    seed = zlib.crc32((text or str(workspace)).encode("utf-8")) % 20000
    return f"127.0.0.1:{20000 + seed}"
